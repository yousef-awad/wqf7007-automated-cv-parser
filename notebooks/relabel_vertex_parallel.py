"""
Parallel Vertex AI Gemini relabeling task.

Default behavior matches the current relabeling plan:
  - use Vertex AI, not AI Studio or OpenRouter
  - model: gemini-3.1-flash-lite
  - thinking level: LOW
  - annotate 100% of each split deterministically
  - annotate requests in parallel while enforcing one global request rate
  - checkpoint each split so the task can resume
  - ask Gemini for exact text spans in section-aware chunks
  - resolve spans to offsets locally and reject obvious noisy one-token labels before BIO tagging

Dry run, no Vertex calls:
  python notebooks/relabel_vertex_parallel.py --dry-run

Full task:
  python notebooks/relabel_vertex_parallel.py --workers 3 --requests-per-minute 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CATEGORY_CSV = PROJECT_ROOT / "data" / "raw" / "resume_original_category_subset.csv"
RAW_PARQUET_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "resume_original_train.parquet",
    PROJECT_ROOT / "data" / "raw" / "train-00000-of-00001.parquet",
]
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "resume_bio_annotated_dataset4.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "processed" / "annotation_checkpoints"

CATEGORY_QUESTION = "What job category does this resume best fit?"
CAT_RE = re.compile(r"best fits the\s+([A-Z\-]+)\s+category", re.IGNORECASE)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
MIN_CLASS_SAMPLES = 3

LABEL_LIST = [
    "O",
    "B-JOB_TITLE",
    "I-JOB_TITLE",
    "B-SKILL",
    "I-SKILL",
    "B-EDUCATION",
    "I-EDUCATION",
]

SYSTEM_PROMPT = (
    "You are an expert CV parser creating token-level NER training labels. "
    "Extract useful resume entities that appear exactly in the supplied resume chunk. "
    "Always respond with valid JSON only, no explanation and no markdown fences."
)

USER_TEMPLATE = """Extract named entities from this resume chunk.

Return ONLY this JSON structure:
{{
  "entities": [
    {{"label": "JOB_TITLE", "text": "..."}},
    {{"label": "SKILL", "text": "..."}},
    {{"label": "EDUCATION", "text": "..."}}
  ]
}}

Rules:
- text MUST be copied exactly from this chunk.
- Use only these labels: JOB_TITLE, SKILL, EDUCATION.
- JOB_TITLE: job roles/positions only, not industries or departments.
- SKILL: concrete tools, technologies, methods, certifications, domain skills, professional competencies, software, languages, and well-formed abilities.
- Include common resume skills when they appear as actual competencies, for example customer service, sales, inventory control, research, quality assurance, financial analysis, communication skills, leadership, marketing, documentation, billing, accounting, policies, client relations, or project management.
- Do not label a vague one-word noun as SKILL when it has no competency meaning in context, such as clients, meetings, office, focus, materials, or business.
- EDUCATION: degrees, institutions, majors/minors, formal certificates, or fields of study in education/training/certification context.
- Do not label every mention of a subject word as EDUCATION; only label it when it is part of an education credential/context.
- Copy spans EXACTLY as they appear in the text.
- Assign each exact span to only one label. If ambiguous, prefer EDUCATION only in education/training/certification context.

Resume chunk:
{chunk_text}"""

ALLOWED_ENTITY_LABELS = {"JOB_TITLE", "SKILL", "EDUCATION"}

ENTITY_LABEL_PRIORITY = {
    "EDUCATION": 0,
    "JOB_TITLE": 1,
    "SKILL": 2,
}

ENTITY_LABEL_MAP = {
    "job_titles": "JOB_TITLE",
    "skills": "SKILL",
    "education": "EDUCATION",
}

GENERIC_SINGLE_TOKEN_SKILLS_STRICT = {
    "accounting",
    "benefits",
    "billing",
    "budget",
    "business",
    "client",
    "clients",
    "contracts",
    "credit",
    "documentation",
    "financial",
    "focus",
    "inventory",
    "managing",
    "marketing",
    "materials",
    "meetings",
    "office",
    "personnel",
    "policies",
    "processes",
    "quality",
    "research",
    "retail",
}

GENERIC_SINGLE_TOKEN_SKILLS_RELAXED = {
    "business",
    "client",
    "clients",
    "focus",
    "materials",
    "meetings",
    "office",
}

KNOWN_SINGLE_TOKEN_SKILLS = {
    "access",
    "ajax",
    "aws",
    "azure",
    "c",
    "c#",
    "c++",
    "css",
    "excel",
    "git",
    "github",
    "html",
    "java",
    "javascript",
    "jira",
    "jquery",
    "linux",
    "mysql",
    "oracle",
    "outlook",
    "powerpoint",
    "python",
    "quickbooks",
    "react",
    "sap",
    "sql",
    "tableau",
    "unix",
    "vb",
    "word",
}

SECTION_HEADER_RE = re.compile(
    r"(?im)^\s*(summary|profile|skills?|technical skills|core competencies|"
    r"professional experience|work experience|employment|education|certifications?|"
    r"licenses?|training|projects?|highlights?|accomplishments?)\s*:?\s*$"
)
EDUCATION_CONTEXT_RE = re.compile(
    r"\b(education|degree|university|college|school|institute|academy|"
    r"certification|certificate|certified|training|coursework|major|minor|"
    r"graduate|graduated|postgraduate|"
    r"bachelor|bachelors|master|masters|mba|phd|associate|diploma|gpa)\b",
    re.IGNORECASE,
)
DEGREE_OR_CERT_RE = re.compile(
    r"\b(b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|mba|phd|bachelor|bachelors|"
    r"master|masters|graduate|postgraduate|associate|degree|diploma|certificate|certification|"
    r"certified|license|licensed|gpa)\b",
    re.IGNORECASE,
)


@dataclass
class RunStats:
    calls_started: int = 0
    calls_ok: int = 0
    rate_limits: int = 0
    retries: int = 0
    failures: int = 0
    entity_total: int = 0
    entity_misses: int = 0


class AdaptiveRateLimiter:
    """Global start-rate limiter shared by all worker threads."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.base_interval = 60.0 / requests_per_minute
        self.penalty = 1.0
        self.next_at = 0.0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + (self.base_interval * self.penalty)
        if wait:
            time.sleep(wait)

    def record_rate_limit(self) -> None:
        with self.lock:
            self.penalty = min(self.penalty * 1.75, 10.0)

    def record_success(self) -> None:
        with self.lock:
            if self.penalty > 1.0:
                self.penalty = max(1.0, self.penalty * 0.98)

    @property
    def effective_rpm(self) -> float:
        with self.lock:
            return 60.0 / (self.base_interval * self.penalty)


def clean_resume(text: str) -> str:
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def strip_category_prefix(text: str, category: str) -> str:
    """Remove the question+category header prepended to resume_text in the raw CSV."""
    if not text.startswith(CATEGORY_QUESTION):
        return text
    after_q = text[len(CATEGORY_QUESTION):].strip()
    if after_q.upper().startswith(category.upper()):
        after_q = after_q[len(category):].strip()
    return after_q


def load_local_raw_dataset() -> pd.DataFrame:
    if RAW_CATEGORY_CSV.exists():
        df = pd.read_csv(RAW_CATEGORY_CSV)
        if "row_idx" not in df.columns:
            df["row_idx"] = df.index
        df = df[["row_idx", "resume_text", "job_category"]].copy()
    else:
        parquet_path = next((p for p in RAW_PARQUET_CANDIDATES if p.exists()), None)
        if parquet_path is None:
            raise FileNotFoundError(
                "Missing local raw dataset. Expected data/raw/resume_original_category_subset.csv "
                "or one of the original parquet files."
            )
        df = extract_category_rows_from_parquet(parquet_path)

    df["resume_text"] = df["resume_text"].astype(str).apply(clean_resume)
    df["job_category"] = df["job_category"].astype(str).str.upper()
    df["resume_text"] = df.apply(
        lambda row: strip_category_prefix(row["resume_text"], row["job_category"]), axis=1
    )
    df = df[df["resume_text"].str.len() > 0].reset_index(drop=True)
    return df


def extract_category_rows_from_parquet(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    raw_df = table.to_pandas()
    records: list[dict[str, Any]] = []
    for row_idx, row in raw_df.iterrows():
        messages = row.get("messages")
        user_msg = get_message_content(messages, "user")
        assistant_msg = get_message_content(messages, "assistant")
        lines = user_msg.strip().splitlines()
        if not lines or lines[0] != CATEGORY_QUESTION:
            continue
        match = CAT_RE.search(assistant_msg)
        if not match:
            continue
        parts = re.split(r"resume:", user_msg, maxsplit=1, flags=re.IGNORECASE)
        resume = parts[1].strip() if len(parts) > 1 else user_msg.strip()
        records.append(
            {
                "row_idx": row_idx,
                "resume_text": resume,
                "job_category": match.group(1).upper(),
            }
        )
    return pd.DataFrame(records)


def get_message_content(messages: Any, role: str) -> str:
    if messages is None:
        return ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == role:
            return str(msg.get("content", ""))
    return ""


def split_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = df_raw["job_category"].value_counts()
    small_classes = counts[counts < MIN_CLASS_SAMPLES].index.tolist()
    df = df_raw.copy()
    df["_strat_label"] = df["job_category"].apply(
        lambda c: "OTHER" if c in small_classes else c
    )

    try:
        from sklearn.model_selection import train_test_split

        df_trainval, df_test = train_test_split(
            df,
            test_size=TEST_RATIO,
            stratify=df["_strat_label"],
            random_state=RANDOM_SEED,
        )
        val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        df_train, df_val = train_test_split(
            df_trainval,
            test_size=val_size_adjusted,
            stratify=df_trainval["_strat_label"],
            random_state=RANDOM_SEED,
        )
    except ModuleNotFoundError:
        print("scikit-learn not installed; using pandas stratified split fallback.")
        df_trainval, df_test = pandas_stratified_holdout(
            df, "_strat_label", test_size=TEST_RATIO, seed=RANDOM_SEED
        )
        val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        df_train, df_val = pandas_stratified_holdout(
            df_trainval, "_strat_label", test_size=val_size_adjusted, seed=RANDOM_SEED
        )

    splits = []
    for split_df in (df_train, df_val, df_test):
        split_df = split_df.drop(columns=["_strat_label"]).reset_index(drop=True)
        splits.append(split_df)
    return splits[0], splits[1], splits[2]


def pandas_stratified_holdout(
    df: pd.DataFrame,
    strat_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for label, group in df.groupby(strat_col, sort=False):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_test = int(round(len(shuffled) * test_size))
        if len(shuffled) > 1:
            n_test = min(max(1, n_test), len(shuffled) - 1)
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def sample_split(df: pd.DataFrame, sample_frac: float, split_name: str) -> pd.DataFrame:
    if sample_frac >= 1.0:
        return df.copy().reset_index(drop=True)
    if sample_frac <= 0.0:
        raise ValueError("sample_frac must be greater than 0")
    n = max(1, int(len(df) * sample_frac))
    return df.sample(n=n, random_state=RANDOM_SEED).sort_index().reset_index(drop=True)


def make_client(args: argparse.Namespace) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Missing google-genai. Install it before running the Vertex task: "
            "python -m pip install google-genai"
        ) from exc

    return genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location,
    )


def ensure_vertex_dependencies() -> None:
    try:
        from google import genai  # noqa: F401
        from google.genai import errors as genai_errors  # noqa: F401
        from google.genai import types  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Missing google-genai. Install it before running the Vertex task: "
            "python -m pip install google-genai"
        ) from exc


def split_text_chunks(text: str, max_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    """Split a resume into offset-preserving chunks, preferring section/sentence boundaries."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap_chars = max(0, min(overlap_chars, max_chars // 3))
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        hard_end = min(text_len, start + max_chars)
        end = hard_end
        if hard_end < text_len:
            boundary_candidates = [
                m.start()
                for m in SECTION_HEADER_RE.finditer(text, start + max_chars // 3, hard_end)
            ]
            if boundary_candidates:
                end = boundary_candidates[-1]
            else:
                sentence_break = text.rfind(". ", start + max_chars // 2, hard_end)
                newline_break = text.rfind("\n", start + max_chars // 2, hard_end)
                space_break = text.rfind(" ", start + max_chars // 2, hard_end)
                end = max(sentence_break + 1, newline_break, space_break)
                if end <= start:
                    end = hard_end
        chunk_text = text[start:end].strip()
        leading_ws = len(text[start:end]) - len(text[start:end].lstrip())
        if chunk_text:
            chunks.append({"start": start + leading_ws, "text": chunk_text})
        if end >= text_len:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def extract_response_json(text_content: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text_content)
    except json.JSONDecodeError:
        start = text_content.find("{")
        end = text_content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return extract_entity_objects(text_content)
        try:
            parsed = json.loads(text_content[start : end + 1])
        except json.JSONDecodeError:
            return extract_entity_objects(text_content)

    if isinstance(parsed, list):
        values = parsed
    elif isinstance(parsed, dict):
        values = parsed.get("entities", [])
    else:
        return []
    if not isinstance(values, list):
        return []

    entities: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        label = str(value.get("label", "")).strip().upper().replace("-", "_")
        text = str(value.get("text", "")).strip()
        if label in ALLOWED_ENTITY_LABELS and text:
            entities.append({"label": label, "text": text})
    return entities


def extract_entity_objects(text_content: str) -> list[dict[str, Any]]:
    """Best-effort fallback for flash-lite responses with missing commas.

    The API is already constrained to JSON, but smaller models sometimes emit a
    valid sequence of entity objects inside an invalid wrapper. Salvage only
    objects with explicit label/text fields; anything else is ignored.
    """
    entities: list[dict[str, Any]] = []
    pattern = re.compile(
        r'\{\s*"label"\s*:\s*"(?P<label>[^"]+)"\s*,\s*"text"\s*:\s*"(?P<text>(?:\\.|[^"])*)"\s*\}',
        re.DOTALL,
    )
    for match in pattern.finditer(text_content):
        label = match.group("label").strip().upper().replace("-", "_")
        text = bytes(match.group("text"), "utf-8").decode("unicode_escape").strip()
        if label in ALLOWED_ENTITY_LABELS and text:
            entities.append({"label": label, "text": text})
    return entities


def call_gemini_chunk(
    chunk_text: str,
    chunk_start: int,
    args: argparse.Namespace,
    rate_limiter: AdaptiveRateLimiter,
    stats: RunStats,
    stats_lock: threading.Lock,
    local_state: threading.local,
) -> list[dict[str, Any]] | None:
    from google.genai import errors as genai_errors
    from google.genai import types

    if not hasattr(local_state, "client"):
        local_state.client = make_client(args)

    for attempt in range(args.max_retries):
        try:
            rate_limiter.acquire()
            with stats_lock:
                stats.calls_started += 1
            resp = local_state.client.models.generate_content(
                model=args.model,
                contents=USER_TEMPLATE.format(
                    chunk_start=chunk_start,
                    chunk_text=chunk_text,
                ),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=args.max_output_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    ),
                ),
            )
            parts = (resp.candidates[0].content.parts or []) if resp.candidates else []
            text_content = "".join(
                part.text
                for part in parts
                if hasattr(part, "text") and part.text
            )
            entities = extract_response_json(text_content)
            rate_limiter.record_success()
            with stats_lock:
                stats.calls_ok += 1
            return entities
        except genai_errors.ClientError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is None and ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)):
                status_code = 429
            retryable = status_code in {429, 500, 502, 503, 504}
            if status_code == 429:
                rate_limiter.record_rate_limit()
                with stats_lock:
                    stats.rate_limits += 1
            if not retryable:
                print(f"API error {status_code}: {exc}", flush=True)
                break
            with stats_lock:
                stats.retries += 1
            wait = min(args.max_backoff_seconds, (2**attempt) + random.random())
            print(
                f"Retryable API error {status_code}; sleeping {wait:.1f}s "
                f"(effective rate {rate_limiter.effective_rpm:.1f} RPM)",
                flush=True,
            )
            time.sleep(wait)
        except Exception as exc:
            with stats_lock:
                stats.retries += 1
            wait = min(args.max_backoff_seconds, (2**attempt) + random.random())
            print(f"Call error: {exc}; sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)

    with stats_lock:
        stats.failures += 1
    return None


def token_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def clean_match_token(token: str) -> str:
    return token.lower().strip(".,;:()\"'[]{}<>-*")


def is_generic_skill(span_text: str, skill_policy: str) -> bool:
    cleaned = clean_match_token(span_text)
    if " " in span_text.strip():
        return False
    if cleaned in KNOWN_SINGLE_TOKEN_SKILLS:
        return False
    generic_skills = (
        GENERIC_SINGLE_TOKEN_SKILLS_STRICT
        if skill_policy == "strict"
        else GENERIC_SINGLE_TOKEN_SKILLS_RELAXED
    )
    if cleaned in generic_skills:
        return True
    if len(cleaned) <= 2 and not span_text.strip().isupper():
        return True
    return False


def has_education_context(text: str, start: int, end: int, span_text: str) -> bool:
    if DEGREE_OR_CERT_RE.search(span_text):
        return True
    window = text[max(0, start - 220) : min(len(text), end + 160)]
    return bool(EDUCATION_CONTEXT_RE.search(window))


def find_span_in_chunk(chunk_text: str, span_text: str) -> tuple[int, int] | None:
    span_text = span_text.strip()
    if not span_text:
        return None

    start = chunk_text.find(span_text)
    if start >= 0:
        return start, start + len(span_text)

    if span_text.split():
        pattern = r"\s+".join(re.escape(part) for part in span_text.split())
        for flags in (0, re.IGNORECASE):
            match = re.search(pattern, chunk_text, flags=flags)
            if match:
                return match.start(), match.end()

    span_tokens = [clean_match_token(token) for token in span_text.split()]
    span_tokens = [token for token in span_tokens if token]
    if not span_tokens:
        return None

    chunk_offsets = token_offsets(chunk_text)
    chunk_norm = [clean_match_token(token) for token, _, _ in chunk_offsets]
    width = len(span_tokens)
    for idx in range(len(chunk_norm) - width + 1):
        if chunk_norm[idx : idx + width] == span_tokens:
            return chunk_offsets[idx][1], chunk_offsets[idx + width - 1][2]
    return None


def resolve_chunk_entity(
    full_text: str,
    chunk_start: int,
    chunk_text: str,
    entity: dict[str, Any],
    skill_policy: str,
) -> tuple[dict[str, Any] | None, str | None]:
    label = entity["label"]
    text_value = entity["text"]
    span = find_span_in_chunk(chunk_text, text_value)
    if span is None:
        return None, "span_not_found_in_chunk"

    start, end = span
    global_start = chunk_start + start
    global_end = chunk_start + end
    span_text = full_text[global_start:global_end]

    if label == "SKILL" and is_generic_skill(span_text, skill_policy):
        return None, "generic_single_token_skill"
    if label == "EDUCATION" and not has_education_context(full_text, global_start, global_end, span_text):
        return None, "education_without_context"

    return {
        "label": label,
        "text": span_text,
        "source_text": text_value,
        "start": global_start,
        "end": global_end,
    }, None

def spans_to_bio(
    text: str,
    entities: list[dict[str, Any]],
) -> tuple[list[str], list[str], int]:
    offsets = token_offsets(text)
    tokens = [token for token, _, _ in offsets]
    tags = ["O"] * len(tokens)
    misses = 0
    candidates = []
    seen = set()

    for entity in entities:
        key = (entity["start"], entity["end"], entity["label"])
        if key in seen:
            continue
        seen.add(key)

        token_indexes = [
            idx
            for idx, (_, tok_start, tok_end) in enumerate(offsets)
            if tok_start >= entity["start"] and tok_end <= entity["end"]
        ]
        if (
            not token_indexes
            or offsets[token_indexes[0]][1] != entity["start"]
            or offsets[token_indexes[-1]][2] != entity["end"]
        ):
            misses += 1
            continue
        candidates.append(
            {
                "token_start": token_indexes[0],
                "token_end": token_indexes[-1] + 1,
                "char_start": entity["start"],
                "char_end": entity["end"],
                "label": entity["label"],
            }
        )

    candidates.sort(
        key=lambda c: (
            ENTITY_LABEL_PRIORITY[c["label"]],
            -(c["token_end"] - c["token_start"]),
            c["token_start"],
        )
    )

    occupied = [False] * len(tokens)
    for candidate in candidates:
        start = candidate["token_start"]
        end = candidate["token_end"]
        label = candidate["label"]
        if any(occupied[start:end]):
            misses += 1
            continue
        tags[start] = f"B-{label}"
        for idx in range(start + 1, end):
            tags[idx] = f"I-{label}"
        for idx in range(start, end):
            occupied[idx] = True

    return tokens, tags, misses


def annotate_text(
    text: str,
    args: argparse.Namespace,
    rate_limiter: AdaptiveRateLimiter,
    stats: RunStats,
    stats_lock: threading.Lock,
    local_state: threading.local,
) -> tuple[list[str], list[str], dict[str, Any], int] | None:
    accepted = []
    rejected = []
    raw_count = 0

    for chunk in split_text_chunks(text, args.chunk_chars, args.chunk_overlap):
        raw_entities = call_gemini_chunk(
            chunk["text"],
            chunk["start"],
            args,
            rate_limiter,
            stats,
            stats_lock,
            local_state,
        )
        if raw_entities is None:
            return None
        raw_count += len(raw_entities)
        for entity in raw_entities:
            validated, reason = resolve_chunk_entity(
                text,
                chunk["start"],
                chunk["text"],
                entity,
                args.skill_policy,
            )
            if validated is None:
                rejected.append({"reason": reason, **entity, "chunk_start": chunk["start"]})
            else:
                accepted.append(validated)

    deduped = []
    seen = set()
    for entity in accepted:
        key = (entity["start"], entity["end"], entity["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    accepted = deduped

    tokens, tags, misses = spans_to_bio(text, accepted)
    raw_payload = {
        "entities": accepted,
        "rejected": rejected,
        "chunk_count": len(split_text_chunks(text, args.chunk_chars, args.chunk_overlap)),
    }
    return tokens, tags, raw_payload, (raw_count - len(accepted)) + misses


def row_to_annotation(
    item: tuple[int, pd.Series],
    split_name: str,
    args: argparse.Namespace,
    rate_limiter: AdaptiveRateLimiter,
    stats: RunStats,
    stats_lock: threading.Lock,
    local_state: threading.local,
) -> dict[str, Any] | None:
    ordinal, row = item
    annotated = annotate_text(
        row["resume_text"],
        args,
        rate_limiter,
        stats,
        stats_lock,
        local_state,
    )
    if annotated is None:
        return None
    tokens, tags, entities, misses = annotated
    entity_total = len(entities["entities"]) + len(entities["rejected"])
    with stats_lock:
        stats.entity_total += entity_total
        stats.entity_misses += misses
    return {
        "split": split_name,
        "source_row_idx": row["row_idx"],
        "job_category": row["job_category"],
        "resume_text": row["resume_text"],
        "tokens": json.dumps(tokens),
        "bio_tags": json.dumps(tags),
        "entities_raw": json.dumps(entities),
        "_ordinal": ordinal,
    }


def checkpoint_path(split_name: str, sample_frac: float, dataset_name: str) -> Path:
    percent = int(sample_frac * 100)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("_")
    return CHECKPOINT_DIR / f"vertex_{safe_name}_{percent:03d}pct_{split_name}.csv"


def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded checkpoint {path.name}: {len(df):,} rows", flush=True)
    return df


def save_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("_ordinal")
    df.to_csv(path, index=False)


def annotate_split(
    df: pd.DataFrame,
    split_name: str,
    args: argparse.Namespace,
    rate_limiter: AdaptiveRateLimiter,
    stats: RunStats,
    stats_lock: threading.Lock,
) -> pd.DataFrame:
    path = checkpoint_path(split_name, args.sample_frac, args.dataset_name)
    checkpoint_df = load_checkpoint(path)
    completed = set(checkpoint_df["source_row_idx"].tolist()) if not checkpoint_df.empty else set()
    rows = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []
    pending_df = df[~df["row_idx"].isin(completed)].reset_index(drop=True)

    total = len(df)
    print(
        f"\nAnnotating {split_name}: {len(pending_df):,} pending / {total:,} total "
        f"with {args.workers} workers, target {args.requests_per_minute:.1f} RPM",
        flush=True,
    )
    if pending_df.empty:
        return checkpoint_df

    local_state = threading.local()
    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                row_to_annotation,
                (idx, row),
                split_name,
                args,
                rate_limiter,
                stats,
                stats_lock,
                local_state,
            ): idx
            for idx, row in pending_df.iterrows()
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)
            completed_since_save += 1
            done = len(rows)
            if completed_since_save >= args.checkpoint_every or done == total:
                save_checkpoint(path, rows)
                completed_since_save = 0
            if done % args.progress_every == 0 or done == total:
                print(
                    f"{split_name}: {done:,}/{total:,} saved "
                    f"(rate-limit events: {stats.rate_limits})",
                    flush=True,
                )

    save_checkpoint(path, rows)
    final_df = pd.DataFrame(rows)
    if final_df.empty:
        return final_df
    final_df = final_df.sort_values("_ordinal").drop(columns=["_ordinal"], errors="ignore")
    print(f"{split_name} complete: {len(final_df):,}/{total:,} annotated", flush=True)
    return final_df


def validate_annotation_csv(df: pd.DataFrame) -> None:
    required = {"split", "job_category", "resume_text", "tokens", "bio_tags", "entities_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing output columns: {sorted(missing)}")

    bad_lengths = []
    bad_tags = set()
    allowed = set(LABEL_LIST)
    for idx, row in df.iterrows():
        tokens = json.loads(row["tokens"])
        tags = json.loads(row["bio_tags"])
        if len(tokens) != len(tags):
            bad_lengths.append((idx, len(tokens), len(tags)))
        bad_tags.update(set(tags) - allowed)
    if bad_lengths:
        raise ValueError(f"Token/BIO length mismatches: {bad_lengths[:5]}")
    if bad_tags:
        raise ValueError(f"Unexpected BIO tags: {sorted(bad_tags)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel Vertex AI Gemini relabeling.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--project", default="project-12ae9020-458c-4247-8dd")
    parser.add_argument("--location", default="global")
    parser.add_argument("--dataset-name", default="dataset4_gemini31_lite_relaxed_skills")
    parser.add_argument("--skill-policy", choices=["strict", "relaxed"], default="relaxed")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--requests-per-minute", type=float, default=20.0)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-backoff-seconds", type=float, default=90.0)
    parser.add_argument("--chunk-chars", type=int, default=6000)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path, default=OUT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if args.output.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{args.output} already exists. Use --overwrite to replace it.")

    df_raw = load_local_raw_dataset()
    df_train, df_val, df_test = split_dataset(df_raw)
    df_train = sample_split(df_train, args.sample_frac, "train")
    df_val = sample_split(df_val, args.sample_frac, "validation")
    df_test = sample_split(df_test, args.sample_frac, "test")

    print(f"Model     : {args.model}")
    print("Thinking  : LOW")
    print(f"Dataset   : {args.dataset_name}")
    print(f"Skill rule: {args.skill_policy}")
    print(f"Project   : {args.project}")
    print(f"Location  : {args.location}")
    print(f"Input rows: {len(df_raw):,}")
    print(f"Sample    : {args.sample_frac:.0%} per split")
    print(f"Splits    : train={len(df_train):,} validation={len(df_val):,} test={len(df_test):,}")
    print(f"Chunking  : {args.chunk_chars:,} chars, {args.chunk_overlap:,} overlap")
    print(f"Output    : {args.output}")

    if args.dry_run:
        print("Dry run complete. No Vertex calls made.")
        return 0

    ensure_vertex_dependencies()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    rate_limiter = AdaptiveRateLimiter(args.requests_per_minute)
    stats = RunStats()
    stats_lock = threading.Lock()

    ann_train = annotate_split(df_train, "train", args, rate_limiter, stats, stats_lock)
    ann_val = annotate_split(df_val, "validation", args, rate_limiter, stats, stats_lock)
    ann_test = annotate_split(df_test, "test", args, rate_limiter, stats, stats_lock)

    df_all = pd.concat([ann_train, ann_val, ann_test], ignore_index=True)
    df_all = df_all.drop(columns=["_ordinal"], errors="ignore")
    validate_annotation_csv(df_all)
    df_all.to_csv(args.output, index=False)

    print("\nSaved annotated CSV")
    print(f"Rows      : {len(df_all):,}")
    print(f"Output    : {args.output}")
    match_rate = 1.0 - stats.entity_misses / max(1, stats.entity_total)
    print(
        "Stats     : "
        f"started={stats.calls_started:,}, ok={stats.calls_ok:,}, "
        f"retries={stats.retries:,}, rate_limits={stats.rate_limits:,}, failures={stats.failures:,}"
    )
    print(
        f"Entities  : {stats.entity_total:,} extracted, "
        f"{stats.entity_misses:,} unmatched ({match_rate:.1%} match rate)"
    )
    if stats.entity_misses > 0:
        print("  Unmatched entities are spans the LLM returned that could not be found verbatim in the resume text.")
    if stats.rate_limits:
        print("Rate limiting occurred; rerun with lower --requests-per-minute if needed.")
    else:
        print("No rate-limit responses observed.")
    return 0 if stats.failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
