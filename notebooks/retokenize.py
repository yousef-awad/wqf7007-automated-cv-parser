"""
Regenerate a HuggingFace DatasetDict from the existing BIO-annotated CSV.
Skips annotation entirely — just tokenizer alignment + save.

Usage:
    python retokenize.py bert       -> data/processed/resume_ner_hf/
    python retokenize.py roberta    -> data/processed/resume_ner_hf_roberta-base/
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.append("..")

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_FAMILY = sys.argv[1].lower() if len(sys.argv) > 1 else "roberta"

if MODEL_FAMILY == "bert":
    MODEL_CHECKPOINT = "bert-base-uncased"
else:
    MODEL_CHECKPOINT = "roberta-base"

MAX_LENGTH  = 512
STRIDE      = 128
IGNORE_IDX  = -100

LABEL_LIST = ["O", "B-JOB_TITLE", "I-JOB_TITLE", "B-SKILL", "I-SKILL", "B-EDUCATION", "I-EDUCATION"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}

MODEL_OUTPUT_SUFFIX = "" if MODEL_CHECKPOINT == "bert-base-uncased" else f"_{MODEL_CHECKPOINT}"
OUT_HF  = Path(f"../data/processed/resume_ner_hf{MODEL_OUTPUT_SUFFIX}")
IN_CSV  = Path("../data/processed/resume_bio_annotated_full.csv")

print(f"Model     : {MODEL_CHECKPOINT}")
print(f"Input CSV : {IN_CSV.resolve()}")
print(f"Output    : {OUT_HF.resolve()}")

# ── Load CSV ──────────────────────────────────────────────────────────────────
df_all   = pd.read_csv(IN_CSV)
df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
df_val   = df_all[df_all["split"] == "val"].reset_index(drop=True)
df_test  = df_all[df_all["split"] == "test"].reset_index(drop=True)
print(f"Loaded CSV: train={len(df_train):,} / val={len(df_val):,} / test={len(df_test):,}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok_kwargs = {"add_prefix_space": True} if "roberta" in MODEL_CHECKPOINT.lower() else {}
tokenizer  = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, **tok_kwargs)
assert tokenizer.is_fast, "Need a fast tokenizer for word_ids()."
print(f"Tokenizer ready: {MODEL_CHECKPOINT}")

# ── Sliding-window alignment ──────────────────────────────────────────────────
def align_labels_sliding_window(tokens, bio_tags):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        padding="max_length",
        return_overflowing_tokens=True,
        return_tensors=None,
    )
    max_word_seen = -1
    chunks = []
    for chunk_idx in range(len(encoding["input_ids"])):
        word_ids       = encoding.word_ids(batch_index=chunk_idx)
        aligned_labels = []
        prev_word_idx  = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(IGNORE_IDX)
            elif word_idx != prev_word_idx:
                if word_idx <= max_word_seen:
                    aligned_labels.append(IGNORE_IDX)
                else:
                    aligned_labels.append(LABEL2ID[bio_tags[word_idx]])
            else:
                aligned_labels.append(IGNORE_IDX)
            prev_word_idx = word_idx
        last_real = max((w for w in word_ids if w is not None), default=max_word_seen)
        max_word_seen = last_real
        chunk = {
            "input_ids":      encoding["input_ids"][chunk_idx],
            "attention_mask": encoding["attention_mask"][chunk_idx],
            "labels":         aligned_labels,
        }
        if "token_type_ids" in encoding:
            chunk["token_type_ids"] = encoding["token_type_ids"][chunk_idx]
        chunks.append(chunk)
    return chunks

# ── Process splits ────────────────────────────────────────────────────────────
def process_split(df, split_name):
    examples = []
    for resume_idx, (_, row) in enumerate(df.iterrows()):
        tokens   = json.loads(row["tokens"])
        bio_tags = json.loads(row["bio_tags"])
        for chunk in align_labels_sliding_window(tokens, bio_tags):
            chunk["job_category"] = row["job_category"]
            chunk["resume_idx"]   = resume_idx
            examples.append(chunk)
    n_chunks = len(examples)
    print(f"  {split_name}: {len(df)} resumes -> {n_chunks} chunks (avg {n_chunks/len(df):.1f}/resume)")
    return Dataset.from_list(examples)

print("\nRunning tokenizer alignment ...")
ds_train = process_split(df_train, "train")
ds_val   = process_split(df_val,   "val")
ds_test  = process_split(df_test,  "test")

dataset_dict = DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test})

# ── Cast features ─────────────────────────────────────────────────────────────
feature_spec = {
    "input_ids":      Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
    "labels":         Sequence(Value("int32")),
    "job_category":   Value("string"),
    "resume_idx":     Value("int32"),
}
if "token_type_ids" in dataset_dict["train"].features:
    feature_spec["token_type_ids"] = Sequence(Value("int8"))

dataset_dict = dataset_dict.cast(Features(feature_spec))

# ── Label distribution check ──────────────────────────────────────────────────
all_labels = [lid for ex in ds_train for lid in ex["labels"] if lid != IGNORE_IDX]
counts     = Counter(all_labels)
total      = sum(counts.values())
print("\nTrain label distribution:")
for lid, cnt in sorted(counts.items()):
    print(f"  {ID2LABEL[lid]:<20} {cnt:>8,}  ({cnt/total:.1%})")

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_HF.parent.mkdir(parents=True, exist_ok=True)
dataset_dict.save_to_disk(str(OUT_HF))
print(f"\nSaved -> {OUT_HF.resolve()}")
print(f"  train={ds_train.num_rows}  val={ds_val.num_rows}  test={ds_test.num_rows}")
print("DONE.")
