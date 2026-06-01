# WQF7007 Automated CV Parser

NLP pipeline to extract structured candidate information from unstructured resumes — job titles, technical skills, and educational background — framed as a BERT NER task.

**Course:** WQF7007 Natural Language Processing  
**SDG Theme:** Decent Work and Economic Growth (SDG 8)

---

## ✅ Finalized Labelled Dataset

`data/processed/resume_bio_annotated_full.csv` now holds the **chosen, finalized labelled dataset** — the "dataset 4" relabel produced with **Vertex AI Gemini 3.1 Flash-Lite** (LOW thinking, relaxed SKILL policy, section-aware chunking; see `notebooks/relabel_vertex_parallel.py`). This supersedes all earlier annotation passes and is the dataset to train and evaluate against going forward.

It was selected after a head-to-head against the previous relabel: it is the best dataset on **both** models, mainly by recovering SKILL quality while keeping the EDUCATION / JOB_TITLE gains.

**Resume-level test F1 (same label inventory, ~10.6k spans):**

| Model | Prev relabel | **Finalized (dataset 4)** | EDUCATION | JOB_TITLE | SKILL |
|---|---|---|---|---|---|
| BERT (`bert-base-uncased`) | 0.578 | **0.587** | 0.643 | 0.710 | 0.527 |
| RoBERTa (`roberta-base`) | 0.640 | **0.657** | 0.710 | 0.784 | 0.600 |

RoBERTa (lr 3e-5, 5 epochs) remains the best model at **0.657** test F1.

---

## Repository Layout

```
notebooks/
  04_full_pipeline.ipynb   ← single source of truth for the data pipeline
data/
  processed/
    resume_bio_annotated_full.csv   ← FINALIZED BIO labels, dataset 4 / Vertex Gemini 3.1 Flash-Lite (Git LFS, ~69 MB)
    resume_ner_hf/                  ← BERT HF DatasetDict (not in git — regenerate, see below)
    resume_ner_hf_roberta-base/     ← RoBERTa HF DatasetDict when MODEL_CHECKPOINT="roberta-base"
```

---

## Getting Started (Model Engineers)

### 1. Clone and install

```bash
git lfs install
git clone https://github.com/yousef-awad/wqf7007-automated-cv-parser.git
cd wqf7007-automated-cv-parser
pip install datasets transformers torch pandas scikit-learn
```

### 2. Regenerate the HF dataset

Open `notebooks/04_full_pipeline.ipynb` and run all cells.  
Steps 1–4 (Gemini annotation) will **auto-skip** — the annotated CSV is already present. No API key needed.  
Steps 5–7 (tokenizer alignment → save) complete in ~1 minute and write `data/processed/resume_ner_hf/` for BERT. If `MODEL_CHECKPOINT = "roberta-base"`, they write `data/processed/resume_ner_hf_roberta-base/`.

### 3. Load in your trainer

```python
from datasets import load_from_disk

ds = load_from_disk("data/processed/resume_ner_hf")  # BERT
# ds = load_from_disk("data/processed/resume_ner_hf_roberta-base")  # RoBERTa
# ds["train"], ds["validation"], ds["test"]
```

---

## Dataset Summary

| Split | Resumes | Chunks (BERT) |
|---|---|---|
| train | 1,739 | 5,098 |
| validation | 372 | 1,089 |
| test | 372 | 1,044 |

Each row is a 512-token sliding-window chunk (stride = 128). Long resumes produce multiple consecutive chunks sharing the same `resume_idx`.

**Fields:** `input_ids`, `attention_mask`, `labels`, `job_category`, `resume_idx`; BERT also includes `token_type_ids`. RoBERTa usually does not.

**Label map:**

| ID | Label |
|---|---|
| 0 | O |
| 1 | B-JOB_TITLE |
| 2 | I-JOB_TITLE |
| 3 | B-SKILL |
| 4 | I-SKILL |
| 5 | B-EDUCATION |
| 6 | I-EDUCATION |
| -100 | IGNORE (masked in loss) |

---

## Important Notes

- **Do not delete `resume_bio_annotated_full.csv`** — it was generated via the Gemini API and costs money to reproduce.
- `data/processed/resume_ner_hf*/` is excluded from git (arrow files are large and fully reproducible). Regenerate with Steps 5–7 of the notebook.
- Base model: `bert-base-uncased`. To build RoBERTa inputs, set `MODEL_CHECKPOINT = "roberta-base"` in the config cell and re-run Steps 5–7. The notebook automatically uses `add_prefix_space=True` and writes a separate RoBERTa output folder.

---

## Team

| Role | Member |
|---|---|
| Project Manager | Raden |
| Data Engineer | Yousef |
| Model Engineers | Pika Junidak, Alyani |
| Evaluation & Ethics | Izzah |
| Documentation | Baraah |
