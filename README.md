# WQF7007 Automated CV Parser

NLP pipeline to extract structured candidate information from unstructured resumes — job titles, technical skills, and educational background — framed as a BERT NER task.

**Course:** WQF7007 Natural Language Processing  
**SDG Theme:** Decent Work and Economic Growth (SDG 8)

---

## Repository Layout

```
notebooks/
  04_full_pipeline.ipynb   ← single source of truth for the data pipeline
data/
  processed/
    resume_bio_annotated_full.csv   ← Gemini-annotated BIO labels (Git LFS, ~54 MB)
    resume_ner_hf/                  ← HF DatasetDict (not in git — regenerate, see below)
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
Steps 5–7 (tokenizer alignment → save) complete in ~1 minute and write `data/processed/resume_ner_hf/`.

### 3. Load in your trainer

```python
from datasets import load_from_disk

ds = load_from_disk("data/processed/resume_ner_hf")
# ds["train"], ds["validation"], ds["test"]
```

---

## Dataset Summary

| Split | Resumes | Chunks |
|---|---|---|
| train | 1,726 | 5,066 |
| validation | 371 | 1,099 |
| test | 371 | 1,082 |

Each row is a 512-token sliding-window chunk (stride = 128). Long resumes produce multiple consecutive chunks sharing the same `resume_idx`.

**Fields:** `input_ids`, `attention_mask`, `token_type_ids`, `labels`, `job_category`, `resume_idx`

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
- `data/processed/resume_ner_hf/` is excluded from git (arrow files are large and fully reproducible). Regenerate with Steps 5–7 of the notebook.
- Base model: `bert-base-uncased`. To switch models, change `MODEL_CHECKPOINT` and `STRIDE` in the config cell of the notebook and re-run Steps 5–7.

---

## Team

| Role | Member |
|---|---|
| Project Manager | Raden |
| Data Engineer | Yousef |
| Model Engineers | Pika Junidak, Alyani |
| Evaluation & Ethics | Izzah |
| Documentation | Baraah |
