# Automated CV Parser — Project Summary

## Project Overview

This project builds a Named Entity Recognition (NER) system to automatically extract structured information from resumes/CVs. It uses fine-tuned transformer models (BERT, BERT+CRF) to identify three entity types:

- **JOB_TITLE** — e.g., "Software Engineer", "Data Scientist"
- **SKILL** — e.g., "Python", "Machine Learning", "SQL"
- **EDUCATION** — e.g., "Bachelor of Science", "Computer Science"

---

## Dataset

| Property | Value |
|---|---|
| Source | `yashpwr/resume-ner-training-data` (HuggingFace Hub) |
| Total resumes | 2,483 |
| Job categories | 24 |
| Split | 70% train / 15% val / 15% test |
| Train resumes | 1,726 |
| Val resumes | 371 |
| Test resumes | 371 |
| Tokenization | Sliding window (max_length=512, stride=128) |
| Train chunks | 5,066 |
| Val chunks | 1,099 |
| Test chunks | 1,082 |

### BIO Label Scheme

| ID | Label | Description |
|---|---|---|
| 0 | O | Outside any entity |
| 1 | B-JOB_TITLE | Begin job title |
| 2 | I-JOB_TITLE | Inside job title |
| 3 | B-SKILL | Begin skill |
| 4 | I-SKILL | Inside skill |
| 5 | B-EDUCATION | Begin education |
| 6 | I-EDUCATION | Inside education |

> Note: O-tag rate is ~86–92% — significant class imbalance, especially for EDUCATION.

---

## Models

### 1. BERT NER (`train_bert.ipynb`)
- **Architecture**: `bert-base-uncased` + linear classifier head
- **Framework**: HuggingFace `Trainer` + `DataCollatorForTokenClassification`
- **Hyperparameters**: LR=2e-5, epochs=4, batch=4, warmup_ratio=0.1, weight_decay=0.01
- **Export**: `exported_models/bert-base-uncased-ner/` (HuggingFace format)

### 2. BERT-CRF NER (`train_bert_crf.ipynb`)
- **Architecture**: `bert-base-uncased` + linear classifier + CRF layer (`pytorch-crf`, batch_first=True)
- **Framework**: Custom `BertCRFForNER` (`src/bert_crf_model.py`) + HuggingFace `Trainer`
- **Hyperparameters**: LR=2e-5, epochs=4, batch=4, warmup_ratio=0.1, weight_decay=0.01, dropout=0.1
- **Export**: `exported_models/bert-crf-ner/` (state_dict + tokenizer + config)

### 3. BERT NER + Post-processing (`trainer_02.ipynb`)
- **Architecture**: `bert-base-uncased` + linear classifier head
- **Additional features**:
  - `EarlyStoppingCallback(patience=3)` on chunk-level F1
  - BIO post-processing: `fix_bio_sequence`, `remove_isolated_entities`, `fix_fragmented_entities`
  - Resume-level evaluation via sliding-window chunk merging
- **Hyperparameters**: LR=2e-5, epochs=4 (max), batch=4, warmup_ratio=0.1, weight_decay=0.01

---

## Performance Results

### Current Session (May 2026)

#### Validation Set

| Model | Precision | Recall | F1 |
|---|---|---|---|
| BERT NER | 0.6030 | 0.5581 | **0.5797** |
| BERT-CRF NER | 0.5985 | 0.5469 | 0.5715 |
| BERT + Post-processing | 0.6041 | 0.5516 | 0.5767 |

#### Test Set

| Model | Precision | Recall | F1 | Training Time |
|---|---|---|---|---|
| BERT NER | 0.6206 | 0.5819 | **0.6006** | 9.6 min |
| BERT-CRF NER | 0.6136 | 0.5663 | 0.5890 | 98.5 min |
| BERT + Post-processing | 0.6235 | 0.5780 | **0.5999** | 11.1 min |

### Per-Entity Test Results (Current Session)

#### BERT NER (`train_bert.ipynb`)

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| EDUCATION | 0.349 | 0.438 | 0.389 | 280 |
| JOB_TITLE | 0.690 | 0.693 | 0.692 | 2,517 |
| SKILL | 0.587 | 0.528 | 0.556 | 10,515 |
| **Micro avg** | **0.621** | **0.582** | **0.601** | 13,312 |

#### BERT-CRF NER (`train_bert_crf.ipynb`)

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| EDUCATION | 0.302 | 0.433 | 0.356 | 201 |
| JOB_TITLE | 0.683 | 0.676 | 0.679 | 2,546 |
| SKILL | 0.585 | 0.518 | 0.550 | 10,706 |
| **Micro avg** | **0.614** | **0.566** | **0.589** | 13,453 |

#### BERT + Post-processing (`trainer_02.ipynb`)

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| EDUCATION | 0.372 | 0.429 | 0.398 | 280 |
| JOB_TITLE | 0.697 | 0.683 | 0.690 | 2,517 |
| SKILL | 0.613 | 0.557 | 0.584 | 10,515 |
| **Micro avg** | **0.624** | **0.578** | **0.600** | 13,312 |

### Historical BERT Runs (April 2026)

| Date | LR | Epochs | Val F1 | Test F1 | Training Time |
|---|---|---|---|---|---|
| Apr 26 | 3e-5 | 5 | 0.5902 | **0.6113** | 17.4 min |
| Apr 26 | 5e-5 | 8 | 0.5874 | 0.5967 | 13.3 min |
| Apr 26 | 5e-5 | 8 | 0.5566 | 0.5748 | 24.0 min |
| Apr 26 | 3e-5 | 5 | 0.5883 | 0.5984 | 14.3 min |
| Apr 26 | 2e-5 | 3 | 0.5823 | 0.5946 | 7.0 min |
| Apr 26 | 2e-5 | 4 | 0.5839 | 0.6014 | 14.2 min |

> **Best overall**: BERT NER with LR=3e-5, 5 epochs → **Test F1=0.6113**

---

## Model Export

Trained models are exported to `exported_models/` for deployment on different machines.

### BERT NER (`exported_models/bert-base-uncased-ner/`)

Standard HuggingFace format — load with:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

tokenizer = AutoTokenizer.from_pretrained("exported_models/bert-base-uncased-ner")
model = AutoModelForTokenClassification.from_pretrained("exported_models/bert-base-uncased-ner")
with open("exported_models/bert-base-uncased-ner/label_config.json") as f:
    cfg = json.load(f)
id2label = {int(k): v for k, v in cfg["id2label"].items()}
```

### BERT-CRF NER (`exported_models/bert-crf-ner/`)

Custom architecture — requires `src/bert_crf_model.py`:
```python
import torch, json
from transformers import AutoTokenizer
from src.bert_crf_model import BertCRFForNER

with open("exported_models/bert-crf-ner/label_config.json") as f:
    cfg = json.load(f)
model = BertCRFForNER(num_labels=cfg["num_labels"], model_name=cfg["base_model"], dropout_rate=cfg["dropout_rate"])
model.load_state_dict(torch.load("exported_models/bert-crf-ner/pytorch_model_state_dict.pt", map_location="cpu"))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("exported_models/bert-crf-ner")
id2label = {int(k): v for k, v in cfg["id2label"].items()}
```

**Dependencies**: `pip install transformers pytorch-crf torch`

---

## Key Findings

1. **BERT NER (plain)** achieves the best Test F1 (0.6006 in this session; 0.6113 historically with LR=3e-5, 5 epochs) at the lowest training cost (~10 min).
2. **BERT + Post-processing** marginally improves SKILL F1 (+0.028) and EDUCATION F1 (+0.009) via BIO correction rules.
3. **BERT-CRF** underperforms plain BERT at 10× the training cost (98 min vs 10 min). The CRF layer adds structural constraint but does not compensate for the base model's boundary detection errors on this dataset.
4. **EDUCATION** is the hardest entity across all models (F1 ≈ 0.36–0.40) due to very low support (~200–280 instances) and annotation variability in multi-line education sections.
5. **SKILL** entities are semantically diverse (technical and soft skills) and contextually ambiguous, leading to competing conservative/aggressive prediction behavior (high false negatives and false positives simultaneously).

---

## Suggested Enhancements

### High Impact

1. **Domain-specific base model**
   Use `JobBERT` (`jjzha/jobbert-base-cased`), `LinkedIn-BERT`, or `roberta-base` as the encoder. Domain-adaptive pretraining on job postings and resumes significantly improves entity boundary detection for JOB_TITLE and SKILL.

2. **Focal Loss / Class-weighted cross-entropy**
   EDUCATION has ~15× fewer instances than SKILL. Replacing standard cross-entropy with focal loss (`γ=2`) or inverse-frequency class weights will push the model to pay more attention to EDUCATION tokens.
   ```python
   weights = torch.tensor([0.1, 1.0, 1.0, 0.5, 0.5, 4.0, 4.0])  # O, B/I-JOB, B/I-SKILL, B/I-EDU
   loss_fct = CrossEntropyLoss(weight=weights.to(device))
   ```

3. **Larger base model: `bert-large-uncased` or `roberta-large`**
   Already partially explored in `trainer_02.ipynb` (RoBERTa option is commented out). `roberta-base` often outperforms `bert-base-uncased` on NER with no architecture changes.

4. **Improved hyperparameter search**
   Historical results show LR=3e-5, 5 epochs gave Test F1=0.6113 vs 0.6006 with LR=2e-5, 4 epochs. A systematic grid/random search over LR ∈ {2e-5, 3e-5, 5e-5} × epochs ∈ {3, 5, 7} would likely yield F1 > 0.62.

### Medium Impact

5. **EDUCATION data augmentation**
   Collect or synthesize more EDUCATION examples. Since the dataset has only ~200 education entity instances across 371 val resumes, targeted augmentation (paraphrase existing education sections, use templates) would directly improve EDUCATION F1.

6. **External SKILL lexicon (rule-based post-processing)**
   Build a curated SKILL lexicon (from O*NET, LinkedIn skills taxonomy) and apply exact-match entity injection. This addresses false negatives caused by rare skill names not seen during training.

7. **Ensemble: BERT + BERT-CRF**
   Average token-level probabilities from BERT and BERT-CRF before decoding. Ensembles typically yield F1 improvements of 1–3% over individual models.

8. **Larger sliding window or hierarchical model**
   Use `longformer-base-4096` or `bigbird-roberta-base` to process entire resumes without chunking, eliminating boundary artifacts from sliding-window overlap.

### Lower Impact / Research Direction

9. **Active learning on EDUCATION**
   Identify low-confidence predictions on EDUCATION tokens, send them for re-annotation, and add to training data iteratively.

10. **CRF training optimization**
    Current BERT-CRF uses `attention_mask` as the CRF decode mask (includes CLS/SEP tokens). Excluding CLS/SEP from CRF transitions may reduce noise and speed up training.
