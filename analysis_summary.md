# BERT vs RoBERTa NER Analysis Summary

This summary is based on the saved outputs in:

- `notebooks/analysis_bert.ipynb`
- `notebooks/analysis_roberta.ipynb`

Both notebooks evaluate resume-level NER performance for three entity types:

- `JOB_TITLE`
- `SKILL`
- `EDUCATION`

---

## 1. Overall Performance

| Model | Validation F1 | Test F1 |
|---|---:|---:|
| BERT | 0.5659 | 0.5997 |
| RoBERTa | 0.6413 | 0.6590 |

RoBERTa clearly outperforms BERT on both validation and test sets. On the test set, RoBERTa improves over BERT by about **0.0593 absolute F1**.

This suggests that RoBERTa's stronger contextual representations help the model identify resume entities more accurately under the same overall evaluation setup.

---

## 2. Per-Entity Metrics

### BERT Test Results

| Entity | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| JOB_TITLE | 0.7042 | 0.7469 | 0.7250 | 2,209 |
| EDUCATION | 0.6226 | 0.6620 | 0.6417 | 1,799 |
| SKILL | 0.5307 | 0.5623 | 0.5460 | 6,571 |
| Micro avg | 0.5826 | 0.6178 | 0.5997 | 10,579 |
| Macro avg | 0.6192 | 0.6571 | 0.6376 | 10,579 |
| Weighted avg | 0.5825 | 0.6178 | 0.5997 | 10,579 |

### RoBERTa Test Results

| Entity | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| JOB_TITLE | 0.7852 | 0.8026 | 0.7938 | 2,209 |
| EDUCATION | 0.6970 | 0.7276 | 0.7120 | 1,799 |
| SKILL | 0.5834 | 0.6179 | 0.6001 | 6,571 |
| Micro avg | 0.6437 | 0.6751 | 0.6590 | 10,579 |
| Macro avg | 0.6885 | 0.7160 | 0.7020 | 10,579 |
| Weighted avg | 0.6449 | 0.6751 | 0.6596 | 10,579 |

### Per-Entity Interpretation

`JOB_TITLE` is the strongest entity for both models. BERT achieves an F1 of **0.7250**, while RoBERTa improves this to **0.7938**. This suggests that job titles have clearer contextual signals and more consistent wording.

`EDUCATION` is the second-best entity. BERT reaches **0.6417 F1**, while RoBERTa improves to **0.7120 F1**. Education entities are usually structured, but they can still vary between degrees, institutions, certifications, and fields of study.

`SKILL` is the weakest entity for both models. BERT gets **0.5460 F1**, and RoBERTa improves to **0.6001 F1**. Despite the improvement, skill extraction remains the main bottleneck because skill terms are semantically broad and often ambiguous.

---

## 3. Ground Truth Distribution

### BERT Notebook Distribution

| Label | Count |
|---|---:|
| O | 263,035 |
| I-SKILL | 7,371 |
| B-SKILL | 6,571 |
| I-EDUCATION | 5,254 |
| I-JOB_TITLE | 3,727 |
| B-JOB_TITLE | 2,209 |
| B-EDUCATION | 1,799 |

### RoBERTa Notebook Distribution

| Label | Count |
|---|---:|
| O | 263,081 |
| I-SKILL | 7,371 |
| B-SKILL | 6,571 |
| I-EDUCATION | 5,255 |
| I-JOB_TITLE | 3,727 |
| B-JOB_TITLE | 2,209 |
| B-EDUCATION | 1,799 |

The test set is highly imbalanced. Around **90% of tokens are `O`**, meaning most tokens do not belong to any entity.

Among entity starts, the support distribution is:

| Entity | Support | Share of Entity Starts |
|---|---:|---:|
| SKILL | 6,571 | 62.1% |
| JOB_TITLE | 2,209 | 20.9% |
| EDUCATION | 1,799 | 17.0% |

This imbalance strongly affects model behavior. The models must learn rare entity boundaries while seeing mostly non-entity tokens.

---

## 4. Most Common Errors

### BERT Top Errors

| True Label | Predicted Label | Count |
|---|---|---:|
| O | I-SKILL | 2,383 |
| I-SKILL | O | 2,150 |
| B-SKILL | O | 2,063 |
| O | B-SKILL | 1,951 |
| I-EDUCATION | O | 770 |
| I-JOB_TITLE | O | 625 |
| O | I-JOB_TITLE | 575 |
| O | I-EDUCATION | 568 |

### RoBERTa Top Errors

| True Label | Predicted Label | Count |
|---|---|---:|
| O | I-SKILL | 2,222 |
| O | B-SKILL | 1,979 |
| I-SKILL | O | 1,841 |
| B-SKILL | O | 1,797 |
| I-EDUCATION | O | 581 |
| I-JOB_TITLE | O | 480 |
| O | I-EDUCATION | 457 |
| O | I-JOB_TITLE | 430 |

The most common errors for both models are concentrated around `SKILL`.

There are two main patterns:

1. `O -> SKILL`: the model predicts a skill where the ground truth says there is no entity.
2. `SKILL -> O`: the model misses a real skill and predicts non-entity.

This means the models are both over-predicting and under-predicting skills. The problem is not only a conservative recall issue or only an aggressive precision issue; it is a semantic ambiguity issue.

---

## 5. False Negatives

False negatives are cases where the true label is an entity, but the model predicts `O`.

### BERT False Negatives

| Missed Label | Count |
|---|---:|
| I-SKILL | 2,150 |
| B-SKILL | 2,063 |
| I-EDUCATION | 770 |
| I-JOB_TITLE | 625 |
| B-JOB_TITLE | 385 |
| B-EDUCATION | 257 |

BERT misses **4,213 SKILL tokens** in total.

### RoBERTa False Negatives

| Missed Label | Count |
|---|---:|
| I-SKILL | 1,841 |
| B-SKILL | 1,797 |
| I-EDUCATION | 581 |
| I-JOB_TITLE | 480 |
| B-JOB_TITLE | 279 |
| B-EDUCATION | 178 |

RoBERTa misses **3,638 SKILL tokens** in total.

RoBERTa reduces false negatives across all entity types, especially for `SKILL`, `EDUCATION`, and `JOB_TITLE`. This explains its higher recall.

---

## 6. False Positives

False positives are cases where the true label is `O`, but the model predicts an entity.

### BERT False Positives

| Predicted Label | Count |
|---|---:|
| I-SKILL | 2,383 |
| B-SKILL | 1,951 |
| I-JOB_TITLE | 575 |
| I-EDUCATION | 568 |
| B-JOB_TITLE | 364 |
| B-EDUCATION | 180 |

BERT incorrectly predicts **4,334 SKILL tokens** from non-entity text.

### RoBERTa False Positives

| Predicted Label | Count |
|---|---:|
| I-SKILL | 2,222 |
| B-SKILL | 1,979 |
| I-EDUCATION | 457 |
| I-JOB_TITLE | 430 |
| B-JOB_TITLE | 273 |
| B-EDUCATION | 137 |

RoBERTa incorrectly predicts **4,201 SKILL tokens** from non-entity text.

RoBERTa reduces false positives for most labels, but `SKILL` false positives remain high. This confirms that skill extraction is still the main source of precision loss.

---

## 7. Model Comparison

| Area | BERT | RoBERTa | Interpretation |
|---|---:|---:|---|
| Test F1 | 0.5997 | 0.6590 | RoBERTa is clearly stronger overall |
| JOB_TITLE F1 | 0.7250 | 0.7938 | RoBERTa identifies job titles much better |
| EDUCATION F1 | 0.6417 | 0.7120 | RoBERTa handles education spans better |
| SKILL F1 | 0.5460 | 0.6001 | RoBERTa improves skills, but skills remain hardest |
| SKILL false negatives | 4,213 | 3,638 | RoBERTa misses fewer true skills |
| SKILL false positives | 4,334 | 4,201 | RoBERTa slightly reduces skill over-prediction |

RoBERTa improves all entity categories. The biggest practical gains are in `JOB_TITLE` and `EDUCATION`, where the model benefits from clearer context. `SKILL` also improves, but remains the weakest class.

---

## 8. Overall Performance Summary

RoBERTa is the better-performing model for this resume NER task. It achieves a test F1 of **0.6590**, compared with BERT's **0.5997**. The improvement is consistent across `JOB_TITLE`, `EDUCATION`, and `SKILL`.

The strongest entity type is `JOB_TITLE`, because job titles tend to appear in predictable resume contexts and have more recognizable linguistic patterns. `EDUCATION` performs reasonably well but remains more variable because education spans can include degrees, institutions, certificates, and fields of study.

The main weakness for both models is `SKILL`. Most major errors involve confusion between `SKILL` and `O`. The models often label ordinary text as a skill, while also missing real skills. This shows that the bottleneck is not mainly BIO boundary formatting; it is the semantic ambiguity of skill expressions in resumes.

In conclusion, RoBERTa provides a clear improvement over BERT, but overall performance is still limited by skill extraction. Future improvements should focus on better skill annotations, a curated skill lexicon, domain-adaptive pretraining on resumes, or stronger encoder models.
