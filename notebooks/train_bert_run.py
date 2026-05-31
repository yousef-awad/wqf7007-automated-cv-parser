import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.append("..")

import os
import csv
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import torch
from datasets import load_from_disk
from src.evaluation import evaluate_resume_level_standard, save_to_dataframe
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME       = "bert-base-uncased"
DATASET_PATH     = "../data/processed/resume_ner_hf"
OUT_DIR          = f"../checkpoints/{MODEL_NAME}"
LOG_DIR          = f"../logs/{MODEL_NAME}"
RESULT_CSV       = "../results/experiment_results.csv"
ENTITY_CSV       = "../results/per_entity_results.csv"

LEARNING_RATE    = 2e-5
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE  = 4
EPOCH            = 4
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
STRIDE           = 128

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("../results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Dataset ────────────────────────────────────────────────────────────────
ds = load_from_disk(DATASET_PATH)

label_list = ["O", "B-JOB_TITLE", "I-JOB_TITLE", "B-SKILL", "I-SKILL", "B-EDUCATION", "I-EDUCATION"]
id2label   = {i: label for i, label in enumerate(label_list)}
label2id   = {label: i for i, label in enumerate(label_list)}

print(f"Dataset loaded: train={ds['train'].num_rows:,}  val={ds['validation'].num_rows:,}  test={ds['test'].num_rows:,}")

# ── Model ──────────────────────────────────────────────────────────────────
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)
tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=EPOCH,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=LOG_DIR,
    logging_strategy="epoch",
    fp16=torch.cuda.is_available(),
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
)

# ── Train ──────────────────────────────────────────────────────────────────
print(f"\nStarting training: lr={LEARNING_RATE}  epochs={EPOCH}  batch={TRAIN_BATCH_SIZE}")
start_time = time.time()
trainer.train()
training_time = time.time() - start_time
print(f"Training done in {training_time/60:.2f} min")

# ── Validation ─────────────────────────────────────────────────────────────
val_results = evaluate_resume_level_standard(
    trainer=trainer,
    dataset=ds["validation"],
    id2label=id2label,
    stride=STRIDE,
)
print(f"\n===== Resume-Level Validation =====")
print(f"Precision : {val_results['precision']:.4f}")
print(f"Recall    : {val_results['recall']:.4f}")
print(f"F1        : {val_results['f1']:.4f}")
save_to_dataframe(results=val_results, model_name=MODEL_NAME, split="validation", file_path=ENTITY_CSV)

# ── Test ───────────────────────────────────────────────────────────────────
test_results = evaluate_resume_level_standard(
    trainer=trainer,
    dataset=ds["test"],
    id2label=id2label,
    stride=STRIDE,
)
print(f"\n===== Resume-Level Test =====")
print(f"Precision : {test_results['precision']:.4f}")
print(f"Recall    : {test_results['recall']:.4f}")
print(f"F1        : {test_results['f1']:.4f}")
save_to_dataframe(results=test_results, model_name=MODEL_NAME, split="test", file_path=ENTITY_CSV)

# ── Log ────────────────────────────────────────────────────────────────────
def append_row_to_csv(file_path, row):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

max_gpu_mem = torch.cuda.max_memory_allocated(0) / 1e9 if device == "cuda" else 0

log_data = {
    "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model":                MODEL_NAME,
    "learning_rate":        LEARNING_RATE,
    "epochs":               EPOCH,
    "train_batch_size":     TRAIN_BATCH_SIZE,
    "eval_batch_size":      EVAL_BATCH_SIZE,
    "validation_precision": val_results["precision"],
    "validation_recall":    val_results["recall"],
    "validation_f1":        val_results["f1"],
    "test_precision":       test_results["precision"],
    "test_recall":          test_results["recall"],
    "test_f1":              test_results["f1"],
    "training_time_min":    round(training_time / 60, 2),
    "max_gpu_memory_gb":    round(max_gpu_mem, 2),
    "device":               device,
}

append_row_to_csv(RESULT_CSV, log_data)
print(f"\nLogged to {RESULT_CSV}")
print(f"Per-entity logged to {ENTITY_CSV}")
print("\nDONE.")
