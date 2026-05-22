import os

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

def merge_chunks(chunks, stride=128):
    merged = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            merged.extend(chunk)
        else:
            merged.extend(chunk[stride:])
    
    return merged

def build_resume_level_sequences(preds, labels, resume_ids, id2label, stride=128):
    grouped_preds = defaultdict(list)
    grouped_labels = defaultdict(list)

    for i, r_id in enumerate(resume_ids):
        grouped_preds[r_id].append(preds[i])
        grouped_labels[r_id].append(labels[i])
    
    # Merge chunks
    final_preds = []
    final_labels = []

    for r_id in grouped_preds:
        merged_pred = merge_chunks(grouped_preds[r_id], stride=stride)
        merged_label = merge_chunks(grouped_labels[r_id], stride=stride)

        seq_pred = []
        seq_label = []

        for p, l in zip(merged_pred, merged_label):
            if l == -100:
                continue

            seq_pred.append(id2label[int(p)])
            seq_label.append(id2label[int(l)])

        final_preds.append(seq_pred)
        final_labels.append(seq_label)

    return final_labels, final_preds

def compute_seqeval_metrics(final_labels, final_preds):
    report = classification_report(
        final_labels,
        final_preds,
        output_dict=True,
        zero_division=0
    )

    # Metrics
    precision = precision_score(final_labels, final_preds)
    recall = recall_score(final_labels, final_preds)
    f1 = f1_score(final_labels, final_preds)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "final_labels": final_labels,
        "final_preds": final_preds
    }

def evaluate_resume_level_standard(trainer, dataset, id2label, stride=128):
    """
    For standard BERT/RoBERTa token classification models.
    Uses Trainer.predict() + argmax
    """
    output = trainer.predict(dataset)

    logits = output.predictions
    labels = output.label_ids
    preds = np.argmax(logits, axis=2)

    resume_ids = dataset["resume_idx"]

    final_labels, final_preds = build_resume_level_sequences(
        preds=preds,
        labels=labels,
        resume_ids=resume_ids,
        id2label=id2label,
        stride=stride
    )

    return compute_seqeval_metrics(final_labels, final_preds)

# bert-crf model evaluation pipeline
def predict_with_crf(model, dataset, data_collator, device, batch_size=4):
    """
    For BERT + CRF model evaluation
    Uses model.decode(), not argmax
    """
    model.eval()
    model.to(device)

    # Keep only the fields the data collator / model need
    tensor_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    cols_to_remove = [c for c in dataset.column_names if c not in tensor_cols]
    clean_dataset = dataset.remove_columns(cols_to_remove)

    loader = DataLoader(
        dataset=clean_dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "token_type_ids", "labels"]
            }

            decoded = model.decode(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids"),
                labels=batch.get("labels")
            )

            labels = batch["labels"].cpu().numpy()
            attn_masks = batch["attention_mask"].cpu().numpy()

            for pred_seq, label_seq, attn in zip(decoded, labels, attn_masks):
                # pred_seq has one entry per real token (attention_mask==1),
                # including [CLS] and [SEP] which have label -100.
                full_preds = []
                pred_idx = 0

                for pos, l in enumerate(label_seq):
                    if int(attn[pos]) == 0:
                        # Padding position — no prediction
                        full_preds.append(-100)
                    else:
                        # Real token (CLS, SEP, subword, or labelled token)
                        if int(l) == -100:
                            full_preds.append(-100)
                        else:
                            full_preds.append(pred_seq[pred_idx])
                        pred_idx += 1  # always advance for real (non-pad) tokens

                all_preds.append(full_preds)
                all_labels.append(label_seq)

    return all_preds, all_labels

def evaluate_resume_level_crf(model, dataset, id2label, data_collator, device, stride=128, batch_size=4):
    preds, labels = predict_with_crf(
        model=model,
        dataset=dataset,
        data_collator=data_collator,
        device=device,
        batch_size=batch_size
    )

    resume_ids = dataset["resume_idx"]

    final_labels, final_preds = build_resume_level_sequences(
        preds=preds,
        labels=labels,
        resume_ids=resume_ids,
        id2label=id2label,
        stride=stride
    )

    return compute_seqeval_metrics(final_labels, final_preds)

# report log
def save_to_dataframe(results, model_name, split, file_path):
    
    rows = []

    for entity, scores in results["report"].items():
        if isinstance(scores, dict) and "f1-score" in scores:
            rows.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_name,
                "split": split,
                "entity": entity,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1-score"],
                "support": scores["support"]
            })

    df = pd.DataFrame(rows)

    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    return df