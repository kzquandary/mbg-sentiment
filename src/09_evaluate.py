import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


class IndoBERTBiLSTM(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        hidden_size: int = 128,
        dropout: float = 0.3,
        freeze_bert: bool = True,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size
        self.bilstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        if unfreeze_last_n > 0 and hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            layers = self.bert.encoder.layer
            for layer in layers[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state
        lstm_out, _ = self.bilstm(seq_out)
        mask = attention_mask.unsqueeze(-1).float()
        masked = lstm_out * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))


def evaluate_model(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 9 - Final evaluation on test set")
    parser.add_argument("--test", type=str, default="data/test.csv")
    parser.add_argument("--model-path", type=str, default="models/best_indobert_bilstm.pt")
    parser.add_argument("--text-col", type=str, default="text_model_input")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metrics-output", type=str, default="outputs/final_metrics.json")
    parser.add_argument("--cm-csv-output", type=str, default="outputs/confusion_matrix.csv")
    parser.add_argument("--report-csv-output", type=str, default="outputs/classification_report.csv")
    parser.add_argument("--cm-fig-output", type=str, default="outputs/figures/confusion_matrix.png")
    parser.add_argument("--pred-output", type=str, default="outputs/test_predictions.csv")
    args = parser.parse_args()

    set_global_seed(SEED)

    test_path = Path(args.test)
    model_path = Path(args.model_path)
    metrics_output = Path(args.metrics_output)
    cm_csv_output = Path(args.cm_csv_output)
    report_csv_output = Path(args.report_csv_output)
    cm_fig_output = Path(args.cm_fig_output)
    pred_output = Path(args.pred_output)

    for p in [test_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    for p in [metrics_output, cm_csv_output, report_csv_output, cm_fig_output, pred_output]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    checkpoint = torch.load(model_path, map_location="cpu")
    best_config = checkpoint.get("best_config", {})
    label2id_raw = checkpoint.get("label2id")
    id2label_raw = checkpoint.get("id2label")
    if not label2id_raw or not id2label_raw:
        raise ValueError("Checkpoint tidak memiliki label mapping.")

    label2id = {str(k): int(v) for k, v in label2id_raw.items()}
    id2label = {int(k): str(v) for k, v in id2label_raw.items()} if isinstance(next(iter(id2label_raw.keys())), str) else {
        int(k): str(v) for k, v in id2label_raw.items()
    }
    ordered_labels = [id2label[i] for i in sorted(id2label.keys())]

    model_name = best_config.get("model_name", "indobenchmark/indobert-base-p1")
    max_len = int(best_config.get("max_len", 96))
    hidden_size = int(best_config.get("hidden_size", 128))
    dropout = float(best_config.get("dropout", 0.3))
    freeze_bert = bool(best_config.get("freeze_bert", True))
    unfreeze_last_n = int(best_config.get("unfreeze_last_n", 0))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    test_df = pd.read_csv(test_path)
    if args.text_col not in test_df.columns:
        raise KeyError(f"Text column not found: {args.text_col}")
    if args.label_col not in test_df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    texts = test_df[args.text_col].fillna("").astype(str).tolist()
    labels_str = test_df[args.label_col].astype(str).tolist()
    labels = [label2id[x] for x in labels_str]

    ds = TextDataset(texts, labels, tokenizer, max_len=max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IndoBERTBiLSTM(
        model_name=model_name,
        num_labels=len(label2id),
        hidden_size=hidden_size,
        dropout=dropout,
        freeze_bert=freeze_bert,
        unfreeze_last_n=unfreeze_last_n,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = evaluate_model(model, loader, device)
    y_true_lbl = [id2label[i] for i in y_true]
    y_pred_lbl = [id2label[i] for i in y_pred]

    acc = accuracy_score(y_true_lbl, y_pred_lbl)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_lbl, y_pred_lbl, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=ordered_labels)
    cm_df = pd.DataFrame(cm, index=ordered_labels, columns=ordered_labels)
    cm_df.to_csv(cm_csv_output, encoding="utf-8-sig")

    report_dict = classification_report(y_true_lbl, y_pred_lbl, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(report_csv_output, encoding="utf-8-sig")

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_fig_output, dpi=220)
    plt.close()

    pred_df = test_df.copy()
    pred_df["y_true"] = y_true_lbl
    pred_df["y_pred"] = y_pred_lbl
    pred_df["is_correct"] = pred_df["y_true"] == pred_df["y_pred"]
    pred_df.to_csv(pred_output, index=False, encoding="utf-8-sig")

    metrics_payload = {
        "seed": SEED,
        "model_path": str(model_path),
        "model_name": model_name,
        "device": str(device),
        "test_size": int(len(test_df)),
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "labels_order": ordered_labels,
    }
    write_json(metrics_payload, metrics_output)

    print(f"[OK] Final metrics saved: {metrics_output}")
    print(f"[OK] Confusion matrix CSV saved: {cm_csv_output}")
    print(f"[OK] Classification report CSV saved: {report_csv_output}")
    print(f"[OK] Confusion matrix figure saved: {cm_fig_output}")
    print(f"[OK] Test predictions saved: {pred_output}")


if __name__ == "__main__":
    main()
