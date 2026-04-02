import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
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
        classifier_type: str = "bilstm",
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size
        self.classifier_type = classifier_type
        self.input_dropout = nn.Dropout(dropout)
        if self.classifier_type == "bilstm":
            self.bilstm = nn.LSTM(
                input_size=bert_hidden,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            classifier_in = hidden_size * 2
        else:
            self.bilstm = None
            classifier_in = bert_hidden
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, num_labels)

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
        seq_out = self.input_dropout(outputs.last_hidden_state)
        if self.classifier_type == "bilstm":
            seq_out, _ = self.bilstm(seq_out)
        mask = attention_mask.unsqueeze(-1).float()
        masked = seq_out.masked_fill(mask == 0, -1e9)
        pooled = masked.max(dim=1).values
        return self.classifier(self.dropout(pooled))


def infer_probs(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    return y_true, torch.tensor(y_prob, dtype=torch.float32)


def macro_f1(y_true_ids, y_pred_ids):
    _, _, f1, _ = precision_recall_fscore_support(y_true_ids, y_pred_ids, average="macro", zero_division=0)
    return float(f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune class multipliers on validation set to improve macro-F1.")
    parser.add_argument("--val", type=str, default="data/val_sub.csv")
    parser.add_argument("--model-path", type=str, default="models/best_indobert_bilstm.pt")
    parser.add_argument("--text-col", type=str, default="text_model_input")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mult-min", type=float, default=0.8)
    parser.add_argument("--mult-max", type=float, default=1.6)
    parser.add_argument("--mult-step", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="outputs/best_class_multipliers.json")
    parser.add_argument("--summary-output", type=str, default="outputs/class_multiplier_tuning_summary.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    val_path = Path(args.val)
    model_path = Path(args.model_path)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    for p in [val_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    for p in [output_path, summary_path]:
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
    labels_sorted_ids = sorted(id2label.keys())

    model_name = best_config.get("model_name", "indobenchmark/indobert-base-p1")
    max_len = int(best_config.get("max_len", 96))
    hidden_size = int(best_config.get("hidden_size", 128))
    dropout = float(best_config.get("dropout", 0.3))
    freeze_bert = bool(best_config.get("freeze_bert", True))
    unfreeze_last_n = int(best_config.get("unfreeze_last_n", 0))
    classifier_type = str(best_config.get("classifier_type", "bilstm"))

    val_df = pd.read_csv(val_path)
    if args.text_col not in val_df.columns:
        raise KeyError(f"Text column not found: {args.text_col}")
    if args.label_col not in val_df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")
    texts = val_df[args.text_col].fillna("").astype(str).tolist()
    labels = [label2id[x] for x in val_df[args.label_col].astype(str).tolist()]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
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
        classifier_type=classifier_type,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, prob = infer_probs(model, loader, device)
    baseline_pred = torch.argmax(prob, dim=1).tolist()
    baseline_f1 = macro_f1(y_true, baseline_pred)

    label_neg = "Negatif"
    label_pos = "Positif"
    if label_neg not in label2id or label_pos not in label2id:
        raise ValueError("Label Negatif/Positif tidak ditemukan untuk tuning multiplier.")

    neg_id = label2id[label_neg]
    pos_id = label2id[label_pos]

    vals = []
    cur = args.mult_min
    while cur <= args.mult_max + 1e-9:
        vals.append(round(cur, 4))
        cur += args.mult_step

    best_f1 = baseline_f1
    best_pair = (1.0, 1.0)
    trials = 0
    for neg_mult in vals:
        for pos_mult in vals:
            mult = torch.ones(len(labels_sorted_ids), dtype=torch.float32)
            mult[neg_id] = float(neg_mult)
            mult[pos_id] = float(pos_mult)
            pred = torch.argmax(prob * mult.unsqueeze(0), dim=1).tolist()
            f1 = macro_f1(y_true, pred)
            trials += 1
            if f1 > best_f1:
                best_f1 = f1
                best_pair = (float(neg_mult), float(pos_mult))

    best_multiplier_map = {id2label[i]: 1.0 for i in labels_sorted_ids}
    best_multiplier_map[label_neg] = best_pair[0]
    best_multiplier_map[label_pos] = best_pair[1]

    write_json(best_multiplier_map, output_path)
    write_json(
        {
            "seed": SEED,
            "val_file": str(val_path),
            "model_path": str(model_path),
            "baseline_macro_f1": baseline_f1,
            "best_macro_f1": best_f1,
            "improvement": float(best_f1 - baseline_f1),
            "search_space": {
                "mult_min": args.mult_min,
                "mult_max": args.mult_max,
                "mult_step": args.mult_step,
                "num_trials": trials,
            },
            "best_multipliers": best_multiplier_map,
        },
        summary_path,
    )

    print(f"[OK] Best class multipliers saved: {output_path}")
    print(f"[OK] Tuning summary saved: {summary_path}")
    print(f"[INFO] Baseline macro F1: {baseline_f1:.6f}")
    print(f"[INFO] Best macro F1 (val): {best_f1:.6f}")
    print(f"[INFO] Improvement: {best_f1 - baseline_f1:.6f}")


if __name__ == "__main__":
    main()
