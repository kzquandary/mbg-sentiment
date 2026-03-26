import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed


@dataclass
class TrialConfig:
    trial_name: str
    model_name: str
    max_len: int
    batch_size: int
    hidden_size: int
    dropout: float
    lr: float
    epochs: int
    freeze_bert: bool
    unfreeze_last_n: int
    optimizer: str
    patience: int
    classifier_type: str = "bilstm"  # bilstm | linear
    loss_type: str = "ce"  # ce | focal
    focal_gamma: float = 2.0
    use_weighted_sampler: bool = False
    bert_lr: float = 0.0
    head_lr: float = 0.0
    warmup_ratio: float = 0.1


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
        seq_out = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        if self.classifier_type == "bilstm":
            lstm_out, _ = self.bilstm(seq_out)
            masked = lstm_out * mask
        else:
            masked = seq_out * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(self.dropout(pooled))
        return logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def train_one_trial(
    cfg: TrialConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    id2label: dict[int, str],
    device: torch.device,
):
    model = IndoBERTBiLSTM(
        model_name=cfg.model_name,
        num_labels=len(id2label),
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
        freeze_bert=cfg.freeze_bert,
        unfreeze_last_n=cfg.unfreeze_last_n,
        classifier_type=cfg.classifier_type,
    ).to(device)

    cls_weight = class_weights.to(device)
    if cfg.loss_type == "focal":
        criterion = FocalLoss(gamma=cfg.focal_gamma, weight=cls_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=cls_weight)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if cfg.bert_lr > 0 and cfg.head_lr > 0:
        bert_params = []
        head_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("bert."):
                bert_params.append(p)
            else:
                head_params.append(p)
        optim_groups = []
        if bert_params:
            optim_groups.append({"params": bert_params, "lr": cfg.bert_lr})
        if head_params:
            optim_groups.append({"params": head_params, "lr": cfg.head_lr})
    else:
        optim_groups = [{"params": trainable_params, "lr": cfg.lr}]

    if cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(optim_groups)
    else:
        optimizer = torch.optim.Adam(optim_groups)

    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = int(total_steps * max(0.0, min(0.5, cfg.warmup_ratio)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = -1.0
    best_state = None
    history_rows = []
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss = train_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, criterion)
        row = {
            "trial_name": cfg.trial_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr_group0": optimizer.param_groups[0]["lr"] if optimizer.param_groups else None,
        }
        history_rows.append(row)

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_val = evaluate(model, val_loader, device, criterion)
    return model, history_rows, best_val


def build_trial_configs(model_name: str) -> list[TrialConfig]:
    return [
        TrialConfig(
            trial_name="frozen_wordpiece_small",
            model_name=model_name,
            max_len=96,
            batch_size=8,
            hidden_size=128,
            dropout=0.3,
            lr=1e-3,
            epochs=4,
            freeze_bert=True,
            unfreeze_last_n=0,
            optimizer="adamw",
            patience=2,
            classifier_type="bilstm",
            loss_type="ce",
            use_weighted_sampler=False,
        ),
        TrialConfig(
            trial_name="frozen_wordpiece_deeper",
            model_name=model_name,
            max_len=128,
            batch_size=6,
            hidden_size=192,
            dropout=0.4,
            lr=8e-4,
            epochs=4,
            freeze_bert=True,
            unfreeze_last_n=0,
            optimizer="adamw",
            patience=2,
            classifier_type="bilstm",
            loss_type="ce",
            use_weighted_sampler=False,
        ),
        TrialConfig(
            trial_name="finetune_last1",
            model_name=model_name,
            max_len=96,
            batch_size=4,
            hidden_size=128,
            dropout=0.3,
            lr=2e-5,
            epochs=3,
            freeze_bert=False,
            unfreeze_last_n=1,
            optimizer="adamw",
            patience=1,
            classifier_type="bilstm",
            loss_type="ce",
            use_weighted_sampler=False,
        ),
    ]


def load_trial_configs_from_json(config_path: Path, default_model_name: str) -> list[TrialConfig]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("trial-configs-json harus berisi object atau list of object.")

    configs: list[TrialConfig] = []
    required_keys = {
        "trial_name",
        "max_len",
        "batch_size",
        "hidden_size",
        "dropout",
        "lr",
        "epochs",
        "freeze_bert",
        "unfreeze_last_n",
        "optimizer",
        "patience",
    }
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Konfigurasi trial index {idx} bukan object.")
        missing = [k for k in required_keys if k not in item]
        if missing:
            raise ValueError(f"Konfigurasi trial index {idx} missing key: {missing}")

        model_name = str(item.get("model_name", default_model_name))
        configs.append(
            TrialConfig(
                trial_name=str(item["trial_name"]),
                model_name=model_name,
                max_len=int(item["max_len"]),
                batch_size=int(item["batch_size"]),
                hidden_size=int(item["hidden_size"]),
                dropout=float(item["dropout"]),
                lr=float(item["lr"]),
                epochs=int(item["epochs"]),
                freeze_bert=bool(item["freeze_bert"]),
                unfreeze_last_n=int(item["unfreeze_last_n"]),
                optimizer=str(item["optimizer"]),
                patience=int(item["patience"]),
                classifier_type=str(item.get("classifier_type", "bilstm")),
                loss_type=str(item.get("loss_type", "ce")),
                focal_gamma=float(item.get("focal_gamma", 2.0)),
                use_weighted_sampler=bool(item.get("use_weighted_sampler", False)),
                bert_lr=float(item.get("bert_lr", 0.0)),
                head_lr=float(item.get("head_lr", 0.0)),
                warmup_ratio=float(item.get("warmup_ratio", 0.1)),
            )
        )
    return configs


def main():
    parser = argparse.ArgumentParser(description="Step 7 - IndoBERT + BiLSTM")
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--text-col", type=str, default="text_model_input")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--model-name", type=str, default="indobenchmark/indobert-lite-base-p2")
    parser.add_argument("--target-f1", type=float, default=None)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--trial-configs-json", type=str, default="")
    parser.add_argument("--best-model-output", type=str, default="models/best_indobert_bilstm.pt")
    parser.add_argument("--history-output", type=str, default="outputs/training_history.csv")
    parser.add_argument("--trials-output", type=str, default="outputs/step7_trials.csv")
    parser.add_argument("--arch-output", type=str, default="outputs/model_architecture.md")
    parser.add_argument("--best-config-output", type=str, default="outputs/step7_best_config.json")
    args = parser.parse_args()

    set_global_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    train_path = Path(args.train)
    val_path = Path(args.val)
    best_model_output = Path(args.best_model_output)
    history_output = Path(args.history_output)
    trials_output = Path(args.trials_output)
    arch_output = Path(args.arch_output)
    best_config_output = Path(args.best_config_output)

    for p in [train_path, val_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    for p in [best_model_output, history_output, trials_output, arch_output, best_config_output]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    for name, df in [("train", train_df), ("val", val_df)]:
        if args.text_col not in df.columns:
            raise KeyError(f"{name} missing text column: {args.text_col}")
        if args.label_col not in df.columns:
            raise KeyError(f"{name} missing label column: {args.label_col}")

    labels_sorted = sorted(train_df[args.label_col].astype(str).unique().tolist())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: l for l, i in label2id.items()}

    train_texts = train_df[args.text_col].fillna("").astype(str).tolist()
    val_texts = val_df[args.text_col].fillna("").astype(str).tolist()
    train_labels = train_df[args.label_col].astype(str).map(label2id).tolist()
    val_labels = val_df[args.label_col].astype(str).map(label2id).tolist()

    class_weight_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(id2label.keys())),
        y=np.array(train_labels),
    )
    class_weights = torch.tensor(class_weight_arr, dtype=torch.float32)

    if args.trial_configs_json:
        cfg_path = Path(args.trial_configs_json)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Trial config JSON not found: {cfg_path}")
        configs = load_trial_configs_from_json(cfg_path, args.model_name)[: args.max_trials]
    else:
        configs = build_trial_configs(args.model_name)[: args.max_trials]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_history = []
    trial_summaries = []
    best_overall = None
    best_overall_f1 = -1.0
    best_model_state = None
    tokenizer = None
    tokenizer_model_name = None

    for cfg in configs:
        if tokenizer is None or tokenizer_model_name != cfg.model_name:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
            tokenizer_model_name = cfg.model_name

        train_ds = TextDataset(train_texts, train_labels, tokenizer, cfg.max_len)
        val_ds = TextDataset(val_texts, val_labels, tokenizer, cfg.max_len)
        if cfg.use_weighted_sampler:
            train_counts = pd.Series(train_labels).value_counts().to_dict()
            sample_w = [1.0 / float(train_counts[int(y)]) for y in train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_w),
                num_samples=len(sample_w),
                replacement=True,
            )
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=False)
        else:
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        model, history_rows, best_val = train_one_trial(
            cfg, train_loader, val_loader, class_weights, id2label, device
        )

        for row in history_rows:
            row["timestamp"] = datetime.now().isoformat()
            row["config_json"] = json.dumps(cfg.__dict__, ensure_ascii=False)
            all_history.append(row)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "trial_name": cfg.trial_name,
            "val_accuracy": round(best_val["accuracy"], 6),
            "val_precision_macro": round(best_val["precision_macro"], 6),
            "val_recall_macro": round(best_val["recall_macro"], 6),
            "val_f1_macro": round(best_val["f1_macro"], 6),
            "config_json": json.dumps(cfg.__dict__, ensure_ascii=False),
        }
        trial_summaries.append(summary)

        if best_val["f1_macro"] > best_overall_f1:
            best_overall_f1 = best_val["f1_macro"]
            best_overall = cfg
            best_model_state = copy.deepcopy(model.state_dict())

        if args.target_f1 is not None and best_overall_f1 >= args.target_f1:
            break

    if best_model_state is None or best_overall is None:
        raise RuntimeError("No trial was successfully trained.")

    torch.save(
        {
            "model_state_dict": best_model_state,
            "label2id": label2id,
            "id2label": id2label,
            "best_config": best_overall.__dict__,
            "best_val_f1_macro": best_overall_f1,
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
        },
        best_model_output,
    )

    pd.DataFrame(all_history).to_csv(history_output, index=False, encoding="utf-8-sig")
    pd.DataFrame(trial_summaries).sort_values("val_f1_macro", ascending=False).to_csv(
        trials_output, index=False, encoding="utf-8-sig"
    )
    Path(best_config_output).write_text(
        json.dumps(
            {
                "best_config": best_overall.__dict__,
                "best_val_f1_macro": best_overall_f1,
                "target_f1_macro": args.target_f1,
                "target_achieved": (bool(best_overall_f1 >= args.target_f1) if args.target_f1 is not None else None),
                "device": str(device),
                "model_name": args.model_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    arch_lines = [
        "# Model Architecture (Step 7)",
        "",
        f"- Timestamp: {datetime.now().isoformat()}",
        f"- Base model: `{args.model_name}`",
        "- Architecture: `IndoBERT -> BiLSTM (bidirectional, 1 layer) -> Dropout -> Linear classifier`",
        f"- Device: `{device}`",
        f"- Labels: `{label2id}`",
        "",
        "## Best Configuration",
        f"- Trial name: `{best_overall.trial_name}`",
        f"- Max length: `{best_overall.max_len}`",
        f"- Batch size: `{best_overall.batch_size}`",
        f"- Hidden size: `{best_overall.hidden_size}`",
        f"- Dropout: `{best_overall.dropout}`",
        f"- Learning rate: `{best_overall.lr}`",
        f"- Freeze BERT: `{best_overall.freeze_bert}`",
        f"- Unfreeze last N: `{best_overall.unfreeze_last_n}`",
        f"- Best validation macro F1: `{round(best_overall_f1, 6)}`",
    ]
    arch_output.write_text("\n".join(arch_lines), encoding="utf-8")

    print(f"[OK] Best model saved: {best_model_output}")
    print(f"[OK] Training history saved: {history_output}")
    print(f"[OK] Trial summary saved: {trials_output}")
    print(f"[OK] Best config saved: {best_config_output}")
    print(f"[OK] Model architecture saved: {arch_output}")
    print(f"[INFO] Best validation macro F1: {best_overall_f1:.6f}")
    if args.target_f1 is not None:
        print(f"[INFO] Target {args.target_f1:.2f} achieved: {best_overall_f1 >= args.target_f1}")
    else:
        print("[INFO] Target check disabled.")


if __name__ == "__main__":
    main()
