import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json

DEFAULT_OVERRIDE_CONFIG = "src/resources/preprocess_overrides.json"


def normalize_text_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def handle_emoji_with_package(text: str) -> str:
    from emosent import get_emoji_sentiment_rank  # type: ignore

    out = []
    for ch in text:
        try:
            rank = get_emoji_sentiment_rank(ch)
        except Exception:
            rank = None

        if rank:
            score = float(rank.get("sentiment_score", 0.0))
            if score > 0:
                out.append(" emopos ")
            elif score < 0:
                out.append(" emoneg ")
            else:
                out.append(" emoneu ")
        else:
            out.append(ch)
    return "".join(out)


def get_emoji_handler():
    try:
        from emosent import get_emoji_sentiment_rank  # noqa: F401

        return handle_emoji_with_package, "emosent.get_emoji_sentiment_rank"
    except Exception as exc:
        raise ImportError("Package emosent-py wajib tersedia. Install: pip install emosent-py") from exc


def remove_urls_mentions_symbols(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    return text


def normalize_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def get_slang_normalizer():
    try:
        from indoNLP.preprocessing import replace_slang  # type: ignore

        return replace_slang, "indoNLP.replace_slang"
    except Exception as exc:
        raise ImportError("Package indoNLP wajib tersedia. Install: pip install indoNLP") from exc


def clean_basic(text: str, emoji_handler) -> str:
    text = text.lower()
    text = emoji_handler(text)
    text = remove_urls_mentions_symbols(text)
    text = normalize_repeated_chars(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_slang_with_protection(text: str, slang_normalizer, preserve_tokens: set[str]) -> str:
    protected = text
    placeholders: dict[str, str] = {}
    for i, tok in enumerate(sorted(preserve_tokens)):
        placeholder = f"__PRESERVE_{i}__"
        placeholders[placeholder] = tok
        protected = re.sub(rf"\b{re.escape(tok)}\b", placeholder, protected)

    normalized = slang_normalizer(protected)
    for placeholder, tok in placeholders.items():
        normalized = normalized.replace(placeholder.lower(), tok)
        normalized = normalized.replace(placeholder, tok)
    return normalized


def apply_manual_token_overrides(text: str, token_overrides: dict[str, str]) -> str:
    if not token_overrides:
        return text
    tokens = text.split()
    tokens = [token_overrides.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


def clean_normalized(text: str, slang_normalizer, preserve_tokens: set[str], token_overrides: dict[str, str]) -> str:
    text = apply_slang_with_protection(text, slang_normalizer, preserve_tokens)
    text = apply_manual_token_overrides(text, token_overrides)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_emoji_sentiment_tokens(text: str) -> str:
    text = re.sub(r"\b(?:emopos|emoneg|emoneu)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_stopword_set(extra_stopwords: list[str]) -> set[str]:
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # type: ignore
    except Exception as exc:
        raise ImportError("Package Sastrawi wajib tersedia. Install: pip install Sastrawi") from exc

    try:
        import stopwordsiso as stopwordsiso  # type: ignore
    except Exception as exc:
        raise ImportError("Package stopwordsiso wajib tersedia. Install: pip install stopwordsiso") from exc

    sastrawi_set = {w.strip().lower() for w in StopWordRemoverFactory().get_stop_words() if str(w).strip()}
    swiso_id_set = {w.strip().lower() for w in stopwordsiso.stopwords("id") if str(w).strip()}
    extra = {w.strip().lower() for w in extra_stopwords if str(w).strip()}
    return sastrawi_set.union(swiso_id_set).union(extra)


def remove_stopwords_for_eda(text: str, stopword_set: set[str]) -> str:
    tokens = [t for t in text.split() if t and t not in stopword_set]
    return " ".join(tokens)


def load_override_config(path: Path) -> dict:
    if not path.exists():
        return {"preserve_tokens": [], "token_overrides": {}, "eda_extra_stopwords": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    preserve_tokens = data.get("preserve_tokens", [])
    token_overrides = data.get("token_overrides", {})
    eda_extra_stopwords = data.get("eda_extra_stopwords", [])
    if (
        not isinstance(preserve_tokens, list)
        or not isinstance(token_overrides, dict)
        or not isinstance(eda_extra_stopwords, list)
    ):
        raise ValueError("Format override config tidak valid.")
    return {
        "preserve_tokens": [str(x).lower() for x in preserve_tokens],
        "token_overrides": token_overrides,
        "eda_extra_stopwords": [str(x).lower() for x in eda_extra_stopwords],
    }


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str,
    emoji_handler,
    slang_normalizer,
    preserve_tokens: set[str],
    token_overrides: dict[str, str],
    stopword_set: set[str],
) -> pd.DataFrame:
    out = df.copy()
    out["text_original"] = out[text_col].apply(normalize_text_value)
    out["text_clean_basic"] = out["text_original"].apply(lambda x: clean_basic(x, emoji_handler))
    out["text_clean_normalized"] = out["text_clean_basic"].apply(
        lambda x: clean_normalized(x, slang_normalizer, preserve_tokens, token_overrides)
    )
    out["text_model_input"] = out["text_clean_normalized"]
    out["text_eda_input"] = out["text_model_input"].apply(remove_emoji_sentiment_tokens)
    out["text_eda_input"] = out["text_eda_input"].apply(lambda x: remove_stopwords_for_eda(x, stopword_set))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4 - Text preprocessing on train/test branches")
    parser.add_argument("--train-input", type=str, default="data/train.csv")
    parser.add_argument("--test-input", type=str, default="data/test.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--train-output", type=str, default="data/train.csv")
    parser.add_argument("--test-output", type=str, default="data/test.csv")
    parser.add_argument("--combined-output-xlsx", type=str, default="data/preprocessed_dataset.xlsx")
    parser.add_argument("--combined-output-csv", type=str, default="data/preprocessed_dataset.csv")
    parser.add_argument("--log-output", type=str, default="outputs/preprocessing_log.json")
    parser.add_argument("--sample-output", type=str, default="outputs/preprocessing_samples.csv")
    parser.add_argument("--sample-n", type=int, default=20)
    parser.add_argument("--override-config", type=str, default=DEFAULT_OVERRIDE_CONFIG)
    args = parser.parse_args()

    set_global_seed(SEED)

    train_input = Path(args.train_input)
    test_input = Path(args.test_input)
    train_output = Path(args.train_output)
    test_output = Path(args.test_output)
    combined_output_xlsx = Path(args.combined_output_xlsx)
    combined_output_csv = Path(args.combined_output_csv)
    log_output = Path(args.log_output)
    sample_output = Path(args.sample_output)
    override_config_path = Path(args.override_config)

    if not train_input.exists():
        raise FileNotFoundError(f"Train input file not found: {train_input}")
    if not test_input.exists():
        raise FileNotFoundError(f"Test input file not found: {test_input}")

    for p in [train_output, test_output, combined_output_xlsx, combined_output_csv, log_output, sample_output]:
        ensure_dir(p.parent)

    train_df = pd.read_csv(train_input)
    test_df = pd.read_csv(test_input)
    for name, df in [("train", train_df), ("test", test_df)]:
        if args.text_col not in df.columns:
            raise KeyError(f"{name} text column not found: {args.text_col}")

    backup_info = {
        "train_output_backup": backup_if_exists(train_output),
        "test_output_backup": backup_if_exists(test_output),
        "combined_output_xlsx_backup": backup_if_exists(combined_output_xlsx),
        "combined_output_csv_backup": backup_if_exists(combined_output_csv),
        "log_output_backup": backup_if_exists(log_output),
        "sample_output_backup": backup_if_exists(sample_output),
    }

    ensure_dir(log_output.parent)
    ensure_dir(sample_output.parent)

    override_config = load_override_config(override_config_path)
    preserve_tokens = set(override_config["preserve_tokens"])
    token_overrides = {str(k).lower(): str(v).lower() for k, v in override_config["token_overrides"].items()}
    stopword_set = get_stopword_set(override_config["eda_extra_stopwords"])

    slang_normalizer, slang_normalizer_name = get_slang_normalizer()
    emoji_handler, emoji_handler_name = get_emoji_handler()

    train_pp = preprocess_dataframe(
        train_df,
        text_col=args.text_col,
        emoji_handler=emoji_handler,
        slang_normalizer=slang_normalizer,
        preserve_tokens=preserve_tokens,
        token_overrides=token_overrides,
        stopword_set=stopword_set,
    )
    test_pp = preprocess_dataframe(
        test_df,
        text_col=args.text_col,
        emoji_handler=emoji_handler,
        slang_normalizer=slang_normalizer,
        preserve_tokens=preserve_tokens,
        token_overrides=token_overrides,
        stopword_set=stopword_set,
    )

    train_pp.to_csv(train_output, index=False, encoding="utf-8-sig")
    test_pp.to_csv(test_output, index=False, encoding="utf-8-sig")

    combined = pd.concat(
        [train_pp.assign(split_source="train"), test_pp.assign(split_source="test")],
        ignore_index=True,
    )
    combined.to_excel(combined_output_xlsx, index=False)
    combined.to_csv(combined_output_csv, index=False, encoding="utf-8-sig")

    sample_cols = ["text_original", "text_clean_basic", "text_clean_normalized", "text_model_input", "text_eda_input"]
    train_sample = train_pp.assign(split_source="train")[["split_source"] + sample_cols]
    test_sample = test_pp.assign(split_source="test")[["split_source"] + sample_cols]
    pd.concat([train_sample, test_sample], ignore_index=True).head(args.sample_n).to_csv(
        sample_output, index=False, encoding="utf-8-sig"
    )

    preprocessing_log = {
        "seed": SEED,
        "input_files": {"train": str(train_input), "test": str(test_input)},
        "rows_processed": {
            "train": int(len(train_pp)),
            "test": int(len(test_pp)),
            "total": int(len(train_pp) + len(test_pp)),
        },
        "main_split_design": "train_test_main_split_70_30",
        "text_column_source": args.text_col,
        "output_columns": sample_cols,
        "slang_normalizer": slang_normalizer_name,
        "emoji_handler": emoji_handler_name,
        "rules_applied": [
            "case_folding",
            "remove_url",
            "remove_mention",
            "hashtag_symbol_removed_word_kept",
            "emoji_handling",
            "normalize_repeated_characters",
            "slang_normalization",
            "stopword_removal_for_eda",
            "preserve_domain_tokens",
            "manual_token_overrides_optional",
            "trim_extra_spaces",
        ],
        "override_config_path": str(override_config_path),
        "preserve_tokens": sorted(list(preserve_tokens)),
        "token_overrides_count": len(token_overrides),
        "eda_stopword_count": len(stopword_set),
        "eda_stopword_source": "Sastrawi + stopwordsiso(id)",
        "eda_extra_stopwords_count": len(override_config["eda_extra_stopwords"]),
        "backup_info": backup_info,
        "timestamp": datetime.now().isoformat(),
    }
    write_json(preprocessing_log, log_output)

    print(f"[OK] Preprocessed train CSV: {train_output}")
    print(f"[OK] Preprocessed test CSV: {test_output}")
    print(f"[OK] Combined preprocessed Excel: {combined_output_xlsx}")
    print(f"[OK] Combined preprocessed CSV: {combined_output_csv}")
    print(f"[OK] Preprocessing log: {log_output}")
    print(f"[OK] Samples: {sample_output}")
    print(f"[INFO] Rows processed -> train: {len(train_pp)} | test: {len(test_pp)} | total: {len(combined)}")


if __name__ == "__main__":
    main()
