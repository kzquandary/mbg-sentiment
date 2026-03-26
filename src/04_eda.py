import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed


def make_label_distribution(df: pd.DataFrame, label_col: str, out_path: Path) -> dict[str, int]:
    counts = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="Set2")
    plt.title("Distribusi Label Sentimen")
    plt.xlabel("Label")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def make_text_length_plots(df: pd.DataFrame, text_col: str, label_col: str, boxplot_path: Path, hist_path: Path) -> dict:
    data = df.copy()
    data["text_length_words"] = data[text_col].fillna("").astype(str).str.split().str.len()

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=data, x=label_col, y="text_length_words", palette="Set3")
    plt.title("Panjang Teks (Kata) per Label")
    plt.xlabel("Label")
    plt.ylabel("Jumlah Kata")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(data=data, x="text_length_words", hue=label_col, kde=True, bins=40, element="step")
    plt.title("Distribusi Panjang Teks")
    plt.xlabel("Jumlah Kata")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    length_stats = (
        data.groupby(label_col)["text_length_words"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(3)
        .to_dict(orient="index")
    )
    return length_stats


def tokenize(text: str) -> list[str]:
    tokens = [t.strip() for t in text.split() if t.strip()]
    return [t for t in tokens if len(t) > 1 and not t.isdigit()]


def top_words_general(df: pd.DataFrame, text_col: str, top_n: int = 30) -> list[tuple[str, int]]:
    counter = Counter()
    for txt in df[text_col].fillna("").astype(str):
        counter.update(tokenize(txt))
    return counter.most_common(top_n)


def plot_top_words_general(top_words: list[tuple[str, int]], out_dir: Path) -> str | None:
    if not top_words:
        return None
    words = [x[0] for x in top_words]
    freqs = [x[1] for x in top_words]
    plt.figure(figsize=(10, 7))
    sns.barplot(x=freqs, y=words, palette="viridis")
    plt.title("Top Kata - Seluruh Data")
    plt.xlabel("Frekuensi")
    plt.ylabel("Kata")
    plt.tight_layout()
    fpath = out_dir / "top_words_general.png"
    plt.savefig(fpath, dpi=200)
    plt.close()
    return str(fpath)


def make_wordcloud_general(top_words: list[tuple[str, int]], out_dir: Path) -> str | None:
    if not top_words:
        return None
    freq_map = {w: c for w, c in top_words}
    wc = WordCloud(width=1200, height=600, background_color="white")
    wc.generate_from_frequencies(freq_map)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud - Seluruh Data")
    plt.tight_layout()
    fpath = out_dir / "wordcloud_general.png"
    plt.savefig(fpath, dpi=200)
    plt.close()
    return str(fpath)


def write_summary_md(
    output_path: Path,
    n_rows: int,
    n_cols: int,
    label_dist: dict[str, int],
    length_stats: dict,
    top_words: list[tuple[str, int]],
    figure_paths: list[str],
) -> None:
    lines = []
    lines.append("# Ringkasan EDA")
    lines.append("")
    lines.append(f"- Waktu proses: {datetime.now().isoformat()}")
    lines.append(f"- Jumlah data: {n_rows}")
    lines.append(f"- Jumlah kolom: {n_cols}")
    lines.append("")
    lines.append("## Distribusi Label")
    for k, v in label_dist.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Statistik Panjang Teks (kata)")
    for label, stats in length_stats.items():
        lines.append(
            f"- {label}: count={stats['count']}, mean={stats['mean']}, median={stats['median']}, min={stats['min']}, max={stats['max']}"
        )
    lines.append("")
    lines.append("## Top Kata Seluruh Data")
    preview = ", ".join([f"{w}({c})" for w, c in top_words[:20]])
    lines.append(f"- {preview}")
    lines.append("")
    lines.append("## File Visualisasi")
    for fp in figure_paths:
        lines.append(f"- {fp}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4 - Exploratory Data Analysis")
    parser.add_argument("--input", type=str, default="data/preprocessed_dataset.csv")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--text-col", type=str, default="text_eda_input")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures")
    parser.add_argument("--summary-output", type=str, default="outputs/eda_summary.md")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    set_global_seed(SEED)
    sns.set_theme(style="whitegrid")

    input_path = Path(args.input)
    figures_dir = Path(args.figures_dir)
    summary_output = Path(args.summary_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ensure_dir(figures_dir)
    ensure_dir(summary_output.parent)

    backup_if_exists(summary_output)

    df = pd.read_csv(input_path)
    if args.label_col not in df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")
    if args.text_col not in df.columns:
        raise KeyError(f"Text column not found: {args.text_col}")

    fig_paths = []
    label_fig = figures_dir / "label_distribution.png"
    label_dist = make_label_distribution(df, args.label_col, label_fig)
    fig_paths.append(str(label_fig))

    boxplot_fig = figures_dir / "text_length_boxplot.png"
    hist_fig = figures_dir / "text_length_histogram.png"
    length_stats = make_text_length_plots(df, args.text_col, args.label_col, boxplot_fig, hist_fig)
    fig_paths.extend([str(boxplot_fig), str(hist_fig)])

    top_words = top_words_general(df, args.text_col, top_n=args.top_n)
    top_words_path = plot_top_words_general(top_words, figures_dir)
    wordcloud_path = make_wordcloud_general(top_words, figures_dir)
    if top_words_path:
        fig_paths.append(top_words_path)
    if wordcloud_path:
        fig_paths.append(wordcloud_path)

    write_summary_md(
        output_path=summary_output,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        label_dist=label_dist,
        length_stats=length_stats,
        top_words=top_words,
        figure_paths=fig_paths,
    )

    print(f"[OK] EDA summary saved: {summary_output}")
    print(f"[OK] Figures generated: {len(fig_paths)} files")


if __name__ == "__main__":
    main()
