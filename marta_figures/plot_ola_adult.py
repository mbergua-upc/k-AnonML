"""Generate a figure with the F1 score of OLA on ADULT dataset, for each 
classifier (knn, rf, xgboost, svm) and metric (prec, gweight, aecs, dm), 
with different suppression levels (s = 0, 3, 6, 9)."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

DATASET = "adult"
CLASSIFIERS = {"rf": "RF", "xgb": "XGBoost", "knn": "k-NN", "svm": "SVM"}
METRICS = {"prec": "Prec", "gweight": "GWeight", "aecs": "AECS", "dm": "DM"}
# suprimidos (%) -> color (verdes y morados)
SUPPRESSIONS = {0: "darkgreen", 3: "limegreen", 6: "mediumpurple", 9: "indigo"}


def load_f1(results_dir, metric, classifier, s):
    folder = results_dir / metric / DATASET / "ola" / classifier
    files = sorted(folder.glob(f"{DATASET}_ola_{classifier}_k_*_s_{s}.csv"))
    if not files:
        print(f"AVISO: falta {folder} (s={s})")
        return None
    df = pd.read_csv(files[0])
    df["k"] = range(2, 2 + len(df))  # la primera fila es k=2
    return df


def load_baselines(results_dir):
    """Devuelve {clasificador: (zero_rule_f1, original_f1)}"""
    baselines = {}
    for classifier in CLASSIFIERS:
        files = sorted(results_dir.glob(f"*/original_baseline/{DATASET}_{classifier}.csv"))
        if not files:
            raise FileNotFoundError(f"No hay baseline de {DATASET} {classifier} en {results_dir}")
        df = pd.read_csv(files[0]).set_index("predictor")
        baselines[classifier] = (
            df.loc["zero_rule_baseline", "f1_score"],
            df.loc["original_data", "f1_score"],
        )
    return baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "ola_experiments")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "ola_adult_f1_score.png")
    args = parser.parse_args()

    baselines = load_baselines(args.results_dir)

    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True, sharey=True)
    for i, (c, c_label) in enumerate(CLASSIFIERS.items()):
        zero, orig = baselines[c]
        for j, (m, m_label) in enumerate(METRICS.items()):
            ax = axes[i, j]
            for s, color in SUPPRESSIONS.items():
                df = load_f1(args.results_dir, m, c, s)
                if df is not None:
                    ax.plot(df["k"], df["f1"], color=color, linewidth=1.5)
            ax.axhline(zero, color="black", linestyle=":", linewidth=1.5)
            ax.axhline(orig, color="gray", linestyle="--", linewidth=1.5)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)
            ax.set_title(f"{c_label} - {m_label}", fontsize=12)
            ax.grid(alpha=0.3)
            if i == 3:
                ax.set_xlabel("k")
            if j == 0:
                ax.set_ylabel("F1 score")

    handles = [
        Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="Zero-Rule baseline"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Non-Anonymized baseline"),
    ]
    handles += [
        Line2D([0], [0], color=color, linewidth=2, label=f"{s}% suppression")
        for s, color in SUPPRESSIONS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=13, frameon=False)
    fig.suptitle("OLA on ADULT: F1 score vs k", fontsize=18)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Guardada {args.output}")


if __name__ == "__main__":
    main()
