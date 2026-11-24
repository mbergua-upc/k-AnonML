"""
Generate plots for the metrics in the results tables.
Plot for each dataset and classifier, the metric vs k for each anonymizer, 
with the baselines (original and zero-rule) as horizontal lines.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "marta_results"
RESULTS = ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent

DATASETS = {"adult": "ADULT", "cahousing": "CAHOUSING", "cmc": "CMC", "mgm": "MGM"}
CLASSIFIERS = {"rf": "RF", "xgb": "XGBoost", "knn": "k-NN", "svm": "SVM"}
ANONYMIZERS = {"mondrian": "Mondrian", "ola": "OLA", "tdg": "TDG", "cb": "CB", "arx_sample": "ARX genetic", "arx_optimal": "ARX Flash", "arx_bottom_up": "ARX Lightning", "incognito": "Incoognito"}
COLORS = {"mondrian": "tab:red", "ola": "tab:orange", "tdg": "tab:green", "cb": "tab:blue", "arx_sample": "tab:purple", "arx_optimal": "tab:brown", "arx_bottom_up": "tab:pink", "incognito": "tab:gray"}
ANONYMIZERS = {"arx_sample": "ARX genetic", "mdav": "MDAV", "mondrian": "Mondrian", "ola": "OLA", "datafly": "Datafly", "incognito": "Incoognito"}
COLORS = {"arx_sample": "tab:purple", "mdav": "tab:gray", "mondrian": "tab:red", "ola": "tab:orange", "datafly": "tab:blue", "incognito": "tab:green"}


ANONYMIZERS = {"mondrian": "Mondrian", "arx_sample": "ARX optimal"}
COLORS = {"mondrian": "tab:red", "arx_sample": "tab:blue"}
# metrica en la tabla de resultados, columna en el baseline, titulo, limite eje y
METRICS = {
    "accuracy": ("accuracy", "accuracy", "Accuracy (%)", (0, 100)),
    "precision": ("precision", "precision", "Precision", (0, 1)),
    "recall": ("recall", "recall", "Recall", (0, 1)),
    "f1_score": ("f1", "f1_score", "F1 score", (0, 1)),
}

START_K = 2
STOP_K = {"adult": 15000, "cahousing": 10000, "cmc": 735, "mgm": 415}
STEP_K = {"adult": 198, "cahousing": 132, "cmc": 10, "mgm": 6}


def load_table(dataset, anonymizer, classifier):
    folder = RESULTS / dataset / anonymizer / classifier
    files = sorted(folder.glob(f"{dataset}_{anonymizer}_{classifier}_k_*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay tabla en {folder}")
    df = pd.read_csv(files[0])
    df["k"] = range(START_K, STOP_K[dataset] + 1, STEP_K[dataset])
    return df


def main():
    baseline = pd.read_csv(RESULTS / "original_baseline_summary.csv")
    tables = {
        (d, a, c): load_table(d, a, c)
        for d in DATASETS
        for a in ANONYMIZERS
        for c in CLASSIFIERS
    }

    for name, (col, base_col, title, ylim) in METRICS.items():
        fig, axes = plt.subplots(4, 4, figsize=(30, 16), sharex=True, sharey=True)
        for i, (c, c_label) in enumerate(CLASSIFIERS.items()):
            for j, (d, d_label) in enumerate(DATASETS.items()):

                ax = axes[i, j]
                for a in ANONYMIZERS:
                    df = tables[(d, a, c)]
                    ax.plot(df["k"], df[col], color=COLORS[a], linewidth=1.5)

                base = baseline[(baseline["dataset"] == d) & (baseline["classifier"] == c)]
                zero = base.loc[base["predictor"] == "zero_rule_baseline", base_col].iloc[0]
                orig = base.loc[base["predictor"] == "original_data", base_col].iloc[0]
                ax.axhline(zero, color="black", linestyle=":", linewidth=1.5)
                ax.axhline(orig, color="gray", linestyle="--", linewidth=1.5)

                ax.set_xlim(0, STOP_K[d])
                ax.set_ylim(*ylim)
                ax.set_title(f"{d_label} - {c_label}", fontsize=12)
                ax.grid(alpha=0.3)
                if i == 3:
                    ax.set_xlabel("k")
                if j == 0:
                    ax.set_ylabel(title)

        handles = [
            Line2D([0], [0], color=COLORS[a], linewidth=2, label=label)
            for a, label in ANONYMIZERS.items()
        ]
        handles += [
            Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="Zero-Rule baseline"),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Non-Anonymized baseline"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=13, frameon=False)
        fig.suptitle(f"{title} vs k", fontsize=18)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        out = OUT_DIR / f"{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Guardada {out}")


if __name__ == "__main__":
    main()
