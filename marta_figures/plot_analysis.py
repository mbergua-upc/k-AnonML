"""Genera tres figuras:
  1. ola_generalisation_adult.png: efecto combinado de generalizacion y supresion de OLA en ADULT
  2. ola_generalisation_mgm.png: lo mismo para MGM
  3. correlation_adult.png: matriz de correlacion de Spearman de ADULT

Las dos primeras usan los resultados de run_ola_experiments.py (results/ola_experiments):
barras = nivel de generalizacion de cada atributo en el nodo elegido por OLA para cada k,
curva de supresion = numero de registros suprimidos (todos los QI a '*' en las tablas anonimizadas),
curva de F1 = F1 score del clasificador. Si faltan resultados de un dataset se avisa y se
pasa a la siguiente figura."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_ola_experiments import QI_INDEX  # noqa: E402

# colores de las curvas (supresion, F1) como en las figuras de referencia
CURVE_COLORS = {"adult": ("tab:green", "tab:blue"), "mgm": ("tab:green", "tab:blue")}
TARGET = {"adult": "salary-class", "mgm": "severity"}
# atributos categoricos entre los QI de ADULT (como IS_CAT2 en baseline_with_repetitions.py)
ADULT_CATEGORICAL = ["sex", "race", "marital-status", "education", "native-country", "workclass", "occupation"]


def read_header(dataset):
    return pd.read_csv(ROOT / "datasets" / dataset / f"{dataset}.csv", sep=";", nrows=0).columns.tolist()


def load_generalisation(results_dir, dataset, metric, s):
    nodes = pd.read_csv(results_dir / "ola_selected_nodes.csv")
    nodes = nodes[(nodes["dataset"] == dataset) & (nodes["metric"] == metric) & (nodes["s"] == s)]
    nodes = nodes.drop_duplicates("k", keep="last").set_index("k").sort_index()
    return nodes["node"].str.split().apply(lambda levels: [int(v) for v in levels])


def count_suppressed(results_dir, dataset, metric, s, k, qi_names):
    path = results_dir / metric / dataset / "ola" / "anon_tables" / f"metric-{metric}" / f"s_{s}" / f"{dataset}_anonymized_{k}_0.csv"
    df = pd.read_csv(path, sep=";", usecols=qi_names, dtype=str)
    # l1sub tambien generaliza a '*', asi que un registro suprimido es el que tiene '*' en todos los QI
    return int((df == "*").all(axis=1).sum())


def plot_generalisation(results_dir, dataset, classifier, metric, s, k_step, output):
    qi_names = [read_header(dataset)[i] for i in QI_INDEX[dataset]]
    try:
        levels = load_generalisation(results_dir, dataset, metric, s)
        f1_files = sorted((results_dir / metric / dataset / "ola" / classifier).glob(
            f"{dataset}_ola_{classifier}_k_*_s_{s}.csv"))
        if levels.empty or not f1_files:
            raise FileNotFoundError("sin resultados")
        f1 = pd.read_csv(f1_files[0])["f1"]
        f1.index = range(2, 2 + len(f1))  # la primera fila es k=2
        ks = [k for k in levels.index if k in f1.index][::k_step]
        suppressed = [count_suppressed(results_dir, dataset, metric, s, k, qi_names) for k in ks]
    except FileNotFoundError as error:
        print(f"AVISO: faltan resultados de {dataset} ({classifier}, {metric}, s={s}): {error}. "
              f"Ejecuta run_ola_experiments.py primero.")
        return

    suppressed_color, f1_color = CURVE_COLORS[dataset]
    bar_colors = sns.color_palette("Set2", len(qi_names))
    width = 0.8 * k_step / len(qi_names)

    fig, ax = plt.subplots(figsize=(20, 6))
    for j, name in enumerate(qi_names):
        offsets = [k + (j - (len(qi_names) - 1) / 2) * width for k in ks]
        ax.bar(offsets, [levels[k][j] for k in ks], width=width, color=bar_colors[j], label=name)
    ax.set_xlabel("k")
    ax.set_ylabel("Generalisation level")
    ax.set_xlim(min(ks) - k_step, max(ks) + k_step)
    max_level = max(max(levels[k]) for k in ks)
    ax.set_ylim(0, max_level + 1)
    ax.set_yticks(range(0, max_level + 2))

    ax_supp = ax.twinx()
    ax_supp.plot(ks, suppressed, color=suppressed_color, linewidth=2)
    ax_supp.set_ylabel("Suppressed records", color=suppressed_color)
    ax_supp.set_ylim(bottom=0)

    ax_f1 = ax.twinx()
    ax_f1.spines["right"].set_position(("outward", 60))
    ax_f1.plot(ks, [f1[k] for k in ks], color=f1_color, linewidth=2)
    ax_f1.set_ylabel("F1 score", color=f1_color)
    ax_f1.set_ylim(0, 1)

    handles = [Patch(color=bar_colors[j], label=name) for j, name in enumerate(qi_names)]
    handles += [
        Line2D([0], [0], color=suppressed_color, linewidth=2, label="Suppressed records"),
        Line2D([0], [0], color=f1_color, linewidth=2, label="F1 score"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    ax.set_title(f"OLA ({metric}, {s}% suppression) on {dataset.upper()} with {classifier.upper()}")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Guardada {output}")


def plot_correlation(output):
    data = pd.read_csv(ROOT / "datasets" / "adult" / "adult.csv", sep=";")
    target = TARGET["adult"]
    qi_names = [data.columns[i] for i in QI_INDEX["adult"]]
    data[target] = data[target].astype("category").cat.codes
    # mismo preprocesado que baseline_with_repetitions.py: one-hot de las categoricas y sin ID
    features = pd.get_dummies(data[qi_names], columns=ADULT_CATEGORICAL, drop_first=True).astype(float)
    features[target] = data[target].astype(float)
    corr = features.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(48, 42))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax, annot=True, fmt=".2f",
                annot_kws={"fontsize": 5}, xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Spearman correlation", "shrink": 0.7})
    ax.tick_params(labelsize=9)
    ax.set_title("Spearman correlation matrix of ADULT (categorical features one-hot encoded)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Guardada {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "ola_experiments")
    parser.add_argument("--classifier", default="rf", choices=["rf", "xgb", "knn", "svm"])
    parser.add_argument("--metric", default="gweight", choices=["prec", "gweight", "aecs", "dm"])
    parser.add_argument("--suppression", type=int, default=3, choices=[0, 3, 6, 9], help="s (%% de supresion)")
    parser.add_argument("--k-step", type=int, default=1, help="dibuja una de cada k-step barras")
    args = parser.parse_args()

    for dataset in ("adult", "mgm"):
        plot_generalisation(args.results_dir, dataset, args.classifier, args.metric, args.suppression,
                            args.k_step, OUT_DIR / f"ola_generalisation_{dataset}.png")
    plot_correlation(OUT_DIR / "correlation_adult.png")


if __name__ == "__main__":
    main()
