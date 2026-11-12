"""Histogramas del numero de clases de equivalencia segun su homogeneidad (Mondrian):
  1. homogeneity_cahousing_mondrian_rf.png: RF en CAHOUSING, k = 78 y 79
  2. homogeneity_mgm_mondrian_knn.png: k-NN en MGM, k = 58, 59, 62 y 72

Para cada k se reproduce el pipeline de baseline_with_repetitions.py sobre la tabla
anonimizada de marta_results: se entrena el clasificador con el mismo split (30 % test,
random_state=42) y se predicen los registros de test. La homogeneidad de una clase de
equivalencia (registros con los mismos valores de QI anonimizados) es la fraccion de sus
registros de test que pertenecen a la clase predicha mayoritaria (1 = todos predicen lo
mismo). Solo cuentan las clases con registros de test. La linea horizontal (eje derecho) es
el F1 score de marta_results para ese k."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
RESULTS = ROOT / "marta_results"
sys.path.insert(0, str(ROOT))
import utils.utility as util  # noqa: E402

# misma configuracion que baseline_with_repetitions.py
CONFIG = {
    "cahousing": {
        "qi": ["longitude", "latitude", "housing_median_age", "median_income", "median_house_value"],
        "is_cat": [False, False, False, False, False],
        "sa": ["ID", "ocean_proximity"],
        "target": "ocean_proximity",
        "max_numeric": {"longitude": 119.33, "latitude": 37.245, "housing_median_age": 32.5,
                        "median_house_value": 257500, "median_income": 5.2035},
    },
    "mgm": {
        "qi": ["bi_rads_assessment", "age", "shape", "margin", "density"],
        "is_cat": [True, False, True, True, True],
        "sa": ["ID", "severity"],
        "target": "severity",
        "max_numeric": {"age": 50.5},
    },
}
BINS = 10


def convert_numeric(value, new_value):
    # suprimido -> valor maximo; rango a-b -> media (igual que baseline_with_repetitions.py)
    if value == "*":
        return new_value
    try:
        parts = value.split("-")
        if len(parts) == 1 or parts[0] == "" or parts[1] == "":
            return value
        return (float(parts[0]) + float(parts[1])) / 2
    except AttributeError:
        return value


def homogeneity(dataset, classifier, k):
    """Devuelve (homogeneidad de cada clase de equivalencia, F1 recalculado, numero de clases de la variable objetivo)"""
    cfg = CONFIG[dataset]
    path = RESULTS / dataset / "mondrian" / "anon_tables" / "default" / "s_0" / f"{dataset}_anonymized_{k}_0.csv"
    anon = pd.read_csv(path, delimiter=";").sort_values(by=["ID"]).reset_index(drop=True)
    groups = anon.groupby(cfg["qi"], sort=False).ngroup().to_numpy()

    anon[cfg["target"]] = anon[cfg["target"]].astype("category").cat.codes
    y = anon[cfg["target"]]
    X = anon.drop(cfg["sa"], axis=1).astype(object)
    for name, cat in zip(cfg["qi"], cfg["is_cat"]):
        if not cat:
            X[name] = X[name].map(lambda v, n=cfg["max_numeric"].get(name): convert_numeric(v, n))
    for name in cfg["max_numeric"]:
        X[name] = pd.to_numeric(X[name])
    X = pd.get_dummies(X, columns=[n for n, cat in zip(cfg["qi"], cfg["is_cat"]) if cat], drop_first=True)
    X = MinMaxScaler().fit_transform(X)

    X_train, X_test, y_train, y_test, _, test_index = train_test_split(
        X, y, np.arange(len(y)), test_size=0.3, random_state=42)
    clf = util.create_classifier(classifier, dataset)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    f1 = util.get_classifier_metrics(np.asarray(y_test), pred)[3]

    test = pd.DataFrame({"group": groups[test_index], "pred": pred})
    values = test.groupby("group")["pred"].agg(lambda p: p.value_counts().iloc[0] / len(p))
    return values.to_numpy(), f1, y.nunique()


def plot(dataset, classifier, classifier_label, ks, layout, output):
    table = pd.read_csv(RESULTS / dataset / "mondrian" / classifier / f"{dataset}_mondrian_{classifier}_k_100.csv")
    fig, axes = plt.subplots(*layout, figsize=(7 * layout[1], 5 * layout[0]), squeeze=False)
    for ax, k in zip(axes.ravel(), ks):
        values, f1_computed, n_classes = homogeneity(dataset, classifier, k)
        f1 = table["f1"].iloc[k - 2]  # la primera fila es k=2
        print(f"{dataset} {classifier} k={k}: {len(values)} clases de equivalencia con test, "
              f"F1 tabla={f1:.4f}, F1 recalculado={f1_computed:.4f}")
        ax.hist(values, bins=np.linspace(1 / n_classes, 1, BINS + 1), color="tab:blue", edgecolor="white")
        ax.set_xlabel("Homogeneity")
        ax.set_ylabel("Number of equivalence classes")
        ax.set_title(f"k = {k}")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2 = ax.twinx()
        ax2.axhline(f1, color="tab:red", linewidth=2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("F1 score", color="tab:red")
    handles = [
        Line2D([0], [0], color="tab:blue", linewidth=6, label="Number of equivalence classes"),
        Line2D([0], [0], color="tab:red", linewidth=2, label="F1 score"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Mondrian, {classifier_label} on {dataset.upper()}", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Guardada {output}")


def main():
    plot("cahousing", "rf", "RF", [78, 79], (1, 2), OUT_DIR / "homogeneity_cahousing_mondrian_rf.png")
    plot("mgm", "knn", "k-NN", [58, 59, 62, 72], (2, 2), OUT_DIR / "homogeneity_mgm_mondrian_knn.png")


if __name__ == "__main__":
    main()
