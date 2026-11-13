
from pathlib import Path
from itertools import combinations
import pandas as pd
import hashlib

# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results/mgm")

ALGORITHMS = [
    "arx_bottom_up",
    "arx_optimal",
    "arx_sample"
]

ANON_DIR = "anon_tables/sensitive-severity_l_diversity-None_l_variant-distinct_t_closeness-None/s_3"

OUTPUT_FILE = RESULTS_DIR / "table_comparison.csv"


# ============================================================
# FUNCTIONS
# ============================================================

def load_table(path):
    """Load an anonymized CSV as strings."""
    return pd.read_csv(path, dtype=str).fillna("")


def exact_comparison(df1, df2):
    """Check whether two tables are exactly identical."""
    return df1.equals(df2)


def order_independent_comparison(df1, df2):
    """
    Check whether two tables contain exactly the same rows,
    independently of their order.
    """

    if set(df1.columns) != set(df2.columns):
        return False

    # Align columns
    df2 = df2[df1.columns]

    if len(df1) != len(df2):
        return False

    # Sort rows using all columns
    a = df1.sort_values(
        by=list(df1.columns)
    ).reset_index(drop=True)

    b = df2.sort_values(
        by=list(df2.columns)
    ).reset_index(drop=True)

    return a.equals(b)


def row_similarity(df1, df2):
    """
    Calculate row similarity using multiset intersection.
    Repeated rows are counted correctly.
    """

    if set(df1.columns) != set(df2.columns):
        return 0.0

    df2 = df2[df1.columns]

    if len(df1) == 0 or len(df2) == 0:
        return 0.0

    rows1 = df1.value_counts(dropna=False)
    rows2 = df2.value_counts(dropna=False)

    common = sum(
        min(rows1.get(row, 0), rows2.get(row, 0))
        for row in rows1.index
    )

    return common / max(len(df1), len(df2)) * 100


def compare_tables(path1, path2):

    df1 = load_table(path1)
    df2 = load_table(path2)

    return {
        "exact": exact_comparison(df1, df2),
        "same_rows": order_independent_comparison(df1, df2),
        "similarity": row_similarity(df1, df2),
        "rows_1": len(df1),
        "rows_2": len(df2)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    tables = {}

    # Load available tables
    for algorithm in ALGORITHMS:

        folder = RESULTS_DIR / algorithm / ANON_DIR

        if not folder.exists():
            print(f"WARNING: Folder not found: {folder}")
            continue

        for path in (folder).glob("mgm_anonymized_*_0.csv"):

            # Extract k from filename
            k = path.stem.replace(
                "mgm_anonymized_", ""
            ).replace("_0", "")

            tables.setdefault(k, {})[algorithm] = path

    results = []

    # Compare algorithms for each k
    for k, algorithm_tables in sorted(
        tables.items(), key=lambda x: int(x[0])
    ):

        print(f"\n========== k = {k} ==========")

        for alg1, alg2 in combinations(
            algorithm_tables.keys(), 2
        ):

            path1 = algorithm_tables[alg1]
            path2 = algorithm_tables[alg2]

            comparison = compare_tables(path1, path2)

            result = {
                "k": int(k),
                "algorithm_1": alg1,
                "algorithm_2": alg2,
                **comparison
            }

            results.append(result)

            print(
                f"{alg1} vs {alg2}: "
                f"Exact={comparison['exact']} | "
                f"Same rows={comparison['same_rows']} | "
                f"Similarity={comparison['similarity']:.2f}%"
            )

    # Export results
    if results:

        df_results = pd.DataFrame(results)

        df_results.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\n================================")
        print("Comparison completed.")
        print(f"Results saved to: {OUTPUT_FILE}")
        print("================================")


if __name__ == "__main__":
    main()
