"""
Bridge between the experiment script and the MDAV algorithm (Maximum Distance to
Average Vector), implemented in Java in the `mdav` folder.

`get_result_mdav` is the entry point used by baseline_with_repetitions.py for the
mdav method. MDAV is a microaggregation algorithm: it partitions the records into
clusters of at least k records and replaces the quasi-identifiers (QI) of every
record by the representative of its cluster, which makes the table k-anonymous.
  1. The QI are encoded as a numeric matrix: numeric QI are min-max scaled to
     [0, 1] and categorical QI are one-hot encoded, with every category column set
     to 1/sqrt(2) so that two different categories are at distance 1, the same as
     the extremes of a numeric QI.
  2. The matrix is written to a temporary CSV file and `mdav.MdavRunner` (compiled
     with `compile_mdav`, a JDK 21 is located with `arx_wrapper.utils.find_java`)
     returns the cluster of each record.
  3. In every cluster, numeric QI are replaced by their mean and categorical QI
     by their mode. The other attributes are kept unchanged.

The result has the same rows (and order) and columns as the input. A timeout, a
failure of the Java program or a missing JDK raises a RuntimeError.
"""
import glob
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from arx_wrapper.utils import find_java, find_javac

from .config import (MDAV_BUILD_DIR, MDAV_EXCLUDED_SOURCES, MDAV_RUNNER_CLASS,
                     MDAV_SOURCE_DIR, TIMEOUT)

CATEGORY_WEIGHT = 1 / np.sqrt(2)


def compile_mdav():
    """Compiles the Java sources of MDAV into MDAV_BUILD_DIR, only if they changed. Returns the build folder."""
    sources = [f for f in sorted(glob.glob(os.path.join(MDAV_SOURCE_DIR, '*.java')))
               if os.path.basename(f) not in MDAV_EXCLUDED_SOURCES]
    runner_class = os.path.join(MDAV_BUILD_DIR, *MDAV_RUNNER_CLASS.split('.')) + '.class'
    if os.path.isfile(runner_class) and os.path.getmtime(runner_class) >= max(os.path.getmtime(f) for f in sources):
        return MDAV_BUILD_DIR
    os.makedirs(MDAV_BUILD_DIR, exist_ok=True)
    try:
        subprocess.run([find_javac(), '-d', MDAV_BUILD_DIR, *sources], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MDAV compilation failed (code {e.returncode}):\n{e.stderr}")
    return MDAV_BUILD_DIR


def encode_qi(df, qi_names, is_cat):
    """Numeric matrix used to compute the distances: scaled numeric QI and one-hot encoded categorical QI."""
    columns = []
    for name, cat in zip(qi_names, is_cat):
        if cat:
            columns.append(pd.get_dummies(df[name]).to_numpy(dtype=float) * CATEGORY_WEIGHT)
        else:
            values = pd.to_numeric(df[name]).to_numpy(dtype=float)
            span = values.max() - values.min()
            columns.append(((values - values.min()) / span if span > 0 else np.zeros_like(values))[:, None])
    return np.hstack(columns)


def aggregate_clusters(df, qi_names, is_cat, clusters):
    """Replaces every QI by the mean (numeric) or the mode (categorical) of the cluster of the record."""
    result = df.copy()
    for name, cat in zip(qi_names, is_cat):
        if cat:
            counts = pd.DataFrame({'cluster': clusters, 'value': df[name].to_numpy()}).groupby(
                ['cluster', 'value']).size().rename('count').reset_index()
            # on ties the smallest value wins
            mode = (counts.sort_values(['cluster', 'count', 'value'], ascending=[True, False, True])
                    .drop_duplicates('cluster').set_index('cluster')['value'])
            result[name] = mode.reindex(clusters).to_numpy()
        else:
            result[name] = pd.to_numeric(df[name]).groupby(clusters).transform('mean').round(4)
    return result


def get_result_mdav(raw_data, header, k, qi_index, is_cat, timeout=None):
    """
    Data anonymization using MDAV microaggregation (k-anonymity on the quasi-identifiers)

    Args:
        raw_data: Non-processed data (list of lists of str, all columns)
        header: List of column names (same order as raw_data)
        k: Desired k-anonymity level
        qi_index: List of column indices corresponding to the quasi-identifiers
        is_cat: List of booleans, True if the quasi-identifier (same order as qi_index) is categorical
        timeout: Maximum time in seconds to wait for MDAV to finish (None = default timeout)

    Returns:
        Anonymized data (list of lists of str, same columns as raw_data)
    """
    if len(raw_data) < k:
        raise ValueError(f"k={k} is greater than the number of records ({len(raw_data)})")
    timeout = timeout or TIMEOUT
    build_dir = compile_mdav()

    df = pd.DataFrame(raw_data, columns=header)
    qi_names = [header[i] for i in qi_index]
    matrix = encode_qi(df, qi_names, is_cat)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, 'input.csv')
        output_txt = os.path.join(tmpdir, 'clusters.txt')
        np.savetxt(input_csv, matrix, fmt='%.10g', delimiter=',')

        cmd = [find_java(), '-cp', build_dir, MDAV_RUNNER_CLASS,
               '--input', input_csv, '--output', output_txt, '--k', str(k)]
        try:
            subprocess.run(cmd, timeout=timeout, check=True, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"MDAV timeout ({timeout}s) for k={k}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MDAV failed for k={k} (code {e.returncode}):\n{e.stderr}")
        clusters = np.loadtxt(output_txt, dtype=int)

    return aggregate_clusters(df, qi_names, is_cat, clusters).astype(str).values.tolist()
