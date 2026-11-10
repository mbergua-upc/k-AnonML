"""
Bridge between the experiment script and the ARX anonymization tool
(https://arx.deidentifier.org).

`get_result_arx` is the entry point used by baseline_with_repetitions.py for the
arx_sample, arx_optimal and arx_bottom_up methods. ARX is a Java library, so the
function runs it in a subprocess:
  1. Writes the raw rows to a temporary CSV file (`utils.export_to_csv`).
  2. Launches `java/ArxRunner.java` with the ARX jar, passing k, the suppression
     limit, the search algorithm, the generalization hierarchies, the
     quasi-identifiers (QI) and, optionally, l-diversity / t-closeness and the
     sensitive attribute. A JDK 21 is located with `utils.find_java`.
  3. Reads the anonymized CSV back (`utils.read_arx_output`) and returns it as a
     list of rows with the same column order as the input.

The wrapper method names (SAMPLE, OPTIMAL, BOTTOM_UP) are mapped to ARX
algorithms in config.py. A timeout, a failure of ARX or a missing jar raises a
RuntimeError.
"""
import os
import subprocess
import tempfile

from .config import (ARX_JAR_PATH, ARX_RUNNER_PATH, METHOD_TO_ALGORITHM,
                     AVAILABLE_METHODS, TIMEOUTS)
from .utils import export_to_csv, find_java, read_arx_output


def get_result_arx(raw_data, header, k, gen_path, dataset, qi_index,
                   arx_method='OPTIMAL', suppression=0, timeout=None,
                   sensitive=None, l_diversity=None, l_variant='distinct',
                   t_closeness=None):
    """
    Data anonymization using ARX (k-anonymity, optionally with l-diversity and t-closeness)

    Args:
        raw_data: Non-processed data (list of lists of str, all columns)
        header: List of column names (same order as raw_data)
        k: Desired k-anonymity level
        gen_path: Path to the folder containing the generalization hierarchies (one CSV per QI)
        dataset: Name of the dataset (used to locate the hierarchies)
        qi_index: List of column indices corresponding to the quasi-identifiers
        arx_method: ARX algorithm to use (SAMPLE, OPTIMAL, BOTTOM_UP)
        suppression: Maximum percentage of rows that can be suppressed (0-100)
        timeout: Maximum time in seconds to wait for ARX to finish (None = no limit)
        sensitive: Name of the sensitive attribute (required for l-diversity / t-closeness)
        l_diversity: Desired l-diversity level (None = not used)
        l_variant: l-diversity variant (distinct, entropy, recursive)
        t_closeness: Desired t-closeness level (None = not used)

    Returns:
        Anonymized data (list of lists of str, same columns as raw_data)
    """
    arx_method = arx_method.upper()
    if arx_method not in AVAILABLE_METHODS:
        raise ValueError(f"Método ARX desconocido: {arx_method}. Disponibles: {AVAILABLE_METHODS}")
    if (l_diversity is not None or t_closeness is not None) and sensitive not in header:
        raise ValueError(f"l-diversity / t-closeness requieren un atributo sensible válido, recibido: {sensitive}")
    if not 0 <= suppression <= 100:
        raise ValueError("suppression debe estar entre 0 y 100")
    if not os.path.isfile(ARX_JAR_PATH):
        raise RuntimeError(f"ARX jar no encontrado: {ARX_JAR_PATH} (define ARX_JAR)")
    timeout = timeout or TIMEOUTS[arx_method]

    qi_names = [header[i] for i in qi_index]
    hierarchy_prefix = os.path.abspath(gen_path + dataset)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, 'input.csv')
        output_csv = os.path.join(tmpdir, 'output.csv')
        export_to_csv(raw_data, header, input_csv)

        cmd = [
            find_java(), '-cp', ARX_JAR_PATH, ARX_RUNNER_PATH,
            '--input', input_csv, '--output', output_csv, '--k', str(k),
            '--suppression', str(suppression / 100),
            '--algorithm', METHOD_TO_ALGORITHM[arx_method],
            '--hierarchy-prefix', hierarchy_prefix, '--qi', ','.join(qi_names),
        ]
        if sensitive is not None:
            cmd += ['--sensitive', sensitive]
        if l_diversity is not None:
            cmd += ['--l', str(l_diversity), '--l-variant', l_variant]
        if t_closeness is not None:
            cmd += ['--t', str(t_closeness)]
        try:
            subprocess.run(cmd, timeout=timeout, check=True, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ARX timeout ({timeout}s) para k={k}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ARX falló para k={k} (código {e.returncode}):\n{e.stderr}")

        return read_arx_output(output_csv)
