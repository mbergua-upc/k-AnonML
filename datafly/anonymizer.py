"""
Bridge between the experiment script and the Datafly algorithm (datafly.py).

`get_result_datafly` is the entry point used by baseline_with_repetitions.py for
the datafly method. Datafly is a greedy full-domain generalization heuristic: while
more than k records are in groups of fewer than k identical quasi-identifiers (QI),
it generalizes, one level up in its hierarchy, the QI with the most distinct values.
The records that are still in groups smaller than k at the end (at most k) are
suppressed.

datafly.py works on files, so the function:
  1. Writes the raw rows to a temporary CSV file and the generalization hierarchies
     of the QI (the same files used by OLA and ARX, `<gen_path><dataset>_hierarchy_<QI>.csv`,
     ';' delimited) to temporary files with the ',' delimiter read by datafly.py.
  2. Runs `CsvTable.anonymize`.
  3. Reads the anonymized rows back. The records suppressed by Datafly are kept in
     the table with all their QI set to '*', as ARX and OLA do, so the result has the
     same rows (and order) and columns as the input.
"""
import csv
import os
import tempfile

from .datafly import CsvTable

SUPPRESSED = '*'


def get_result_datafly(raw_data, header, k, gen_path, dataset, qi_index):
    """
    Data anonymization using Datafly

    Args:
        raw_data: Non-processed data (list of lists of str, all columns, first column is a unique ID)
        header: List of column names (same order as raw_data)
        k: Desired k-anonymity level
        gen_path: Path to the folder containing the generalization hierarchies (one CSV per QI)
        dataset: Name of the dataset (used to locate the hierarchies)
        qi_index: List of column indices corresponding to the quasi-identifiers

    Returns:
        Anonymized data (list of lists of str, same columns as raw_data)
    """
    qi_names = [header[i] for i in qi_index]
    id_index = header.index('ID')
    if len({row[id_index] for row in raw_data}) != len(raw_data):
        raise ValueError("Datafly needs a unique ID column to identify the suppressed records")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, 'input.csv')
        output_csv = os.path.join(tmpdir, 'output.csv')
        with open(input_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(raw_data)

        dgh_paths = {}
        for name in qi_names:
            source = os.path.join(gen_path, f'{dataset}_hierarchy_{name}.csv')
            dgh_paths[name] = os.path.join(tmpdir, f'{name}_dgh.csv')
            with open(source, 'r', encoding='utf-8', newline='') as src, \
                    open(dgh_paths[name], 'w', encoding='utf-8', newline='') as dst:
                csv.writer(dst).writerows(csv.reader(src, delimiter=';'))

        table = CsvTable(input_csv, dgh_paths)
        try:
            table.anonymize(qi_names, k, output_csv, v=False)
        finally:
            table.table.close()

        with open(output_csv, 'r', encoding='utf-8', newline='') as f:
            anonymized = {row[id_index]: row for row in csv.reader(f)}

    result = []
    for row in raw_data:
        anon_row = anonymized.get(row[id_index])
        if anon_row is None:
            anon_row = list(row)
            for i in qi_index:
                anon_row[i] = SUPPRESSED
        result.append(anon_row)
    return result
