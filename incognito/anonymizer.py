"""
Bridge between the experiment script and the Incognito algorithm (Incognito_K,
Incognito_L and Incognito_T, from the UTD Anonymization Toolbox, in the `incognito`
folder).

`get_result_incognito` is the entry point used by baseline_with_repetitions.py for the
incognito_k, incognito_l and incognito_t methods. Incognito is a full-domain
generalization algorithm: it traverses the lattice of the generalization levels of the
quasi-identifiers (QI) and, among the generalizations that satisfy the privacy models, it
selects the one with the most equivalence classes. Records are never suppressed.
  - K: k-anonymity.
  - L: k-anonymity and l-diversity, entropy l-diversity or recursive (c,l)-diversity.
  - T: k-anonymity and t-closeness (equal distance).

The toolbox reads a configuration file in XML whose generalization hierarchies are intervals
of numbers, so the function:
  1. Reads the hierarchies of the QI (the same files used by OLA, ARX and Datafly,
     `<gen_path><dataset>_hierarchy_<QI>.csv`, ';' delimited) and numbers their ground values
     in an order in which the values of every group are consecutive. Every group of every
     level becomes an interval of those numbers.
  2. Writes the data (the numbers of the QI values and of the sensitive value) and the
     configuration file to a temporary folder and runs `incognito.IncognitoRunner` (compiled
     with `compile_incognito`, a JDK 21 is located with `arx_wrapper.utils.find_java`).
  3. Translates every interval of the output back to the label of its group in the
     hierarchy, so the result has the same rows (and order) and columns as the input.
A timeout, a failure of the Java program, a value that is not in a hierarchy or the absence
of a generalization that satisfies the privacy models raises an error.
"""
import csv
import glob
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from arx_wrapper.utils import find_java, find_javac

from .config import (INCOGNITO_BUILD_DIR, INCOGNITO_DIR, INCOGNITO_RUNNER_CLASS, METHODS,
                     SQLITE_JAR_PATH, TIMEOUT)


def compile_incognito():
    """Compiles the Java sources of Incognito into INCOGNITO_BUILD_DIR, only if they changed. Returns the build folder."""
    sources = sorted(glob.glob(os.path.join(INCOGNITO_DIR, '*.java'))
                     + glob.glob(os.path.join(INCOGNITO_DIR, 'toolbox', '**', '*.java'), recursive=True))
    runner_class = os.path.join(INCOGNITO_BUILD_DIR, *INCOGNITO_RUNNER_CLASS.split('.')) + '.class'
    if os.path.isfile(runner_class) and os.path.getmtime(runner_class) >= max(os.path.getmtime(f) for f in sources):
        return INCOGNITO_BUILD_DIR
    if not os.path.isfile(SQLITE_JAR_PATH):
        raise RuntimeError(f"sqlite-jdbc jar not found: {SQLITE_JAR_PATH} (define SQLITE_JDBC_JAR)")
    os.makedirs(INCOGNITO_BUILD_DIR, exist_ok=True)
    try:
        subprocess.run([find_javac(), '-nowarn', '-cp', SQLITE_JAR_PATH, '-d', INCOGNITO_BUILD_DIR, *sources],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Incognito compilation failed (code {e.returncode}):\n{e.stderr}")
    return INCOGNITO_BUILD_DIR


def _format(number):
    return f'{number:.4f}'


class Hierarchy:
    """Generalization hierarchy of a QI, as the intervals of the numbers given to its ground values."""

    def __init__(self, path):
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rows = [row for row in csv.reader(f, delimiter=';') if row]
        if len({len(row) for row in rows}) != 1:
            raise ValueError(f"All rows of the hierarchy {path} must have the same number of levels")
        self.levels = len(rows[0]) - 1  # generalization levels, the last one is the root
        # the first row of a ground value is kept
        paths = {}
        for row in rows:
            paths.setdefault(row[0], tuple(reversed(row[1:])))  # root, ..., level 1
        # sorted by the path from the root, so the values of every group are consecutive
        self.values = sorted(paths, key=lambda value: (paths[value], value))
        self.number = {value: i for i, value in enumerate(self.values)}

        # (level, path from the root to the group) -> [label, first number, last number]
        groups = {}
        for i, value in enumerate(self.values):
            path = paths[value]
            for level in range(1, self.levels + 1):
                prefix = path[:self.levels - level + 1]
                group = groups.setdefault((level, prefix), [prefix[-1], i, i])
                group[2] = i
        self.groups = groups

        # Intervals are padded by a distance that grows with the level, so that nodes of
        # different levels that contain the same values have different names (the toolbox
        # identifies a node by its interval) and the padding never reaches another value.
        pad = {level: 0.4 * level / self.levels for level in range(1, self.levels + 1)}
        self.interval = {key: f'[{_format(first - pad[key[0]])}:{_format(last + pad[key[0]])}]'
                         for key, (label, first, last) in groups.items()}
        self.label = {self.interval[key]: group[0] for key, group in groups.items()}
        self.label.update({f'[{float(i)!r}]': value for i, value in enumerate(self.values)})

    def to_xml(self, parent):
        """Adds the value generalization hierarchy as <vgh> and nested <node> elements."""
        nodes = {}
        for (level, path), _ in sorted(self.groups.items(), key=lambda item: -item[0][0]):
            if level == self.levels:
                nodes[(level, path)] = ET.SubElement(parent, 'vgh', value=self.interval[(level, path)])
            else:
                # the parent group is the one with the same path without the last element
                parent_key = (level + 1, path[:-1])
                nodes[(level, path)] = ET.SubElement(nodes[parent_key], 'node', value=self.interval[(level, path)])


def get_result_incognito(raw_data, header, k, gen_path, dataset, qi_index, version='K',
                         sensitive=None, l_diversity=None, l_variant='entropy', c=3.0,
                         t_closeness=None, timeout=None):
    """
    Data anonymization using Incognito (k-anonymity, optionally with l-diversity or t-closeness)

    Args:
        raw_data: Non-processed data (list of lists of str, all columns)
        header: List of column names (same order as raw_data)
        k: Desired k-anonymity level
        gen_path: Path to the folder containing the generalization hierarchies (one CSV per QI)
        dataset: Name of the dataset (used to locate the hierarchies)
        qi_index: List of column indices corresponding to the quasi-identifiers
        version: K (k-anonymity), L (k-anonymity and l-diversity) or T (k-anonymity and t-closeness)
        sensitive: Name of the sensitive attribute (required by versions L and T)
        l_diversity: Desired l (> 1) of version L
        l_variant: 'entropy' (entropy l-diversity) or 'recursive' (recursive (c,l)-diversity) for version L
        c: c of the recursive (c,l)-diversity
        t_closeness: Desired t (0-1) of version T
        timeout: Maximum time in seconds to wait for Incognito to finish (None = default timeout)

    Returns:
        Anonymized data (list of lists of str, same columns as raw_data)
    """
    version = version.upper()
    if version not in METHODS:
        raise ValueError(f"Unknown Incognito version: {version}. Available: {list(METHODS)}")
    if version != 'K' and sensitive not in header:
        raise ValueError(f"l-diversity / t-closeness need a valid sensitive attribute, received: {sensitive}")
    if version == 'L' and (l_diversity is None or l_diversity <= 1):
        raise ValueError("l-diversity needs l > 1")
    if version == 'L' and l_variant not in ('entropy', 'recursive'):
        raise ValueError(f"Unknown l-diversity variant: {l_variant}. Available: entropy, recursive")
    if version == 'T' and (t_closeness is None or not 0 < t_closeness <= 1):
        raise ValueError("t-closeness needs 0 < t <= 1")
    timeout = timeout or TIMEOUT
    build_dir = compile_incognito()

    qi_names = [header[i] for i in qi_index]
    hierarchies = []
    for i, name in zip(qi_index, qi_names):
        hierarchy = Hierarchy(os.path.join(gen_path, f'{dataset}_hierarchy_{name}.csv'))
        missing = {row[i] for row in raw_data} - set(hierarchy.number)
        if missing:
            raise ValueError(f"Values {sorted(missing)[:5]} are not in the hierarchy of attribute '{name}'")
        hierarchies.append(hierarchy)

    columns = list(qi_index)
    if version != 'K':
        sensitive_index = header.index(sensitive)
        sensitive_codes = {value: i for i, value in enumerate(sorted({row[sensitive_index] for row in raw_data}))}
        columns.append(sensitive_index)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.csv')
        output_file = os.path.join(tmpdir, 'output.csv')
        config_file = os.path.join(tmpdir, 'config.xml')

        # QI and sensitive values are written as the numbers used in the configuration
        with open(input_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            for row in raw_data:
                numbers = [hierarchies[j].number[row[i]] for j, i in enumerate(qi_index)]
                if version != 'K':
                    numbers.append(sensitive_codes[row[sensitive_index]])
                writer.writerow(numbers)

        root = ET.Element('config', method=METHODS[version], k=str(k))
        if version == 'L':
            root.set('l', str(l_diversity))
            root.set('c', str(c) if l_variant == 'recursive' else '0')  # c <= 0: entropy l-diversity
        elif version == 'T':
            root.set('t', str(t_closeness))
        ET.SubElement(root, 'input', filename=input_file, separator=',')
        ET.SubElement(root, 'output', filename=output_file, format='genVals')
        qid = ET.SubElement(root, 'qid')
        for position, (name, hierarchy) in enumerate(zip(qi_names, hierarchies)):
            hierarchy.to_xml(ET.SubElement(qid, 'att', index=str(position), name=name))
        if version != 'K':
            sens = ET.SubElement(root, 'sens')
            att = ET.SubElement(sens, 'att', index=str(len(qi_names)), name=sensitive)
            mapping = ET.SubElement(att, 'map')
            for code in sensitive_codes.values():
                ET.SubElement(mapping, 'entry', cat=str(code), int=str(code))
        ET.ElementTree(root).write(config_file, encoding='utf-8')

        cmd = [find_java(), '-cp', os.pathsep.join([build_dir, SQLITE_JAR_PATH]), INCOGNITO_RUNNER_CLASS, config_file]
        try:
            subprocess.run(cmd, timeout=timeout, check=True, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Incognito timeout ({timeout}s) for k={k}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Incognito ({version}) failed for k={k} (code {e.returncode}):\n{e.stderr}")

        with open(output_file, 'r', encoding='utf-8', newline='') as f:
            output = [line.rstrip('\r\n').split(',') for line in f if line.strip()]

    if len(output) != len(raw_data):
        raise RuntimeError(f"Incognito returned {len(output)} rows for {len(raw_data)} records")
    result = []
    for row, anon_row in zip(raw_data, output):
        new_row = list(row)
        for j, i in enumerate(qi_index):
            new_row[i] = hierarchies[j].label[anon_row[j]]
        result.append(new_row)
    return result
