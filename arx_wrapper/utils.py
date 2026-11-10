"""
Helper functions of the ARX wrapper.

  - `find_java`: looks for a Java installation of version 21 or higher (the ARX
    jar requires it) in the ARX_JAVA and JAVA_HOME environment variables, the
    usual Windows install folders and the PATH.
  - `export_to_csv`: writes the dataset (list of rows plus header) to a CSV file
    with ';' as delimiter, the format read by ArxRunner.java.
  - `read_arx_output`: reads the CSV produced by ARX and returns its rows
    (without the header) as a list of lists of strings.
"""
import csv
import glob
import os
import shutil
import subprocess


def _java_major_version(java):
    try:
        out = subprocess.run([java, '-version'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return 0
    text = out.stderr or out.stdout
    parts = text.splitlines()[0].split('"') if text else []
    if len(parts) < 2:
        return 0
    version = parts[1].split('.')
    try:
        return int(version[1]) if version[0] == '1' else int(version[0].split('-')[0])
    except (ValueError, IndexError):
        return 0


def find_java(min_version=21):
    """Searches for a Java installation of version `min_version` or higher."""
    candidates = [os.getenv('ARX_JAVA')]
    if os.getenv('JAVA_HOME'):
        candidates.append(os.path.join(os.environ['JAVA_HOME'], 'bin', 'java.exe'))
        candidates.append(os.path.join(os.environ['JAVA_HOME'], 'bin', 'java'))
    for base in (os.getenv('ProgramFiles', r'C:\Program Files'), os.getenv('ProgramFiles(x86)', '')):
        if base:
            candidates += sorted(glob.glob(os.path.join(base, 'Eclipse Adoptium', 'jdk-*', 'bin', 'java.exe')), reverse=True)
            candidates += sorted(glob.glob(os.path.join(base, 'Java', 'jdk*', 'bin', 'java.exe')), reverse=True)
    candidates.append(shutil.which('java'))
    for java in candidates:
        if java and os.path.isfile(java) and _java_major_version(java) >= min_version:
            return java
    raise RuntimeError(f"Not found a Java installation of version {min_version} or higher (set ARX_JAVA or JAVA_HOME, or install Java 21+).")


def export_to_csv(raw_data, header, filepath, delimiter=';'):
    """Writes the dataset (list of rows plus header) to a CSV file with ';' as delimiter, the format read by ArxRunner.java."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(raw_data)


def read_arx_output(filepath, delimiter=';'):
    """Reads the CSV produced by ARX and returns its rows (without the header) as a list of lists of strings."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)
        return [row for row in reader]
