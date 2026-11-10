"""
Utilidades del wrapper de ARX
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
    """Busca un java >= min_version: ARX_JAVA, JAVA_HOME, rutas habituales de Windows y PATH."""
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
    raise RuntimeError(f"No se encontró Java >= {min_version} (JDK). Instálalo o define ARX_JAVA con la ruta a java.exe")


def export_to_csv(raw_data, header, filepath, delimiter=';'):
    """Escribe los datos (lista de listas de str) con cabecera en CSV."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(raw_data)


def read_arx_output(filepath, delimiter=';'):
    """Lee el CSV anonimizado por ARX (sin cabecera) como lista de listas de str."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)
        return [row for row in reader]
