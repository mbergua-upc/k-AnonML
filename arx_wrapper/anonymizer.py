"""
Wrapper para integrar ARX (https://arx.deidentifier.org) mediante subprocess
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
    Anonimiza datos usando ARX (k-anonimidad, opcionalmente con l-diversity y t-closeness)

    Args:
        raw_data: Datos sin procesar (lista de listas de str, todas las columnas)
        header: Nombres de las columnas
        k: Valor de k-anonimidad
        gen_path: Carpeta de jerarquías de generalización (con separador final)
        dataset: Nombre del dataset
        qi_index: Índices de atributos cuasi-identificadores
        arx_method: 'SAMPLE', 'OPTIMAL' o 'BOTTOM_UP'
        suppression: Porcentaje máximo de registros suprimidos (0-100)
        timeout: Máximo tiempo en segundos (por defecto según el método)
        sensitive: Nombre del atributo sensible (obligatorio con l_diversity / t_closeness)
        l_diversity: Valor de l (None = no se aplica)
        l_variant: 'distinct' o 'entropy'
        t_closeness: Valor de t entre 0 y 1 (None = no se aplica)

    Returns:
        Datos anonimizados (lista de listas de str, mismo orden de columnas)
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
