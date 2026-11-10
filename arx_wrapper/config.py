"""
Configuración para ARX wrapper
"""
import os

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WRAPPER_DIR)

# JAR de ARX (variable de entorno ARX_JAR o primer arx*.jar de la raíz del repo)
ARX_JAR_PATH = os.getenv('ARX_JAR') or next(
    (os.path.join(REPO_ROOT, f) for f in sorted(os.listdir(REPO_ROOT))
     if f.startswith('arx') and f.endswith('.jar')),
    os.path.join(REPO_ROOT, 'arx.jar'))

# Programa Java que usa la API de ARX (se ejecuta en modo "source file", requiere JDK 11+)
ARX_RUNNER_PATH = os.path.join(WRAPPER_DIR, 'java', 'ArxRunner.java')

# Método del wrapper -> algoritmo de ARX (ARXConfiguration.AnonymizationAlgorithm)
METHOD_TO_ALGORITHM = {
    'SAMPLE': 'BEST_EFFORT_GENETIC',
    'OPTIMAL': 'OPTIMAL',
    'BOTTOM_UP': 'BEST_EFFORT_BOTTOM_UP',
}
AVAILABLE_METHODS = list(METHOD_TO_ALGORITHM)

# Timeouts por método (segundos)
TIMEOUTS = {
    'SAMPLE': 600,
    'OPTIMAL': 3600,
    'BOTTOM_UP': 1800,
}
