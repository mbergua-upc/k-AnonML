"""
Configuration of the ARX wrapper.

Defines the paths and constants used by anonymizer.py:
  - ARX_JAR_PATH: ARX jar, taken from the ARX_JAR environment variable or, if it
    is not set, the first arx*.jar found in the repository root.
  - ARX_RUNNER_PATH: Java program (java/ArxRunner.java) that calls the ARX API.
  - METHOD_TO_ALGORITHM: maps the wrapper methods (SAMPLE, OPTIMAL, BOTTOM_UP) to
    ARX algorithms (BEST_EFFORT_GENETIC, OPTIMAL, BEST_EFFORT_BOTTOM_UP).
  - AVAILABLE_METHODS: the supported method names.
  - TIMEOUTS: maximum running time, in seconds, of each method.
"""
import os

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WRAPPER_DIR)

# JAR file of ARX (https://arx.deidentifier.org/). Can be set with the ARX_JAR environment variable, otherwise the first arx*.jar found in the repository root is used.
ARX_JAR_PATH = os.getenv('ARX_JAR') or next(
    (os.path.join(REPO_ROOT, f) for f in sorted(os.listdir(REPO_ROOT))
     if f.startswith('arx') and f.endswith('.jar')),
    os.path.join(REPO_ROOT, 'arx.jar'))

# Java program that calls the ARX API (java/ArxRunner.java). It is run in a subprocess by anonymizer.py.
ARX_RUNNER_PATH = os.path.join(WRAPPER_DIR, 'java', 'ArxRunner.java')

# Wrapper methods -> ARX algorithms (ARXConfiguration.AnonymizationAlgorithm)
METHOD_TO_ALGORITHM = {
    'SAMPLE': 'BEST_EFFORT_GENETIC',
    'OPTIMAL': 'OPTIMAL',
    'BOTTOM_UP': 'BEST_EFFORT_BOTTOM_UP',
}
AVAILABLE_METHODS = list(METHOD_TO_ALGORITHM)

# Timeouts per method, in seconds. If None, no timeout is applied. The default values are based on the ARX documentation and empirical tests.
TIMEOUTS = {
    'SAMPLE': 600,
    'OPTIMAL': 3600,
    'BOTTOM_UP': 1800,
}
