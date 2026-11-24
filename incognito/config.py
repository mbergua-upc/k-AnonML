"""
Configuration of the Incognito wrapper.

Defines the paths and constants used by anonymizer.py:
  - INCOGNITO_DIR: folder with the Java sources of Incognito (package `incognito`) and
    the classes of the UTD Anonymization Toolbox that it needs (toolbox/).
  - INCOGNITO_BUILD_DIR: folder where the sources are compiled (ignored by git).
  - SQLITE_JAR_PATH: JDBC driver of SQLite (org.xerial:sqlite-jdbc), used by the toolbox.
  - INCOGNITO_RUNNER_CLASS: command line program (IncognitoRunner.java) run in a subprocess.
  - METHODS: the Incognito versions and the toolbox method name of each one.
  - TIMEOUT: maximum running time of Incognito, in seconds.
"""
import os

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
INCOGNITO_DIR = WRAPPER_DIR
INCOGNITO_BUILD_DIR = os.path.join(INCOGNITO_DIR, 'build')
SQLITE_JAR_PATH = os.getenv('SQLITE_JDBC_JAR') or os.path.join(INCOGNITO_DIR, 'lib', 'sqlite-jdbc-3.42.0.1.jar')
INCOGNITO_RUNNER_CLASS = 'incognito.IncognitoRunner'

# Incognito versions -> method of the toolbox configuration file
METHODS = {
    'K': 'incognito_k',  # k-anonymity
    'L': 'incognito_l',  # k-anonymity and l-diversity
    'T': 'incognito_t',  # k-anonymity and t-closeness
}

TIMEOUT = 3600
