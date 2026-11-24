"""
Configuration of the MDAV wrapper.

Defines the paths and constants used by anonymizer.py:
  - MDAV_SOURCE_DIR: folder with the Java sources of MDAV (package `mdav`).
  - MDAV_BUILD_DIR: folder where the sources are compiled (ignored by git).
  - MDAV_RUNNER_CLASS: command line program (mdav/MdavRunner.java) run in a subprocess.
  - TIMEOUT: maximum running time of MDAV, in seconds.
"""
import os

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WRAPPER_DIR)

MDAV_SOURCE_DIR = os.path.join(REPO_ROOT, 'mdav_wrapper', 'mdav')
MDAV_BUILD_DIR = os.path.join(MDAV_SOURCE_DIR, 'build')
MDAV_RUNNER_CLASS = 'mdav.MdavRunner'

# MDavExample.java is only an example (it needs a FileHelper class that is not in the repository) and is not compiled
MDAV_EXCLUDED_SOURCES = ('MDavExample.java',)

TIMEOUT = 3600
