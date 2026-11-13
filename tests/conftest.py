"""Shared test setup: puts scripts/ on sys.path, where the experiments and their library modules
live, so tests import them by module name."""
import os
import sys

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
