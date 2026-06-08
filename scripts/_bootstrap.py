"""Importing this puts <repo>/src on sys.path so `import era5tw` works when a
script is run directly (python scripts/NN_xxx.py). Every script imports it first.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
