import sys
from pathlib import Path

# Ensure src directory is on sys.path for all test execution
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
