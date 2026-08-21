"""ytint // Intelligence Engine & Executive Dashboard (src/app.py)

Canonical entrypoint for the Streamlit dashboard.
Imports and delegates execution to `ui.app.main()`.
"""

import sys
from pathlib import Path

# Ensure src directory is in sys.path
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ui.app import main

if __name__ == "__main__":
    main()