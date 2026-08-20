import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pytest
from ui.visual_explanations import VISUAL_METADATA

def test_visual_metadata_coverage():
    """Ensure core plots have complete structured explanations."""
    assert len(VISUAL_METADATA) >= 20
    
    required_keys = {"title", "summary", "methodology", "how_to_read", "takeaway"}
    for filename, meta in VISUAL_METADATA.items():
        assert isinstance(filename, str)
        for rk in required_keys:
            assert rk in meta, f"Missing key '{rk}' in metadata for {filename}"
            assert isinstance(meta[rk], str) and len(meta[rk]) > 0, f"Empty '{rk}' for {filename}"
