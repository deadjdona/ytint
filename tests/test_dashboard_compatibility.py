from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def test_canonical_dashboard_avoids_deprecated_container_width_parameter():
    app_source = (PROJECT_ROOT / "src" / "app.py").read_text(encoding="utf-8")

    assert "use_container_width" not in app_source
