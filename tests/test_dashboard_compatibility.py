import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_canonical_dashboard_avoids_deprecated_container_width_parameter():
    app_source = (SRC_DIR / "app.py").read_text(encoding="utf-8")
    ui_app_source = (SRC_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert "use_container_width" not in app_source
    assert "use_container_width" not in ui_app_source


def test_dashboard_ast_validity():
    """Verify both Streamlit app scripts parse into valid AST syntax trees."""
    app_source = (SRC_DIR / "app.py").read_text(encoding="utf-8")
    ui_app_source = (SRC_DIR / "ui" / "app.py").read_text(encoding="utf-8")

    tree1 = ast.parse(app_source)
    tree2 = ast.parse(ui_app_source)

    assert isinstance(tree1, ast.Module)
    assert isinstance(tree2, ast.Module)


def test_app_is_thin_entrypoint():
    """Verify src/app.py is a lightweight entrypoint delegating to ui.app.main."""
    app_source = (SRC_DIR / "app.py").read_text(encoding="utf-8")
    lines = [l for l in app_source.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert len(lines) < 30, f"src/app.py is unexpectedly large ({len(lines)} non-empty lines)"
    assert "from ui.app import main" in app_source or "import ui.app" in app_source


def test_ui_app_exports_callable_main():
    """Verify ui.app exports a callable main function."""
    import ui.app

    assert hasattr(ui.app, "main")
    assert callable(ui.app.main)
