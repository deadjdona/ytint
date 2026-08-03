import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_canonical_dashboard_avoids_deprecated_container_width_parameter():
    app_source = (PROJECT_ROOT / "src" / "app.py").read_text(encoding="utf-8")
    assert "use_container_width" not in app_source


def test_dashboard_ast_validity():
    """Verify both Streamlit app scripts parse into valid AST syntax trees."""
    app_source = (PROJECT_ROOT / "src" / "app.py").read_text(encoding="utf-8")
    ui_app_source = (PROJECT_ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8")

    tree1 = ast.parse(app_source)
    tree2 = ast.parse(ui_app_source)

    assert isinstance(tree1, ast.Module)
    assert isinstance(tree2, ast.Module)
