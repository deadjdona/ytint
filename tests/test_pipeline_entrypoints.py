import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def test_s04_main_guard_calls_compile_ui_metrics_without_arguments():
    tree = ast.parse((PROJECT_ROOT / "src" / "pipeline" / "s04_synthesis.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compile_ui_metrics"
    ]

    assert len(calls) == 1
    assert calls[0].args == []
    assert calls[0].keywords == []
