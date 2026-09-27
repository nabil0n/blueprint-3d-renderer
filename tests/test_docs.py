"""The README figures script uses parser internals; keep it runnable through refactors."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "make_figures.py"


def test_readme_figures_still_render(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", tmp_path)
    module.main()
    assert len(list(tmp_path.glob("*.png"))) >= 6
