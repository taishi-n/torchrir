from __future__ import annotations

import sys
import types


def test_default_viz_style_uses_scienceplots_grid_no_latex(monkeypatch) -> None:
    from torchrir.viz import utils as viz_utils

    viz_utils._MPL_DEFAULT_STYLE_APPLIED = False

    import matplotlib.pyplot as plt

    calls: list[object] = []

    def _fake_use(style):
        calls.append(style)

    monkeypatch.setattr(plt.style, "use", _fake_use)
    monkeypatch.setitem(sys.modules, "scienceplots", types.ModuleType("scienceplots"))

    viz_utils._ensure_default_mpl_style()
    assert calls == [["science", "grid", "no-latex"]]


def test_default_viz_style_is_idempotent(monkeypatch) -> None:
    from torchrir.viz import utils as viz_utils

    viz_utils._MPL_DEFAULT_STYLE_APPLIED = False

    import matplotlib.pyplot as plt

    count = {"n": 0}

    def _fake_use(style):
        del style
        count["n"] += 1

    monkeypatch.setattr(plt.style, "use", _fake_use)
    monkeypatch.setitem(sys.modules, "scienceplots", types.ModuleType("scienceplots"))

    viz_utils._ensure_default_mpl_style()
    viz_utils._ensure_default_mpl_style()
    assert count["n"] == 1
