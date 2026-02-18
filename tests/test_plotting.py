import pytest
import torch

from torchrir import Room


def _has_matplotlib():
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_plot_scene_static_and_dynamic():
    from torchrir.viz import plot_scene_dynamic, plot_scene_static

    room = Room.shoebox(size=[5.0, 4.0], fs=8000)
    sources = torch.tensor([[1.0, 1.0]])
    mics = torch.tensor([[3.0, 2.0]])

    ax = plot_scene_static(room=room, sources=sources, mics=mics)
    assert ax is not None
    assert len(ax.texts) >= 1

    src_traj = torch.tensor(
        [
            [[1.0, 1.0]],
            [[1.5, 1.2]],
            [[2.0, 1.4]],
        ]
    )
    mic_traj = torch.tensor(
        [
            [[3.0, 2.0]],
            [[3.0, 2.2]],
            [[3.0, 2.4]],
        ]
    )
    ax2 = plot_scene_dynamic(room=room, src_traj=src_traj, mic_traj=mic_traj)
    assert ax2 is not None
    assert len(ax2.texts) >= 1


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_plot_entity_uses_uniform_color_for_mics(monkeypatch: pytest.MonkeyPatch):
    from torchrir.viz import scene as viz_scene

    colors: list[str | None] = []

    def _fake_scatter(ax, positions, *, label, marker, color=None):
        del ax, positions, label, marker
        colors.append(color)

    monkeypatch.setattr(viz_scene, "_scatter_positions", _fake_scatter)

    traj = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ],
        dtype=torch.float32,
    )

    class _DummyAx:
        pass

    viz_scene._plot_entity(
        _DummyAx(),
        traj,
        traj[0],
        step=1,
        label="mics",
        marker="o",
        color="tab:orange",
        uniform_color=True,
    )
    assert colors
    assert set(colors) == {"tab:orange"}
