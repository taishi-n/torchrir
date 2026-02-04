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
    from torchrir import plot_scene_dynamic, plot_scene_static

    room = Room.shoebox(size=[5.0, 4.0], fs=8000)
    sources = torch.tensor([[1.0, 1.0]])
    mics = torch.tensor([[3.0, 2.0]])

    ax = plot_scene_static(room=room, sources=sources, mics=mics)
    assert ax is not None

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
