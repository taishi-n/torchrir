from __future__ import annotations

"""CLI helpers shared by example scripts."""

from pathlib import Path

import argparse


def add_output_args(
    parser: argparse.ArgumentParser,
    *,
    out_dir_default: str | Path,
    plot_default: bool = False,
    include_plot: bool = True,
    include_show: bool = True,
    include_gif: bool = False,
) -> None:
    """Add common output/plot/GIF arguments to a parser."""
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(out_dir_default),
        help="Output directory for WAV/metadata/plots/GIFs.",
    )
    if include_plot:
        parser.add_argument(
            "--plot",
            action="store_true",
            default=plot_default,
            help="Plot room layout and trajectories." if not plot_default else "Plot outputs (PNG).",
        )
        if plot_default:
            parser.add_argument(
                "--no-plot",
                action="store_false",
                dest="plot",
                help="Disable plotting.",
            )
    if include_show:
        parser.add_argument("--show", action="store_true", help="show plots interactively")
    if include_gif:
        parser.add_argument("--gif", action="store_true", help="Save trajectory animation GIF.")
        parser.add_argument(
            "--gif-fps",
            type=int,
            default=-1,
            help="GIF frames per second (<=0 uses auto).",
        )
