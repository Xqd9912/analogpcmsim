"""Shared figure style (Nature): one canvas width for the whole paper, one font table.

Notebooks do `from style import *`. Panel functions never write a font size or a figure width.
Geometry: double column 183 mm (W2); single column 89 mm (W1). Text 5-7 pt Arial, panel letters 8 pt bold.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

try:
    from IPython.display import Image as NotebookImage, display
except ImportError:  # pragma: no cover
    NotebookImage, display = None, (lambda *a, **k: None)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_OUT = ROOT / "figures" / "out"

MM = 1 / 25.4
W2 = 183 * MM          # canvas width of every figure (inches); never varied
W1 = 89 * MM
# Conference short paper (IEEE two-column, text width ≈ 181 mm): canvas of conf Figs. 2–4 and the page mock-up.
# Fig. 1 there is fig0_v6 (drawn at W2, placed scaled to W_CONF).
W_CONF = 178 * MM

PAPER_FONTS = {
    "title": 7,        # panel titles (used sparingly)
    "label": 7,        # axis labels
    "tick": 6,         # tick labels
    "legend": 6,
    "annot": 6,
    "panel": 8,        # panel letters (bold, lowercase)
    "cbar_label": 6,
    "cbar_tick": 5,
}


def use_paper_fonts() -> dict:
    """Set rcParams once, before any drawing; returns the font table."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.titlesize": PAPER_FONTS["title"], "axes.labelsize": PAPER_FONTS["label"],
        "xtick.labelsize": PAPER_FONTS["tick"], "ytick.labelsize": PAPER_FONTS["tick"],
        "legend.fontsize": PAPER_FONTS["legend"], "font.size": PAPER_FONTS["annot"],
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlepad": 3, "axes.labelpad": 2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.0, "lines.markersize": 3, "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2,
        "legend.handletextpad": 0.4, "legend.labelspacing": 0.3,
        "figure.dpi": 150, "savefig.dpi": 600,
    })
    return dict(PAPER_FONTS)


def mark(fig, ax, letter: str, dx_in: float = -0.36, dy_in: float = 0.04) -> None:
    """Panel letter at the top-left of `ax` (composite only, never inside draw_x)."""
    W, H = fig.get_size_inches()
    b = ax.get_position()
    fig.text(b.x0 + dx_in / W, b.y1 + dy_in / H, letter, fontsize=PAPER_FONTS["panel"], fontweight="bold",
             ha="left", va="bottom")


def save_matched_panels(fig, groups: dict, outdir: Path, dpi: int = 600) -> dict:
    """Crop each panel group (axes + texts) out of the composite canvas; same scale as the figure."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = {}
    for letter, g in groups.items():
        bbs = [a.get_tightbbox(r) for a in g["axes"]] + [t.get_window_extent(r) for t in g.get("texts", [])]
        bb = Bbox.union([b for b in bbs if b is not None]).transformed(fig.dpi_scale_trans.inverted()).padded(0.03)
        fig.savefig(outdir / f"{letter}.png", dpi=dpi, bbox_inches=bb)
        boxes[letter] = bb
    return boxes


def export_composite(fig, outdir: Path, name: str = "composite") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=600)


def example_tag(ax, color: str = "#D62728", text: str = "EXAMPLE DATA", loc: str = "upper right") -> None:
    """Mark a panel whose data are placeholders (expected values), so it never goes out by mistake."""
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.98, "top") if "upper" in loc else (0.02, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=PAPER_FONTS["annot"] - 1, color=color,
            fontweight="bold", zorder=10)


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color="#E5E5E5", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
