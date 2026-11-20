# -*- coding: utf-8 -*-
"""Nature-style figure formatting, applied on import: sans-serif 7 pt type, thin axes and lines, muted colours, bold
lower-case panel labels at the top left, figure width 183 mm (double column). Every Figure is finalized once before it is
saved: it is rescaled to 7.2 in width (keeping the aspect ratio), line widths and marker sizes are scaled with it, and a
title of the form '(a) text', 'a, text', 'a  text' or 'a' becomes a bold panel label plus a small left-aligned heading."""
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from matplotlib.patches import Patch
from matplotlib.text import Text
WIDTH = 7.2
PALETTE = ["#2e6f9e", "#c4452c", "#3a8f6b", "#8a6bbf", "#d08a2e", "#5f5f5f", "#4fa3c7", "#b8476f", "#8c8c3a", "#1d3b5c"]
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "legend.frameon": False, "legend.handlelength": 1.6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out",
    "ytick.direction": "out", "axes.prop_cycle": mpl.cycler(color=PALETTE), "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "errorbar.capsize": 1.2, "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "figure.dpi": 100, "axes.titlepad": 3.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
_PAT = [re.compile(r"^\(([a-z])\)\s*(.*)$", re.S), re.compile(r"^([a-z]),\s*(.*)$", re.S), re.compile(r"^([a-z])\s{1,}(.*)$", re.S),
        re.compile(r"^([a-z])$")]
def _label(ax):
    for loc in ("left", "center", "right"):
        t = ax.get_title(loc=loc)
        if not t: continue
        for p in _PAT:
            m = p.match(t.strip())
            if m:
                letter = m.group(1); rest = (m.group(2) if m.lastindex and m.lastindex > 1 else "").strip()
                ax.set_title("", loc=loc)
                ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-24, 5), textcoords="offset points",
                            fontsize=9, fontweight="bold", ha="left", va="bottom")
                if rest:
                    ax.set_title(rest, loc="left", fontsize=6.5, fontweight="normal")
                return
        ax.set_title(t, loc=loc, fontsize=6.5, fontweight="normal")
def finalize(fig):
    if getattr(fig, "_nstyle_done", False): return
    fig._nstyle_done = True
    w, h = fig.get_size_inches(); s = WIDTH / w if w > WIDTH else 1.0
    fig.set_size_inches(w * s, h * s)
    for ax in fig.axes:
        _label(ax)
        for art in ax.get_children():
            if isinstance(art, Line2D):
                art.set_linewidth(max(art.get_linewidth() * s, 0.5)); art.set_markersize(art.get_markersize() * s)
                art.set_markeredgewidth(max(art.get_markeredgewidth() * s, 0.5))
            elif isinstance(art, Collection):
                lw = art.get_linewidths()
                if len(lw): art.set_linewidths([max(x * s, 0.4) for x in lw])
                try:
                    sz = art.get_sizes()
                    if len(sz): art.set_sizes(sz * s * s)
                except AttributeError:
                    pass
            elif isinstance(art, Patch) and art.get_linewidth() > 0:
                art.set_linewidth(max(art.get_linewidth() * s, 0.4))
        for t in ax.texts:
            if t.get_fontsize() > 7.5 and t.get_fontweight() != "bold": t.set_fontsize(7)
        lg = ax.get_legend()
        if lg:
            lg.set_frame_on(False)
            for t in lg.get_texts(): t.set_fontsize(6.2)
            for art in lg.findobj(Line2D):
                art.set_linewidth(max(art.get_linewidth() * s, 0.5)); art.set_markersize(art.get_markersize() * s)
            for art in lg.findobj(Collection):
                lw = art.get_linewidths()
                if len(lw): art.set_linewidths([max(x * s, 0.4) for x in lw])
        for sp in ax.spines.values(): sp.set_linewidth(0.6)
        ax.tick_params(width=0.6, length=2.5, labelsize=6.5)
        ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout(pad=0.4, w_pad=1.6, h_pad=1.2, rect=(0.01, 0.0, 1.0, 0.985))
        except Exception as e:
            print("nstyle: tight_layout skipped:", e)
_orig_save = Figure.savefig
def _savefig(self, *a, **k):
    finalize(self); return _orig_save(self, *a, **k)
Figure.savefig = _savefig
