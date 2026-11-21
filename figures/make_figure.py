#!/usr/bin/env python
"""Letter-preference figure (main: prefix none; supplement: prefix eos).
Usage: python make_figure.py match_metrics.csv bootstrap_intervals.csv data_letter_shares.csv [outdir]
All plotted values are read from the three CSVs."""
import sys, os
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

LET = ["A", "B", "C", "D"]
COL = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73", "D": "#CC79A7"}  # Okabe-Ito subset
S2_TOK_PER_STEP = 2 ** 21


def load(m_path, b_path, d_path, prefix):
    m = pd.read_csv(m_path); b = pd.read_csv(b_path); d = pd.read_csv(d_path)
    m = m[(m.label_set == "letters") & (m.condition == "shuffled") & (m.prefix == prefix)].copy()
    b = b[(b.label_set == "letters") & (b.condition == "shuffled") & (b.prefix == prefix)]
    d = d[d.label_set == "letters"]
    for i, l in enumerate(LET):
        bb = b[b.label == l].set_index("revision")
        m[l] = m[f"share_{i}"]
        m[l + "_lo"] = m.revision.map(bb.lo95)
        m[l + "_hi"] = m.revision.map(bb.hi95)
    return m, d


def share_row(d, stage, definition):
    r = d[(d.stage == stage) & (d.definition == definition)].iloc[0]
    return np.array([r[f"s{i}"] for i in range(4)])


def draw(m, d, prefix, out_base):
    s1 = m[(m.stage == 1)].sort_values("step")
    s2 = m[(m.stage == 2) & (m.ingredient == 3) & (m.is_main == 0) & (m.replicate_endpoint == 0)].sort_values("step")
    ep = m[(m.stage == 2) & (m.replicate_endpoint == 1)].sort_values("ingredient")
    assert len(s1) == 26 and len(s2) == 24 and len(ep) == 2, (len(s1), len(s2), len(ep))
    x1 = s1.tokens_seen_x.values
    x2 = s2.step.values * S2_TOK_PER_STEP / 1e9

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row",
                            gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.05, "hspace": 0.3})
    (a1, a2), (b1, b2) = axs

    # ---- top row: shares
    for ax, df, x in ((a1, s1, x1), (a2, s2, x2)):
        ax.axhline(0.25, color="0.4", ls="--", lw=0.8, zorder=1)
        for l in LET:
            ax.fill_between(x, df[l + "_lo"], df[l + "_hi"], color=COL[l], alpha=0.30, lw=0, zorder=2)
            ax.plot(x, df[l], color=COL[l], lw=1.2, marker="o", ms=2.2, zorder=3)
    # stage-2 xlim leaves right margin for end points and data-share markers
    x_end = x2.max()
    xe = {1.0: x_end + 3.0, 2.0: x_end + 7.0}
    xd = {"U_sameline": x_end + 10.0, "U_T14": x_end + 18.5}
    a2.set_xlim(0, x_end + 27.5)
    a1.set_xscale("log"); b1.set_xscale("log"); a1.set_xlim(x1.min() / 1.4, 2.1e13)

    # data share markers: one short horizontal segment per letter, staggered left (A) to right (D) so equal values stay visible
    def put(ax, vals, lo, W, geometric, style):
        for i, l in enumerate(LET):
            f0, f1 = 0.13 * i, 0.13 * i + 0.58
            if geometric:
                x0, x1_ = lo * W ** f0, lo * W ** f1
            else:
                x0, x1_ = lo + W * f0, lo + W * f1
            if style == "solid":
                ax.hlines(vals[i], x0, x1_, color=COL[l], lw=2.6, zorder=4 + i * 0.1)
            else:
                h = 0.009
                ax.add_patch(mpl.patches.Rectangle((x0, vals[i] - h), x1_ - x0, 2 * h, fill=False, ec=COL[l], lw=1.1, zorder=4))
    put(a1, share_row(d, 1, "T1-T4"), 5.5e12, 3.2, True, "solid")
    put(a2, share_row(d, 2, "U_sameline"), xd["U_sameline"], 7.0, False, "solid")
    put(a2, share_row(d, 2, "U_T14"), xd["U_T14"], 7.0, False, "open")
    # replicate end points (open markers)
    for _, r in ep.iterrows():
        mk = "o" if r.ingredient == 1 else "s"
        for l in LET:
            a2.plot(xe[r.ingredient], r[l], marker=mk, mfc="white", mec=COL[l], mew=1.2, ms=5, ls="none", zorder=5)
            a2.plot(xe[r.ingredient], r[l], marker=mk, mfc="white", mec=COL[l], mew=1.2, ms=5, ls="none", zorder=5)

    # ---- bottom row: accuracy
    for ax, df, x in ((b1, s1, x1), (b2, s2, x2)):
        ax.axhline(0.25, color="0.4", ls="--", lw=0.8, zorder=1)
        ax.plot(x, df.accuracy, color="0.15", lw=1.2, marker="o", ms=2.2, zorder=3)
    for _, r in ep.iterrows():
        mk = "o" if r.ingredient == 1 else "s"
        b2.plot(xe[r.ingredient], r.accuracy, marker=mk, mfc="white", mec="0.15", mew=1.2, ms=5, ls="none", zorder=5)
    b2.set_xlim(a2.get_xlim()); b1.set_xlim(x1.min() / 1.4, 2.1e13)

    # ---- axes cosmetics
    a1.set_ylim(0, 0.70)
    b1.set_ylim(0.22, 0.48)
    a1.set_title("Stage 1 (olmo-mix-1124)", loc="left")
    a2.set_title("Stage 2 (dolmino-mix-1124, ingredient 3)", loc="left")
    a1.set_ylabel("Share of answers\n(letter A\u2013D)")
    b1.set_ylabel("Accuracy")
    fmt = lambda v, p: ("%gT" % (v / 1e12)) if v >= 1e12 else ("%gB" % (v / 1e9))
    b1.set_xticks([1e9, 1e10, 1e11, 1e12]); a1.set_xticks([1e9, 1e10, 1e11, 1e12])
    b1.xaxis.set_major_formatter(FuncFormatter(fmt))
    a1.xaxis.set_minor_locator(mpl.ticker.NullLocator()); b1.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    b1.set_xlabel("Tokens seen (log scale)")
    b2.set_xlabel("Stage-2 tokens seen (billions)")
    ticks = [0, 10, 20, 30, 40, 50]
    for ax in (a2, b2):
        ax.set_xticks(ticks)
    a2.set_xticklabels([]); a1.tick_params(labelbottom=False)
    for ax in (a2, b2):
        ax.tick_params(labelleft=False)
        ax.spines["left"].set_visible(True)
    # x labels for data/end-point markers (above the markers; below axis of top panel)
    for xx, txt in ((xe[1.0], "ing. 1"), (xe[2.0], "ing. 2")):
        pass

    # ---- legend (figure-level, below the panels)
    H = [Line2D([], [], color=COL[l], lw=1.5, marker="o", ms=3, label=f"Letter {l}") for l in LET]
    H += [Patch(facecolor="0.5", alpha=0.3, label="95% question-bootstrap interval"),
          Line2D([], [], color="0.15", lw=1.2, marker="o", ms=3, label="Accuracy"),
          Line2D([], [], color="0.4", ls="--", lw=0.8, label="0.25 (uniform / chance)"),
          Line2D([], [], color="0.4", lw=2.6, label="Data share, stage 1 (T1\u2013T4)"),
          Line2D([], [], color="0.4", lw=2.6, label="Data share, stage 2 same-line (left)"),
          Patch(facecolor="white", edgecolor="0.4", lw=1.0, label="Data share, stage 2 matched to T1\u2013T4 (right)"),
          Line2D([], [], marker="o", mfc="white", mec="0.4", mew=1.2, ls="none", ms=5, label="Ingredient 1 end point"),
          Line2D([], [], marker="s", mfc="white", mec="0.4", mew=1.2, ls="none", ms=5, label="Ingredient 2 end point")]
    fig.legend(handles=H, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7,
               handlelength=1.8, columnspacing=1.2)
    for ax, t in ((a1, "a"), (a2, "b"), (b1, "c"), (b2, "d")):
        ax.text(-0.02 if ax in (a1, b1) else -0.02, 1.12 if ax in (a1, a2) else 1.06, t, transform=ax.transAxes,
                fontweight="bold", fontsize=10, va="bottom", ha="right")
    a1.annotate("data share\n(stage 1)", xy=(xs1, 0.0), xycoords=("data", "axes fraction"), xytext=(xs1, -0.0),
                textcoords=("data", "axes fraction"), fontsize=6, ha="center", va="bottom", annotation_clip=False) if False else None
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    return fig, (s1, s2, ep)


if __name__ == "__main__":
    m_path, b_path, d_path = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else "."
    for prefix, name in (("none", "fig_letter_preference"), ("eos", "fig_letter_preference_eos")):
        m, d = load(m_path, b_path, d_path, prefix)
        fig, _ = draw(m, d, prefix, os.path.join(out, name))
        plt.close(fig)
