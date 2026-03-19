#!/usr/bin/env python
"""
plot_all_lightcurves.py
Generate multi-band light curve plots for all THATCH targets.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ── paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
FIG_DIR = os.path.join(BASE, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── object metadata ──
OBJECTS = {
    "AT2017gfo": {"mjd_ref": 57982.529, "xlabel": "Days since merger"},
    "SN2011fe": {"mjd_ref": 55814.5, "xlabel": "Days from B-max"},
    "SN2014J": {"mjd_ref": 56681.0, "xlabel": "Days from B-max"},
    "SN_Refsdal": {"mjd_ref": 57000.0, "xlabel": "Days from discovery"},
    "SN1987A": {"mjd_ref": 46849.0, "xlabel": "Days from explosion"},
    "Sw_J1644+57": {"mjd_ref": 55648.0, "xlabel": "Days from trigger"},
}

# ── filter colour map (HST filters) ──
FILTER_COLORS = {
    "F225W": "#8b00ff",
    "F275W": "#6a0dad",
    "F336W": "#4400cc",
    "F438W": "#0044cc",
    "F475W": "#0077ff",
    "F555W": "#00aa44",
    "F606W": "#55aa00",
    "F625W": "#88aa00",
    "F775W": "#cc6600",
    "F814W": "#cc3300",
    "F850LP": "#aa0000",
    "F098M": "#990044",
    "F105W": "#993300",
    "F110W": "#aa4400",
    "F125W": "#bb5500",
    "F140W": "#cc6600",
    "F153M": "#cc7700",
    "F160W": "#dd0000",
    "F435W": "#0055dd",
    "F658N": "#bb4400",
    "F502N": "#00bb66",
}
_extra_colors = plt.cm.tab20(np.linspace(0, 1, 20))


def _color_for(filt, idx):
    if filt in FILTER_COLORS:
        return FILTER_COLORS[filt]
    return _extra_colors[idx % len(_extra_colors)]


def load_and_filter(csv_path, mjd_ref):
    """Load CSV and return cleaned dataframe with phase column."""
    df = pd.read_csv(csv_path)
    # keep valid detections
    mask = (
        df["ab_mag"].notna()
        & (df["ab_mag"] < 30)
        & (df["ab_mag_err"].notna())
        & (df["ab_mag_err"] < 2)
    )
    df = df[mask].copy()
    df["phase"] = df["mjd"] - mjd_ref
    return df


def plot_single(obj_name, meta, ax=None, standalone=True):
    """Plot light curve for one object. If ax is given, plot into it."""
    csv_path = os.path.join(DATA_DIR, obj_name, f"{obj_name}_photometry.csv")
    df = load_and_filter(csv_path, meta["mjd_ref"])

    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    filters_sorted = sorted(df["filter"].unique())
    for i, filt in enumerate(filters_sorted):
        sub = df[df["filter"] == filt].sort_values("phase")
        ax.errorbar(
            sub["phase"],
            sub["ab_mag"],
            yerr=sub["ab_mag_err"],
            fmt="o",
            ms=4,
            capsize=2,
            label=filt,
            color=_color_for(filt, i),
            alpha=0.85,
            lw=0.8,
        )

    ax.invert_yaxis()
    ax.set_xlabel(meta["xlabel"], fontsize=10)
    ax.set_ylabel("AB mag", fontsize=10)
    ax.set_title(f"THATCH — {obj_name}", fontsize=11, fontweight="bold")
    ax.legend(
        fontsize=6,
        ncol=max(1, len(filters_sorted) // 4),
        loc="best",
        framealpha=0.7,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    ax.tick_params(labelsize=8)

    if standalone:
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(
                os.path.join(FIG_DIR, f"{obj_name}_lightcurve_all.{ext}"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close(fig)
        print(f"  Saved {obj_name}_lightcurve_all.pdf/png")


def main():
    # ── individual plots ──
    print("Generating individual light-curve plots …")
    for obj_name, meta in OBJECTS.items():
        plot_single(obj_name, meta, standalone=True)

    # ── summary 2×3 grid ──
    print("Generating summary grid …")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (obj_name, meta) in zip(axes.flat, OBJECTS.items()):
        plot_single(obj_name, meta, ax=ax, standalone=False)

    fig.suptitle("THATCH — Multi-band HST Light Curves", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(FIG_DIR, f"THATCH_lightcurves_summary.{ext}"), dpi=200, bbox_inches="tight"
        )
    plt.close(fig)
    print("  Saved THATCH_lightcurves_summary.pdf/png")
    print("Done.")


if __name__ == "__main__":
    main()
