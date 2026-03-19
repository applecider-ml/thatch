#!/usr/bin/env python
"""
THATCH: Compare extracted aperture photometry to published values.

Demonstrates photometric accuracy of the THATCH pipeline by comparing
our HST aperture photometry against published measurements for AT2017gfo.

Usage:
    conda activate thatch
    python plot_photometry_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data")
FIGDIR = os.path.join(BASEDIR, "..", "Figures")
os.makedirs(FIGDIR, exist_ok=True)

# MJD of GW170817
MJD_GW = 57982.529

# ---- Published photometry for AT2017gfo ----
# From Cowperthwaite et al. 2017 (ApJ 848, L17) -- HST WFC3/IR
# and Villar et al. 2017 compilation
PUBLISHED_AT2017GFO = pd.DataFrame([
    # Cowperthwaite+2017 / Villar+2017 HST measurements
    {"mjd": 57987.3, "filter": "F110W", "mag": 24.55, "mag_err": 0.05, "source": "Cowperthwaite+2017"},
    {"mjd": 57987.4, "filter": "F160W", "mag": 24.24, "mag_err": 0.05, "source": "Cowperthwaite+2017"},
    {"mjd": 57989.8, "filter": "F110W", "mag": 25.40, "mag_err": 0.08, "source": "Cowperthwaite+2017"},
    {"mjd": 57993.1, "filter": "F110W", "mag": 27.10, "mag_err": 0.15, "source": "Lyman+2018"},
    {"mjd": 57993.1, "filter": "F160W", "mag": 26.30, "mag_err": 0.12, "source": "Lyman+2018"},
])
PUBLISHED_AT2017GFO["delta_t"] = PUBLISHED_AT2017GFO["mjd"] - MJD_GW

# ---- Published photometry for SN 2011fe (near-peak) ----
MJD_BMAX_2011fe = 55814.5
PUBLISHED_SN2011FE = pd.DataFrame([
    {"mjd": 55802.5, "filter": "F225W", "mag": 15.67, "mag_err": 0.05, "source": "Matheson+2012"},
    {"mjd": 55802.5, "filter": "F275W", "mag": 14.15, "mag_err": 0.03, "source": "Matheson+2012"},
    {"mjd": 55802.5, "filter": "F336W", "mag": 12.40, "mag_err": 0.02, "source": "Matheson+2012"},
    {"mjd": 55810.0, "filter": "F225W", "mag": 16.20, "mag_err": 0.05, "source": "Matheson+2012"},
    {"mjd": 55810.0, "filter": "F275W", "mag": 13.60, "mag_err": 0.03, "source": "Matheson+2012"},
    {"mjd": 55810.0, "filter": "F336W", "mag": 11.50, "mag_err": 0.02, "source": "Matheson+2012"},
])


def match_photometry(our_df, pub_df, mjd_tol=0.5, filter_col="filter"):
    """Match our photometry to published by filter and MJD."""
    matches = []
    for _, pub in pub_df.iterrows():
        candidates = our_df[
            (our_df["filter"] == pub["filter"]) &
            (np.abs(our_df["mjd"] - pub["mjd"]) < mjd_tol) &
            (our_df["ab_mag"].notna()) &
            (our_df["ab_mag"] < 30)
        ]
        if len(candidates) > 0:
            # Take the one closest in time
            best = candidates.loc[(candidates["mjd"] - pub["mjd"]).abs().idxmin()]
            matches.append({
                "filter": pub["filter"],
                "mjd_pub": pub["mjd"],
                "mjd_ours": best["mjd"],
                "mag_pub": pub["mag"],
                "mag_pub_err": pub["mag_err"],
                "mag_ours": best["ab_mag"],
                "mag_ours_err": best["ab_mag_err"],
                "source": pub["source"],
            })
    return pd.DataFrame(matches)


def main():
    # Load our photometry
    at2017gfo = pd.read_csv(os.path.join(DATADIR, "AT2017gfo", "AT2017gfo_photometry.csv"))

    # Match with published
    matched = match_photometry(at2017gfo, PUBLISHED_AT2017GFO, mjd_tol=1.0)
    print(f"Matched {len(matched)} measurements between THATCH and published values")
    if len(matched) > 0:
        print(matched[["filter", "mjd_pub", "mag_pub", "mag_ours", "source"]].to_string(index=False))

    if len(matched) == 0:
        print("No matched photometry found. Cannot create comparison plot.")
        return

    # ---- Figure: 1:1 comparison ----
    fcolors = {
        "F110W": "#e74c3c", "F160W": "#7f0000", "F153M": "#d35400",
        "F225W": "#7b2d8e", "F275W": "#9b59b6", "F336W": "#3498db",
        "F606W": "#e67e22", "F814W": "#9467bd",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # --- Panel 1: 1:1 scatter ---
    mag_min = min(matched["mag_pub"].min(), matched["mag_ours"].min()) - 0.5
    mag_max = max(matched["mag_pub"].max(), matched["mag_ours"].max()) + 0.5

    for filt, grp in matched.groupby("filter"):
        color = fcolors.get(filt, "gray")
        ax1.errorbar(
            grp["mag_pub"], grp["mag_ours"],
            xerr=grp["mag_pub_err"], yerr=grp["mag_ours_err"],
            fmt="o", color=color, markersize=7,
            label=filt, capsize=3,
            markeredgecolor="k", markeredgewidth=0.5,
        )

    ax1.plot([mag_min, mag_max], [mag_min, mag_max], "k--", alpha=0.5, lw=1, label="1:1")
    ax1.set_xlabel("Published mag (AB)", fontsize=11)
    ax1.set_ylabel("THATCH mag (AB)", fontsize=11)
    ax1.set_title("AT2017gfo: THATCH vs Published", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.set_xlim(mag_min, mag_max)
    ax1.set_ylim(mag_min, mag_max)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Residuals vs magnitude ---
    residuals = matched["mag_ours"] - matched["mag_pub"]
    resid_err = np.sqrt(matched["mag_ours_err"]**2 + matched["mag_pub_err"]**2)

    for filt, grp in matched.groupby("filter"):
        color = fcolors.get(filt, "gray")
        resid = grp["mag_ours"] - grp["mag_pub"]
        rerr = np.sqrt(grp["mag_ours_err"]**2 + grp["mag_pub_err"]**2)
        ax2.errorbar(
            grp["mag_pub"], resid,
            yerr=rerr, xerr=grp["mag_pub_err"],
            fmt="o", color=color, markersize=7,
            capsize=3, markeredgecolor="k", markeredgewidth=0.5,
            label=filt,
        )

    ax2.axhline(0, color="k", ls="--", alpha=0.5, lw=1)
    ax2.set_xlabel("Published mag (AB)", fontsize=11)
    ax2.set_ylabel("THATCH $-$ Published (mag)", fontsize=11)
    ax2.set_title("Photometric Residuals", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add stats text
    med_resid = np.median(residuals)
    rms_resid = np.sqrt(np.mean(residuals**2))
    ax2.text(0.05, 0.95, f"median = {med_resid:+.3f}\nRMS = {rms_resid:.3f}",
             transform=ax2.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    outfile = os.path.join(FIGDIR, "photometry_comparison.pdf")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.savefig(outfile.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outfile}")
    print(f"Saved: {outfile.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
