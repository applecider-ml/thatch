#!/usr/bin/env python
"""
THATCH Demo: End-to-end HST light curve extraction for SN 2011fe.

Best-studied nearby Type Ia supernova (in M101).
292 HST observations, 24 programs, 10 STIS spectroscopic epochs,
photometry spanning 11.5 years.

Usage:
    conda activate thatch
    python demo_sn2011fe.py
"""

import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astroquery.mast import Observations

warnings.filterwarnings("ignore")

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data", "SN2011fe")
FIGDIR = os.path.join(BASEDIR, "..", "Figures")
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# SN 2011fe coordinates and discovery info
RA = 210.77420
DEC = 54.27370
COORD = SkyCoord(ra=RA, dec=DEC, unit="deg")
# Discovery: 2011 Aug 24 (PTF11kly); B-max ~2011 Sep 10
MJD_BMAX = 55814.5  # Approximate B-band maximum (2011 Sep 10)
MJD_DISCOVERY = 55796.7  # 2011 Aug 24

# Aperture corrections: encircled energy fraction within r=4 pixels.
# From WFC3 Instrument Handbook Table 7.7 (UVIS, 0.04"/pix, r=0.16")
# and Table 7.8 (IR, 0.13"/pix, r=0.52"), and ACS Instrument Handbook.
APERTURE_CORRECTIONS_4PX = {
    # WFC3/UVIS (r=4px = 0.16"): EE ~ 0.74-0.80
    "F225W": 0.740, "F275W": 0.755, "F336W": 0.775,
    "F438W": 0.795, "F467M": 0.800, "F469N": 0.800,
    "F475W": 0.800, "F475X": 0.800, "F547M": 0.810,
    "F555W": 0.810, "F600LP": 0.810, "F606W": 0.810,
    "F625W": 0.810, "F775W": 0.800, "F814W": 0.795,
    # WFC3/IR (r=4px = 0.52"): EE ~ 0.90-0.93
    "F105W": 0.915, "F110W": 0.920, "F125W": 0.920,
    "F140W": 0.920, "F153M": 0.915, "F160W": 0.910,
    # ACS/WFC (r=4px = 0.20"): EE ~ 0.78-0.82
    "F435W": 0.785, "F475W": 0.795, "F555W": 0.805,
    "F606W": 0.810, "F625W": 0.810, "F775W": 0.800,
    "F814W": 0.795, "F850LP": 0.790,
    # WFPC2/PC (r=4px = 0.18"): EE ~ 0.75-0.80
    "F439W": 0.770, "F555W": 0.790, "F606W": 0.795,
    "F814W": 0.780, "F1042M": 0.760,
}


def query_and_filter():
    """Query MAST for SN2011fe HST imaging."""
    print("=" * 60)
    print("Step 1: Querying MAST for SN 2011fe")
    print("=" * 60)

    obs_table = Observations.query_region(COORD, radius=3.0 * u.arcsec)

    hst_mask = obs_table["obs_collection"] == "HST"
    hst_obs = obs_table[hst_mask]

    # Separate imaging from spectroscopy
    imaging = []
    spectroscopy = []
    for row in hst_obs:
        dtype = str(row["dataproduct_type"]).lower()
        filt = str(row["filters"])
        is_grism = any(g in filt for g in [
            "G102", "G141", "G130M", "G140L", "G230L", "G230LB",
            "G430L", "G750L", "MIRVIS", "MIRFUV"
        ])
        if "image" in dtype and not is_grism:
            imaging.append(True)
            spectroscopy.append(False)
        else:
            imaging.append(False)
            spectroscopy.append("spectrum" in dtype or is_grism)

    img_mask = np.array(imaging)
    img_obs = hst_obs[img_mask]

    print(f"  Total HST obs: {len(hst_obs)}")
    print(f"  Imaging: {len(img_obs)}")
    print(f"  Spectroscopy: {sum(spectroscopy)}")

    # Summarize filters
    filters = {}
    for row in img_obs:
        f = str(row["filters"])
        filters[f] = filters.get(f, 0) + 1
    print("  Imaging filters:")
    for f, n in sorted(filters.items()):
        print(f"    {f}: {n}")

    # Summarize programs
    programs = set()
    for row in img_obs:
        try:
            programs.add(str(row["proposal_id"]))
        except Exception:
            pass
    print(f"  Programs: {sorted(programs)}")

    return img_obs


def download_products(obs_table, max_products=80):
    """Download drizzled images, prioritizing near-peak epochs."""
    print("\n" + "=" * 60)
    print("Step 2: Downloading calibrated images")
    print("=" * 60)

    # Sort observations by proximity to B-max to prioritize peak coverage
    obs_copy = obs_table.copy()
    t_mid = np.array([
        float(row["t_min"]) if row["t_min"] and str(row["t_min"]) != "--" else np.nan
        for row in obs_copy
    ])
    dt_from_peak = np.abs(t_mid - MJD_BMAX)
    # Replace NaN with large value so they sort to the end
    dt_from_peak[np.isnan(dt_from_peak)] = 1e6
    sort_idx = np.argsort(dt_from_peak)
    obs_sorted = obs_copy[sort_idx]

    # Take the observations closest to peak first
    if len(obs_sorted) > max_products:
        obs_sorted = obs_sorted[:max_products]
        print(f"  Selected {max_products} observations closest to B-max")

    products = Observations.get_product_list(obs_sorted)

    # Filter to drizzled images
    drz_mask = np.array([
        any(ext in str(pn) for ext in ["_drz.fits", "_drc.fits"])
        for pn in products["productFilename"]
    ])
    drz_products = products[drz_mask]
    print(f"  Found {len(drz_products)} drizzled products")

    manifest = Observations.download_products(
        drz_products,
        download_dir=DATADIR,
        flat=True,
    )
    print(f"  Downloaded {len(manifest)} files")
    return manifest


def extract_photometry():
    """Aperture photometry on all downloaded images."""
    print("\n" + "=" * 60)
    print("Step 3: Extracting aperture photometry")
    print("=" * 60)

    fits_files = glob(os.path.join(DATADIR, "**", "*.fits"), recursive=True)
    fits_files = [f for f in fits_files if any(
        ext in f for ext in ["_drz.fits", "_drc.fits"]
    )]
    print(f"  Processing {len(fits_files)} FITS files")

    records = []
    for fpath in sorted(fits_files):
        fname = os.path.basename(fpath)
        try:
            result = measure_one_image(fpath, fname)
            if result:
                records.append(result)
                dt = result["delta_t_days"]
                filt = result["filter"]
                mag = result["ab_mag"]
                err = result["ab_mag_err"]
                if np.isfinite(mag):
                    print(f"  {fname}: {filt:8s} dt={dt:+8.1f}d  mag={mag:.2f}+/-{err:.2f}")
        except Exception as e:
            pass

    if not records:
        print("  No photometry extracted!")
        return None

    df = pd.DataFrame(records)
    df = df.sort_values("mjd")
    outfile = os.path.join(DATADIR, "SN2011fe_photometry.csv")
    df.to_csv(outfile, index=False)
    print(f"\n  Saved {len(df)} measurements to {outfile}")
    return df


def measure_one_image(fpath, fname):
    """Extract aperture photometry from one FITS image."""
    with fits.open(fpath) as hdul:
        # Find SCI extension
        sci_ext = None
        for i, hdu in enumerate(hdul):
            if hdu.name == "SCI":
                sci_ext = i
                break
        if sci_ext is None and hdul[0].data is not None and hdul[0].data.ndim == 2:
            sci_ext = 0
        if sci_ext is None:
            return None

        header = hdul[sci_ext].header
        data = hdul[sci_ext].data.astype(float)
        pri = hdul[0].header

        # Get filter
        filt = get_filter(header, pri)

        # Get instrument
        instrument = f"{pri.get('INSTRUME', '?')}/{pri.get('DETECTOR', '?')}"

        # Get time
        date_obs = pri.get("DATE-OBS", header.get("DATE-OBS"))
        time_obs = pri.get("TIME-OBS", header.get("TIME-OBS", "00:00:00"))
        if not date_obs:
            return None
        try:
            t = Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")
            mjd = t.mjd
        except Exception:
            return None

        exptime = float(header.get("EXPTIME", pri.get("EXPTIME", 1.0)))

        # WCS -> pixel position
        try:
            wcs = WCS(header, naxis=2)
            x_pix, y_pix = wcs.world_to_pixel(COORD)
            x_pix, y_pix = float(x_pix), float(y_pix)
        except Exception:
            return None

        ny, nx = data.shape
        if not (8 < x_pix < nx - 8 and 8 < y_pix < ny - 8):
            return None

        # Photometric calibration
        photflam = float(header.get("PHOTFLAM", pri.get("PHOTFLAM", 0)))
        photplam = float(header.get("PHOTPLAM", pri.get("PHOTPLAM", 0)))
        photzpt = float(header.get("PHOTZPT", pri.get("PHOTZPT", -21.1)))

        # Aperture photometry
        # For SN in galaxy: use small aperture to reduce host contamination
        ap_r = 4  # pixels
        bg_in, bg_out = 12, 18

        yy, xx = np.mgrid[:ny, :nx]
        dist = np.sqrt((xx - x_pix) ** 2 + (yy - y_pix) ** 2)

        ap_mask = dist <= ap_r
        bg_mask = (dist >= bg_in) & (dist <= bg_out)

        valid_ap = ap_mask & np.isfinite(data)
        valid_bg = bg_mask & np.isfinite(data)

        if np.sum(valid_ap) < 5 or np.sum(valid_bg) < 20:
            return None

        bg_pixels = data[valid_bg]
        _, bg_med, bg_std = sigma_clipped_stats(bg_pixels, sigma=3.0)

        src_flux = np.sum(data[valid_ap]) - bg_med * np.sum(valid_ap)
        n_ap = float(np.sum(valid_ap))
        src_err = np.sqrt(n_ap * bg_std ** 2 + np.abs(src_flux))

        # Count rate (drz files are already e-/s)
        # Check if data is in counts or count rate
        bunit = str(header.get("BUNIT", pri.get("BUNIT", ""))).strip().upper()
        if "SECOND" in bunit or "/S" in bunit:
            rate = src_flux
            rate_err = src_err
        elif exptime > 0:
            rate = src_flux / exptime
            rate_err = src_err / exptime
        else:
            rate = src_flux
            rate_err = src_err

        # Apply aperture correction (divide by encircled energy fraction)
        ee_frac = APERTURE_CORRECTIONS_4PX.get(filt, 0.85)
        rate /= ee_frac
        rate_err /= ee_frac

        # AB magnitude
        abmag, magerr = compute_abmag(rate, rate_err, photflam, photplam, photzpt)

        return {
            "filename": fname,
            "mjd": mjd,
            "delta_t_days": mjd - MJD_BMAX,
            "filter": filt,
            "instrument": instrument,
            "exptime_s": exptime,
            "count_rate": rate,
            "count_rate_err": rate_err,
            "ab_mag": abmag,
            "ab_mag_err": magerr,
            "x_pix": x_pix,
            "y_pix": y_pix,
            "photflam": photflam,
            "photplam": photplam,
            "bg_median": bg_med,
            "bg_std": bg_std,
            "proposal_id": str(pri.get("PROPOSID", "?")),
        }


def get_filter(header, pri):
    """Extract the actual filter name from HST headers."""
    # Try various header keywords
    for hdr in [header, pri]:
        for key in ["FILTER", "FILTER1", "FILTER2"]:
            val = hdr.get(key, "")
            if val and "CLEAR" not in str(val).upper() and str(val) not in ["N/A", ""]:
                return str(val)
    # Fallback: try FILTER1 then FILTER2
    f1 = str(header.get("FILTER1", pri.get("FILTER1", "")))
    f2 = str(header.get("FILTER2", pri.get("FILTER2", "")))
    if f2 and "CLEAR" not in f2.upper():
        return f2
    if f1 and "CLEAR" not in f1.upper():
        return f1
    return "UNKNOWN"


def compute_abmag(rate, rate_err, photflam, photplam, photzpt):
    """Convert count rate to AB magnitude."""
    if rate <= 0:
        return np.nan, np.nan

    if photflam > 0:
        flux_cgs = rate * photflam
        if flux_cgs <= 0:
            return np.nan, np.nan
        stmag = -2.5 * np.log10(flux_cgs) + photzpt
        if photplam > 0:
            abmag = stmag - 5.0 * np.log10(photplam) + 18.6921
        else:
            abmag = stmag
    else:
        # Rough instrumental magnitude
        abmag = -2.5 * np.log10(rate) + 25.0

    if rate_err > 0 and rate > 0:
        magerr = 2.5 / np.log(10) * rate_err / rate
    else:
        magerr = 0.0

    return abmag, magerr


def plot_lightcurve(df):
    """Plot the SN 2011fe multi-band HST light curve."""
    print("\n" + "=" * 60)
    print("Step 4: Plotting light curve")
    print("=" * 60)

    if df is None or len(df) == 0:
        print("  No data to plot!")
        return

    # Quality cuts
    good = (
        df["ab_mag"].between(10, 30)
        & df["ab_mag_err"].between(0, 1.0)
        & (df["count_rate"] > 0)
    )
    df_good = df[good].copy()
    print(f"  {len(df_good)}/{len(df)} measurements pass quality cuts")

    if len(df_good) == 0:
        print("  No good measurements. Plotting count rates instead.")
        plot_count_rates(df)
        return

    # Filter colors (blue=UV, green=optical, red=IR)
    fcolors = {
        "F225W": "#7b2d8e", "F275W": "#9b59b6", "F336W": "#3498db",
        "F435W": "#2980b9", "F438W": "#2980b9",
        "F467M": "#27ae60", "F469N": "#27ae60",
        "F475W": "#2ecc71", "F475X": "#2ecc71",
        "F502N": "#1abc9c",
        "F547M": "#f1c40f", "F555W": "#f1c40f",
        "F600LP": "#e67e22", "F606W": "#e67e22",
        "F625W": "#e74c3c", "F775W": "#c0392b",
        "F814W": "#8e44ad",
        "F105W": "#d35400", "F110W": "#e74c3c",
        "F125W": "#c0392b", "F160W": "#7f0000",
    }

    # ---- Full light curve ----
    fig, ax = plt.subplots(figsize=(12, 6))

    plotted_filters = set()
    for filt, grp in df_good.groupby("filter"):
        color = fcolors.get(filt, "gray")
        # Average duplicate epochs (same MJD, same filter)
        grp_avg = grp.groupby(grp["mjd"].round(1)).agg({
            "delta_t_days": "mean",
            "ab_mag": "mean",
            "ab_mag_err": lambda x: np.sqrt(np.sum(x**2)) / len(x),
        }).reset_index(drop=True)

        ax.errorbar(
            grp_avg["delta_t_days"], grp_avg["ab_mag"],
            yerr=grp_avg["ab_mag_err"],
            fmt="o", color=color, markersize=6,
            label=filt, capsize=2, elinewidth=1,
            markeredgecolor="k", markeredgewidth=0.3,
        )
        plotted_filters.add(filt)

    ax.invert_yaxis()
    ax.set_xlabel("Days relative to B-band maximum", fontsize=13)
    ax.set_ylabel("AB Magnitude", fontsize=13)
    ax.set_title("SN 2011fe (Type Ia) — HST Light Curve Extracted by THATCH", fontsize=14)
    ax.axvline(0, color="gray", ls="--", alpha=0.5, label="B-max")
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(FIGDIR, "SN2011fe_lightcurve.pdf")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.savefig(out1.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}")

    # ---- Zoomed near-peak ----
    near_peak = df_good[df_good["delta_t_days"].between(-30, 100)]
    if len(near_peak) > 3:
        fig, ax = plt.subplots(figsize=(10, 6))
        for filt, grp in near_peak.groupby("filter"):
            color = fcolors.get(filt, "gray")
            ax.errorbar(
                grp["delta_t_days"], grp["ab_mag"],
                yerr=grp["ab_mag_err"],
                fmt="o-", color=color, markersize=7,
                label=filt, capsize=2, elinewidth=1,
                markeredgecolor="k", markeredgewidth=0.3,
            )
        ax.invert_yaxis()
        ax.axvline(0, color="gray", ls="--", alpha=0.5, label="B-max")
        ax.set_xlabel("Days relative to B-band maximum", fontsize=13)
        ax.set_ylabel("AB Magnitude", fontsize=13)
        ax.set_title("SN 2011fe — HST Near-Peak Light Curve", fontsize=14)
        ax.legend(fontsize=9, ncol=3)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(FIGDIR, "SN2011fe_lightcurve_peak.pdf")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out2}")

    # Print summary statistics
    print(f"\n  Summary:")
    print(f"    Filters with detections: {sorted(plotted_filters)}")
    print(f"    Time range: {df_good['delta_t_days'].min():.0f} to {df_good['delta_t_days'].max():.0f} days from B-max")
    print(f"    Brightest: {df_good['ab_mag'].min():.1f} mag")
    print(f"    Programs: {sorted(df_good['proposal_id'].unique())}")


def plot_count_rates(df):
    """Fallback plot using count rates."""
    pos = df[df["count_rate"] > 0].copy()
    if len(pos) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for filt, grp in pos.groupby("filter"):
        ax.scatter(grp["delta_t_days"], grp["count_rate"], label=filt, s=30)
    ax.set_yscale("log")
    ax.set_xlabel("Days from B-max")
    ax.set_ylabel("Count rate (e-/s)")
    ax.set_title("SN 2011fe — HST count rates")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "SN2011fe_countrates.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("=" * 60)
    print("THATCH DEMO: SN 2011fe (Type Ia) Light Curve")
    print("=" * 60)

    img_obs = query_and_filter()
    download_products(img_obs, max_products=100)
    df = extract_photometry()
    plot_lightcurve(df)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
