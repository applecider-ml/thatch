#!/usr/bin/env python
"""
THATCH Demo: End-to-end HST light curve extraction for AT2017gfo (GW170817).

This script demonstrates the full THATCH pipeline on one object:
1. Query MAST for all HST imaging observations of AT2017gfo
2. Download calibrated drizzled images
3. Perform aperture photometry at the transient position
4. Produce a multi-band light curve plot

Usage:
    conda activate thatch
    python demo_lightcurve.py
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

# Directories
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data", "AT2017gfo")
FIGDIR = os.path.join(BASEDIR, "..", "Figures")
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# AT2017gfo coordinates
RA = 197.45037
DEC = -23.38148
COORD = SkyCoord(ra=RA, dec=DEC, unit="deg")

# Discovery date (MJD of GW170817)
MJD_DISCOVERY = 57982.529  # 2017-08-17 12:41:04 UTC

# Zeropoints for AB magnitudes (approximate, instrument-dependent)
# These are rough AB zeropoints for HST instruments from the handbook.
# For full THATCH, we would use the image headers (PHOTFLAM, PHOTPLAM).
# Here we compute from the header keywords directly.

# Aperture corrections: encircled energy fraction within r=5 pixels.
# From WFC3 Instrument Handbook Table 7.7 (UVIS, 0.04"/pix)
# and Table 7.8 (IR, 0.13"/pix), and ACS Instrument Handbook.
# These give the fraction of total PSF flux captured in a 5-pixel
# radius aperture. We divide by this to correct to total flux.
APERTURE_CORRECTIONS = {
    # WFC3/UVIS (r=5px = 0.20"): EE ~ 0.80-0.85 depending on wavelength
    "F225W": 0.800,
    "F275W": 0.810,
    "F336W": 0.830,
    "F438W": 0.845,
    "F475W": 0.850,
    "F555W": 0.855,
    "F600LP": 0.855,
    "F606W": 0.855,
    "F625W": 0.855,
    "F775W": 0.850,
    "F814W": 0.845,
    # WFC3/IR (r=5px = 0.65"): EE ~ 0.93-0.96
    "F105W": 0.945,
    "F110W": 0.950,
    "F125W": 0.950,
    "F140W": 0.950,
    "F153M": 0.945,
    "F160W": 0.945,
    # ACS/WFC (r=5px = 0.25"): EE ~ 0.83-0.87
    "F435W": 0.835,
    "F475W": 0.845,
    "F555W": 0.850,
    "F606W": 0.855,
    "F625W": 0.855,
    "F775W": 0.850,
    "F814W": 0.845,
}


def step1_query_mast():
    """Query MAST for all HST imaging observations of AT2017gfo."""
    print("=" * 60)
    print("Step 1: Querying MAST for AT2017gfo HST observations")
    print("=" * 60)

    obs_table = Observations.query_region(COORD, radius=5.0 * u.arcsec)

    # Filter to HST imaging only
    hst_mask = obs_table["obs_collection"] == "HST"
    hst_obs = obs_table[hst_mask]

    # Keep only imaging (not spectra/grism)
    imaging_types = []
    for row in hst_obs:
        dtype = str(row["dataproduct_type"]).lower()
        intent = str(row["intentType"]).lower() if "intentType" in row.colnames else ""
        is_imaging = "image" in dtype
        # Exclude grism observations
        filt = str(row["filters"])
        is_grism = any(g in filt for g in ["G102", "G141", "G230L", "G430L", "G750L"])
        if is_imaging and not is_grism:
            imaging_types.append(True)
        else:
            imaging_types.append(False)

    imaging_mask = np.array(imaging_types)
    img_obs = hst_obs[imaging_mask]

    print(f"  Total HST observations: {len(hst_obs)}")
    print(f"  Imaging (non-grism): {len(img_obs)}")

    # Print summary by filter
    filters = {}
    for row in img_obs:
        f = str(row["filters"])
        if f not in filters:
            filters[f] = 0
        filters[f] += 1
    print("  Filters found:")
    for f, n in sorted(filters.items()):
        print(f"    {f}: {n} observations")

    return img_obs


def step2_get_products(obs_table, max_download=50):
    """Get and download calibrated data products."""
    print("\n" + "=" * 60)
    print("Step 2: Downloading calibrated HST images")
    print("=" * 60)

    # Get data products
    products = Observations.get_product_list(obs_table)

    # Filter to calibrated drizzled images only
    # _drz.fits or _drc.fits are the final combined products
    drz_mask = np.array(
        [
            any(ext in str(pn) for ext in ["_drz.fits", "_drc.fits"])
            for pn in products["productFilename"]
        ]
    )
    # Also want minimum recommended products
    mrp_mask = products["productGroupDescription"] == "Minimum Recommended Products"

    # Combine: drizzled files that are MRP
    combined_mask = drz_mask  # Take all drizzled products
    drz_products = products[combined_mask]

    if len(drz_products) == 0:
        print("  No drizzled products found. Trying _flt.fits instead...")
        flt_mask = np.array(
            ["_flt.fits" in str(pn) or "_flc.fits" in str(pn) for pn in products["productFilename"]]
        )
        drz_products = products[flt_mask]

    print(f"  Found {len(drz_products)} calibrated image products")

    # Limit downloads for demo
    if len(drz_products) > max_download:
        print(f"  Limiting download to {max_download} products for demo")
        drz_products = drz_products[:max_download]

    # Download
    print(f"  Downloading {len(drz_products)} files to {DATADIR}...")
    manifest = Observations.download_products(
        drz_products,
        download_dir=DATADIR,
        flat=True,  # Put all files in one directory
    )

    downloaded = [
        str(f)
        for f in manifest["Local Path"]
        if "COMPLETE" in str(manifest["Status"][manifest["Local Path"] == f])
    ]
    print(f"  Successfully downloaded {len(manifest)} files")

    return manifest


def step3_extract_photometry():
    """Extract aperture photometry from downloaded HST images."""
    print("\n" + "=" * 60)
    print("Step 3: Extracting aperture photometry")
    print("=" * 60)

    # Find all downloaded FITS files
    fits_files = glob(os.path.join(DATADIR, "**", "*.fits"), recursive=True)
    # Filter to drz/drc/flt/flc
    fits_files = [
        f
        for f in fits_files
        if any(ext in f for ext in ["_drz.fits", "_drc.fits", "_flt.fits", "_flc.fits"])
    ]

    print(f"  Found {len(fits_files)} FITS files to process")

    photometry_records = []

    for fpath in sorted(fits_files):
        fname = os.path.basename(fpath)
        try:
            with fits.open(fpath) as hdul:
                # Find the science extension
                sci_ext = None
                for i, hdu in enumerate(hdul):
                    if hdu.name == "SCI" or (
                        hasattr(hdu, "data")
                        and hdu.data is not None
                        and hdu.data.ndim == 2
                        and i > 0
                    ):
                        sci_ext = i
                        break
                if sci_ext is None and hdul[0].data is not None and hdul[0].data.ndim == 2:
                    sci_ext = 0

                if sci_ext is None:
                    continue

                header = hdul[sci_ext].header
                data = hdul[sci_ext].data.astype(float)

                # Also check primary header for keywords
                pri_header = hdul[0].header

                # Get observation metadata
                filt = (
                    header.get("FILTER", None)
                    or header.get("FILTER1", None)
                    or pri_header.get("FILTER", None)
                    or pri_header.get("FILTER1", None)
                    or "UNKNOWN"
                )
                # Some instruments use FILTER2 for the actual filter
                filt2 = header.get("FILTER2", pri_header.get("FILTER2", ""))
                if filt in ["CLEAR1L", "CLEAR1S", "CLEAR", "N/A"] and filt2:
                    filt = filt2
                if (
                    filt2
                    and filt2 not in ["CLEAR2L", "CLEAR2S", "CLEAR", "N/A", ""]
                    and filt2 != filt
                ):
                    # For ACS, the filter might be in FILTER1 or FILTER2
                    if "CLEAR" in str(filt):
                        filt = filt2

                instrument = pri_header.get("INSTRUME", "") + "/" + pri_header.get("DETECTOR", "")

                # Get observation time
                date_obs = pri_header.get("DATE-OBS", header.get("DATE-OBS"))
                time_obs = pri_header.get("TIME-OBS", header.get("TIME-OBS", "00:00:00"))
                exptime = float(header.get("EXPTIME", pri_header.get("EXPTIME", 1.0)))

                if date_obs is None:
                    continue

                try:
                    t = Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")
                    mjd = t.mjd
                except Exception:
                    continue

                # Get WCS and find pixel position of transient
                try:
                    wcs = WCS(header, naxis=2)
                    x_pix, y_pix = wcs.world_to_pixel(COORD)
                    x_pix, y_pix = float(x_pix), float(y_pix)
                except Exception:
                    continue

                # Check if position is within image
                ny, nx = data.shape
                if not (5 < x_pix < nx - 5 and 5 < y_pix < ny - 5):
                    continue

                # Get photometric calibration from header
                photflam = float(header.get("PHOTFLAM", pri_header.get("PHOTFLAM", 0)))
                photplam = float(header.get("PHOTPLAM", pri_header.get("PHOTPLAM", 0)))
                photzpt = float(header.get("PHOTZPT", pri_header.get("PHOTZPT", -21.1)))

                # Aperture photometry
                # Use a simple circular aperture (radius=5 pixels)
                # and annular background (inner=10, outer=15 pixels)
                ap_radius = 5
                bg_inner = 10
                bg_outer = 15

                yy, xx = np.mgrid[:ny, :nx]
                dist = np.sqrt((xx - x_pix) ** 2 + (yy - y_pix) ** 2)

                # Source aperture
                ap_mask = dist <= ap_radius
                # Background annulus
                bg_mask = (dist >= bg_inner) & (dist <= bg_outer)

                # Handle NaN/inf values
                valid_ap = ap_mask & np.isfinite(data)
                valid_bg = bg_mask & np.isfinite(data)

                if np.sum(valid_ap) < 10 or np.sum(valid_bg) < 20:
                    continue

                # Background estimation
                bg_pixels = data[valid_bg]
                _, bg_median, bg_std = sigma_clipped_stats(bg_pixels, sigma=3.0)

                # Source flux (background-subtracted)
                src_flux = np.sum(data[valid_ap]) - bg_median * np.sum(valid_ap)
                src_flux_err = np.sqrt(
                    np.sum(valid_ap) * bg_std**2 + np.abs(src_flux)  # Poisson noise (approximate)
                )

                # Convert to count rate (electrons/sec)
                if exptime > 0:
                    count_rate = src_flux / exptime
                    count_rate_err = src_flux_err / exptime
                else:
                    # drz files are already in electrons/sec
                    count_rate = src_flux
                    count_rate_err = src_flux_err

                # Apply aperture correction (divide by encircled energy fraction)
                ee_frac = APERTURE_CORRECTIONS.get(filt, 0.90)
                count_rate /= ee_frac
                count_rate_err /= ee_frac

                # Convert to AB magnitude
                if photflam > 0 and count_rate > 0:
                    # AB mag = -2.5*log10(PHOTFLAM * count_rate) - 21.10 - 5*log10(PHOTPLAM) + 18.6921
                    # Simplified: use STmag zeropoint
                    flux_cgs = count_rate * photflam  # erg/s/cm^2/Angstrom
                    if flux_cgs > 0:
                        stmag = -2.5 * np.log10(flux_cgs) + photzpt
                        # Convert ST mag to AB mag
                        if photplam > 0:
                            abmag = stmag - 5.0 * np.log10(photplam) + 18.6921
                        else:
                            abmag = stmag  # Approximate
                        # Error propagation
                        if count_rate_err > 0 and count_rate > 0:
                            mag_err = 2.5 / np.log(10) * count_rate_err / count_rate
                        else:
                            mag_err = 0.0
                    else:
                        abmag = np.nan
                        mag_err = np.nan
                elif count_rate > 0:
                    # No PHOTFLAM; use instrumental magnitude
                    abmag = -2.5 * np.log10(count_rate) + 25.0  # rough zeropoint
                    mag_err = (
                        2.5 / np.log(10) * count_rate_err / count_rate if count_rate_err > 0 else 0
                    )
                else:
                    abmag = np.nan
                    mag_err = np.nan

                # Days since discovery
                delta_t = mjd - MJD_DISCOVERY

                record = {
                    "filename": fname,
                    "mjd": mjd,
                    "delta_t_days": delta_t,
                    "filter": filt,
                    "instrument": instrument,
                    "exptime_s": exptime,
                    "count_rate": count_rate,
                    "count_rate_err": count_rate_err,
                    "ab_mag": abmag,
                    "ab_mag_err": mag_err,
                    "x_pix": x_pix,
                    "y_pix": y_pix,
                    "photflam": photflam,
                    "photplam": photplam,
                    "bg_median": bg_median,
                    "bg_std": bg_std,
                }
                photometry_records.append(record)
                print(
                    f"  {fname}: {filt} MJD={mjd:.3f} "
                    f"dt={delta_t:.1f}d "
                    f"rate={count_rate:.3f} "
                    f"mag={abmag:.2f}+/-{mag_err:.2f}"
                )

        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue

    if not photometry_records:
        print("  WARNING: No photometry extracted!")
        return None

    df = pd.DataFrame(photometry_records)
    df = df.sort_values("mjd")

    outfile = os.path.join(DATADIR, "AT2017gfo_photometry.csv")
    df.to_csv(outfile, index=False)
    print(f"\n  Saved {len(df)} photometric measurements to {outfile}")

    return df


def step4_plot_lightcurve(df):
    """Plot the multi-band HST light curve."""
    print("\n" + "=" * 60)
    print("Step 4: Plotting multi-band light curve")
    print("=" * 60)

    if df is None or len(df) == 0:
        print("  No photometry to plot!")
        return

    # Filter to reasonable measurements
    good = df["ab_mag"].between(15, 35) & df["ab_mag_err"].between(0, 2)
    df_good = df[good].copy()

    if len(df_good) == 0:
        print("  No good photometric measurements to plot!")
        # Fall back: plot count rates
        plot_count_rates(df)
        return

    # Color scheme for filters
    filter_colors = {
        "F606W": "#1f77b4",
        "F475W": "#2ca02c",
        "F625W": "#ff7f0e",
        "F775W": "#d62728",
        "F814W": "#9467bd",
        "F850LP": "#8c564b",
        "F110W": "#e377c2",
        "F140W": "#7f7f7f",
        "F160W": "#bcbd22",
        "F336W": "#17becf",
        "F275W": "#aec7e8",
        "F225W": "#c5b0d5",
        "F153M": "#c49c94",
        "F502N": "#98df8a",
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Group by filter
    for filt, group in df_good.groupby("filter"):
        color = filter_colors.get(filt, "gray")
        ax1.errorbar(
            group["delta_t_days"],
            group["ab_mag"],
            yerr=group["ab_mag_err"],
            fmt="o",
            color=color,
            markersize=6,
            label=filt,
            capsize=2,
            elinewidth=1,
            markeredgecolor="k",
            markeredgewidth=0.5,
        )

    ax1.invert_yaxis()
    ax1.set_ylabel("AB Magnitude", fontsize=13)
    ax1.set_title("AT2017gfo (GW170817) — HST Light Curve Extracted by THATCH", fontsize=14)
    ax1.legend(fontsize=8, ncol=3, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: number of observations per epoch
    for filt, group in df_good.groupby("filter"):
        color = filter_colors.get(filt, "gray")
        ax2.scatter(
            group["delta_t_days"],
            [filt] * len(group),
            c=color,
            s=30,
            edgecolors="k",
            linewidths=0.3,
        )

    ax2.set_xlabel("Days since GW170817 (2017-08-17)", fontsize=13)
    ax2.set_ylabel("Filter", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = os.path.join(FIGDIR, "AT2017gfo_lightcurve.pdf")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.savefig(outfile.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outfile}")

    # Also make a zoomed early-time plot
    early = df_good[df_good["delta_t_days"] < 30]
    if len(early) > 3:
        fig, ax = plt.subplots(figsize=(8, 5))
        for filt, group in early.groupby("filter"):
            color = filter_colors.get(filt, "gray")
            ax.errorbar(
                group["delta_t_days"],
                group["ab_mag"],
                yerr=group["ab_mag_err"],
                fmt="o-",
                color=color,
                markersize=7,
                label=filt,
                capsize=2,
                elinewidth=1,
                markeredgecolor="k",
                markeredgewidth=0.5,
            )
        ax.invert_yaxis()
        ax.set_xlabel("Days since GW170817", fontsize=13)
        ax.set_ylabel("AB Magnitude", fontsize=13)
        ax.set_title("AT2017gfo Early HST Light Curve (first 30 days)", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        outfile2 = os.path.join(FIGDIR, "AT2017gfo_lightcurve_early.pdf")
        plt.savefig(outfile2, dpi=150, bbox_inches="tight")
        plt.savefig(outfile2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outfile2}")


def plot_count_rates(df):
    """Fallback: plot count rates if mag calibration failed."""
    filter_colors = {
        "F606W": "#1f77b4",
        "F475W": "#2ca02c",
        "F625W": "#ff7f0e",
        "F775W": "#d62728",
        "F814W": "#9467bd",
        "F850LP": "#8c564b",
        "F110W": "#e377c2",
        "F140W": "#7f7f7f",
        "F160W": "#bcbd22",
        "F336W": "#17becf",
        "F275W": "#aec7e8",
    }

    good = df[df["count_rate"] > 0].copy()
    if len(good) == 0:
        print("  No positive count rates to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for filt, group in good.groupby("filter"):
        color = filter_colors.get(filt, "gray")
        ax.errorbar(
            group["delta_t_days"],
            group["count_rate"],
            yerr=group["count_rate_err"],
            fmt="o",
            color=color,
            markersize=6,
            label=filt,
            capsize=2,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Days since GW170817", fontsize=13)
    ax.set_ylabel("Count Rate (e-/s)", fontsize=13)
    ax.set_title("AT2017gfo — HST Count Rates (THATCH extraction)", fontsize=14)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = os.path.join(FIGDIR, "AT2017gfo_countrates.pdf")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.savefig(outfile.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outfile}")


def main():
    print("=" * 60)
    print("THATCH DEMO: AT2017gfo Light Curve Extraction")
    print("=" * 60)

    # Step 1: Query MAST
    img_obs = step1_query_mast()

    # Step 2: Download data
    manifest = step2_get_products(img_obs, max_download=50)

    # Step 3: Extract photometry
    df = step3_extract_photometry()

    # Step 4: Plot light curve
    step4_plot_lightcurve(df)

    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    if df is not None:
        print(f"  Photometric measurements: {len(df)}")
        print(f"  Filters: {sorted(df['filter'].unique())}")
        print(
            f"  Time range: {df['delta_t_days'].min():.1f} to {df['delta_t_days'].max():.1f} days"
        )
        print(f"  Data: {DATADIR}/AT2017gfo_photometry.csv")
    print(f"  Figures: {FIGDIR}/")


if __name__ == "__main__":
    main()
