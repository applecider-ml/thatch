#!/usr/bin/env python
"""
THATCH Batch Photometry Pipeline.

Runs the full HST aperture-photometry pipeline on multiple transients:
  1. Query MAST for HST imaging observations
  2. Download calibrated drizzled images (drz/drc)
  3. Perform aperture photometry with aperture corrections
  4. Save per-object photometry CSVs

Skips objects/images that have already been processed.
Includes retry logic for MAST intermittent failures.

Usage:
    conda activate thatch
    python run_photometry_batch.py
"""

import os
import sys
import time
import warnings
import traceback
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

# ============================================================
# Configuration
# ============================================================
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data")
FIGDIR = os.path.join(BASEDIR, "..", "Figures")
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

MAX_DOWNLOAD_PER_OBJECT = 80   # cap downloads per object
MAX_RETRIES = 3                # MAST retry attempts
RETRY_DELAY = 10               # seconds between retries

# Aperture corrections: encircled energy fraction within r=5 pixels.
APERTURE_CORRECTIONS = {
    "F225W": 0.800, "F275W": 0.810, "F336W": 0.830,
    "F438W": 0.845, "F475W": 0.850, "F555W": 0.855,
    "F600LP": 0.855, "F606W": 0.855, "F625W": 0.855,
    "F775W": 0.850, "F814W": 0.845,
    "F105W": 0.945, "F110W": 0.950, "F125W": 0.950,
    "F140W": 0.950, "F153M": 0.945, "F160W": 0.945,
    "F435W": 0.835,
    # extras for broader filter coverage
    "F218W": 0.790, "F390M": 0.840, "F410M": 0.845,
    "F467M": 0.850, "F469N": 0.850, "F475X": 0.850,
    "F502N": 0.855, "F547M": 0.855, "F631N": 0.855,
    "F656N": 0.855, "F658N": 0.855, "F673N": 0.855,
    "F845M": 0.845, "F850LP": 0.840,
    "F128N": 0.945, "F164N": 0.940, "F373N": 0.830,
    "F280N": 0.810, "F487N": 0.850,
    "F439W": 0.840, "F702W": 0.855, "F675W": 0.855,
    "F336W": 0.830, "F160LP": 0.800,
}

# Target transients with coordinates
TARGETS = [
    {
        "name": "SN2014J",
        "ra": 148.92554,
        "dec": 69.67387,
        "type": "SN Ia",
        "redshift": 0.000677,
        "search_radius_arcsec": 5.0,
    },
    {
        "name": "SN_Refsdal",
        "ra": 177.39792,
        "dec": 22.39569,
        "type": "SN II (lensed)",
        "redshift": 1.489,
        "search_radius_arcsec": 5.0,
    },
    {
        "name": "SN1987A",
        "ra": 83.86667,
        "dec": -69.26972,
        "type": "SN II-pec",
        "redshift": 0.000927,
        "search_radius_arcsec": 5.0,
    },
    {
        "name": "Sw_J1644+57",
        "ra": 251.20529,
        "dec": 57.58089,
        "type": "TDE (jetted)",
        "redshift": 0.354,
        "search_radius_arcsec": 5.0,
    },
]


# ============================================================
# Utility: retry wrapper for MAST calls
# ============================================================
def mast_retry(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Call a MAST function with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if attempt < max_retries:
                wait = RETRY_DELAY * attempt
                print(f"    MAST call failed (attempt {attempt}/{max_retries}): {msg}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"MAST call failed after {max_retries} attempts: {msg}"
                ) from e


# ============================================================
# Step 1 - Query MAST
# ============================================================
def query_mast_imaging(ra, dec, radius_arcsec=5.0):
    """Query MAST for HST imaging (non-grism) observations."""
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    obs_table = mast_retry(
        Observations.query_region, coord, radius=radius_arcsec * u.arcsec
    )

    # Filter to HST
    hst_mask = obs_table["obs_collection"] == "HST"
    hst_obs = obs_table[hst_mask]

    # Keep imaging, exclude grisms/spectroscopy
    keep = []
    grism_tokens = [
        "G102", "G141", "G130M", "G140L", "G160M",
        "G230L", "G230LB", "G430L", "G430M",
        "G750L", "G750M", "MIRVIS", "MIRFUV", "PRISM",
    ]
    for row in hst_obs:
        dtype = str(row["dataproduct_type"]).lower()
        filt = str(row["filters"])
        is_imaging = "image" in dtype
        is_grism = any(g in filt for g in grism_tokens)
        keep.append(is_imaging and not is_grism)

    img_obs = hst_obs[np.array(keep)]
    return img_obs


# ============================================================
# Step 2 - Download drz/drc products
# ============================================================
def download_drz_products(obs_table, dest_dir, max_products=MAX_DOWNLOAD_PER_OBJECT):
    """Download drizzled FITS products to dest_dir. Returns manifest."""
    os.makedirs(dest_dir, exist_ok=True)

    # Already-downloaded files (skip re-download)
    existing = set(os.path.basename(f) for f in
                   glob(os.path.join(dest_dir, "*.fits")))

    products = mast_retry(Observations.get_product_list, obs_table)

    # Filter to drz/drc
    drz_mask = np.array([
        any(ext in str(pn) for ext in ["_drz.fits", "_drc.fits"])
        for pn in products["productFilename"]
    ])
    drz_products = products[drz_mask]

    # Remove already-downloaded
    new_mask = np.array([
        str(pn) not in existing for pn in drz_products["productFilename"]
    ])
    drz_new = drz_products[new_mask]

    if len(drz_new) == 0 and len(drz_products) > 0:
        print(f"    All {len(drz_products)} drizzled products already downloaded")
        return None

    if len(drz_new) > max_products:
        print(f"    Limiting download to {max_products} of {len(drz_new)} new products")
        drz_new = drz_new[:max_products]

    print(f"    Downloading {len(drz_new)} new drizzled products...")
    manifest = mast_retry(
        Observations.download_products,
        drz_new,
        download_dir=dest_dir,
        flat=True,
    )
    n_ok = sum(1 for s in manifest["Status"] if "COMPLETE" in str(s))
    print(f"    Downloaded {n_ok}/{len(manifest)} files")
    return manifest


# ============================================================
# Step 3 - Aperture photometry helpers
# ============================================================
def get_filter_name(header, pri):
    """Extract the actual filter name from HST FITS headers.
    Handles WFC3, ACS, and WFPC2 naming conventions.
    """
    # WFPC2 uses FILTNAM1/FILTNAM2 for proper filter names
    for hdr in [header, pri]:
        for key in ["FILTNAM1", "FILTNAM2"]:
            val = str(hdr.get(key, "")).strip()
            if val and val.startswith("F") and len(val) >= 4:
                return val

    # Standard: FILTER, FILTER1, FILTER2
    for hdr in [header, pri]:
        for key in ["FILTER", "FILTER1", "FILTER2"]:
            val = str(hdr.get(key, "")).strip()
            if (val and val.startswith("F") and len(val) >= 4
                    and "CLEAR" not in val.upper()):
                return val

    # Fallback: try PHOTMODE to extract filter name (e.g. "WFPC2,1,A2D7,F547M,,CAL")
    for hdr in [header, pri]:
        photmode = str(hdr.get("PHOTMODE", ""))
        for token in photmode.split(","):
            token = token.strip()
            if token.startswith("F") and len(token) >= 4 and token[1:].replace("W", "").replace("M", "").replace("N", "").replace("LP", "").replace("X", "")[:3].isdigit():
                return token

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
        abmag = stmag - 5.0 * np.log10(photplam) + 18.6921 if photplam > 0 else stmag
    else:
        abmag = -2.5 * np.log10(rate) + 25.0
    magerr = (2.5 / np.log(10) * rate_err / rate) if (rate_err > 0 and rate > 0) else 0.0
    return abmag, magerr


def measure_one_image(fpath, coord):
    """Run aperture photometry on one FITS image at the given sky coordinate.
    Returns a dict of measurements or None on failure.
    """
    fname = os.path.basename(fpath)
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

        filt = get_filter_name(header, pri)
        instrument = f"{pri.get('INSTRUME', '?')}/{pri.get('DETECTOR', '?')}"

        # Observation time
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
            x_pix, y_pix = wcs.world_to_pixel(coord)
            x_pix, y_pix = float(x_pix), float(y_pix)
        except Exception:
            return None

        ny, nx = data.shape
        if not (8 < x_pix < nx - 8 and 8 < y_pix < ny - 8):
            return None

        # Photometric calibration keywords
        photflam = float(header.get("PHOTFLAM", pri.get("PHOTFLAM", 0)))
        photplam = float(header.get("PHOTPLAM", pri.get("PHOTPLAM", 0)))
        photzpt = float(header.get("PHOTZPT", pri.get("PHOTZPT", -21.1)))

        # Aperture photometry: r=5 px source, annulus 10-15 px background
        ap_r = 5
        bg_in, bg_out = 10, 15

        yy, xx = np.mgrid[:ny, :nx]
        dist = np.sqrt((xx - x_pix) ** 2 + (yy - y_pix) ** 2)

        ap_mask = dist <= ap_r
        bg_mask = (dist >= bg_in) & (dist <= bg_out)
        valid_ap = ap_mask & np.isfinite(data)
        valid_bg = bg_mask & np.isfinite(data)

        if np.sum(valid_ap) < 10 or np.sum(valid_bg) < 20:
            return None

        bg_pixels = data[valid_bg]
        _, bg_med, bg_std = sigma_clipped_stats(bg_pixels, sigma=3.0)

        n_ap = float(np.sum(valid_ap))
        src_flux = np.sum(data[valid_ap]) - bg_med * n_ap
        src_err = np.sqrt(n_ap * bg_std ** 2 + np.abs(src_flux))

        # Count rate (drz files are already in e-/s typically)
        bunit = str(header.get("BUNIT", pri.get("BUNIT", ""))).strip().upper()
        if "SECOND" in bunit or "/S" in bunit:
            rate, rate_err = src_flux, src_err
        elif exptime > 0:
            rate, rate_err = src_flux / exptime, src_err / exptime
        else:
            rate, rate_err = src_flux, src_err

        # Aperture correction
        ee_frac = APERTURE_CORRECTIONS.get(filt, 0.85)
        rate /= ee_frac
        rate_err /= ee_frac

        abmag, magerr = compute_abmag(rate, rate_err, photflam, photplam, photzpt)

        return {
            "filename": fname,
            "mjd": mjd,
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


# ============================================================
# Step 3 main: run photometry on all downloaded images
# ============================================================
def extract_photometry(data_dir, coord):
    """Run aperture photometry on all drz/drc FITS files in data_dir."""
    fits_files = glob(os.path.join(data_dir, "**", "*.fits"), recursive=True)
    fits_files = [f for f in fits_files if any(
        ext in f for ext in ["_drz.fits", "_drc.fits"]
    )]
    if not fits_files:
        return None

    records = []
    for fpath in sorted(fits_files):
        try:
            rec = measure_one_image(fpath, coord)
            if rec is not None:
                records.append(rec)
        except Exception:
            continue

    if not records:
        return None

    df = pd.DataFrame(records)
    df = df.sort_values("mjd").reset_index(drop=True)
    return df


# ============================================================
# Main batch runner
# ============================================================
def process_one_target(target):
    """Full pipeline for one transient. Returns (name, photometry_df or None)."""
    name = target["name"]
    ra, dec = target["ra"], target["dec"]
    radius = target.get("search_radius_arcsec", 5.0)
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    obj_dir = os.path.join(DATADIR, name)
    phot_file = os.path.join(obj_dir, f"{name}_photometry.csv")

    # Check if already fully processed
    if os.path.exists(phot_file):
        existing = pd.read_csv(phot_file)
        if len(existing) > 0:
            print(f"  Already processed ({len(existing)} measurements). Skipping.")
            return name, existing

    print(f"  Querying MAST (radius={radius}\")")
    img_obs = query_mast_imaging(ra, dec, radius_arcsec=radius)
    n_img = len(img_obs) if img_obs is not None else 0
    print(f"  Found {n_img} HST imaging observations")

    if n_img == 0:
        print(f"  No imaging observations found. Skipping.")
        return name, None

    # Download
    print(f"  Downloading drizzled products...")
    download_drz_products(img_obs, obj_dir)

    # Photometry
    print(f"  Extracting aperture photometry...")
    df = extract_photometry(obj_dir, coord)

    if df is not None and len(df) > 0:
        df.insert(0, "object", name)
        df.to_csv(phot_file, index=False)
        n_good = df["ab_mag"].between(10, 35).sum()
        print(f"  Saved {len(df)} measurements ({n_good} with valid mags) -> {phot_file}")
    else:
        print(f"  No photometry could be extracted.")

    return name, df


def main():
    print("=" * 70)
    print("THATCH Batch Photometry Pipeline")
    print(f"Targets: {', '.join(t['name'] for t in TARGETS)}")
    print("=" * 70)

    results = {}
    for i, target in enumerate(TARGETS):
        name = target["name"]
        print(f"\n[{i+1}/{len(TARGETS)}] === {name} (z={target['redshift']}, {target['type']}) ===")
        try:
            _, df = process_one_target(target)
            results[name] = df
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[name] = None

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for name, df in results.items():
        if df is not None and len(df) > 0:
            n = len(df)
            filts = sorted(df["filter"].unique())
            mjd_range = f"{df['mjd'].min():.1f} - {df['mjd'].max():.1f}"
            print(f"  {name:20s}: {n:4d} measurements, filters={filts}, MJD={mjd_range}")
        else:
            print(f"  {name:20s}: NO DATA")
    print("=" * 70)


if __name__ == "__main__":
    main()
