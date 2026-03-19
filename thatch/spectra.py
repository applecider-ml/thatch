#!/usr/bin/env python
"""
thatch-spectra: HST spectral extraction for transients.

Queries MAST for spectroscopic observations (STIS slit spectra, WFC3/IR grisms,
ACS grisms) of known transients, downloads pipeline-extracted 1D spectra
(x1d/sx1 files), and produces standardized spectral products.

For STIS and COS, the HST calibration pipeline (calstis/calcos) produces
high-quality 1D extracted spectra (x1d.fits). For grism data, we use the
pipeline-extracted spectra when available, or flag for hstaxe reprocessing.

Usage:
    conda activate thatch
    python hustle_spectra.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astroquery.mast import Observations

warnings.filterwarnings("ignore")

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data")
FIGDIR = os.path.join(BASEDIR, "..", "Figures")
os.makedirs(FIGDIR, exist_ok=True)

# Spectroscopic instrument/grating configurations
STIS_GRATINGS = [
    "G140L", "G140M", "G230L", "G230LB", "G230MB",
    "G430L", "G430M", "G750L", "G750M",
]
WFC3_GRISMS = ["G102", "G141", "G280"]
ACS_GRISMS = ["G800L"]
COS_GRATINGS = ["G130M", "G160M", "G230L", "G140L"]

ALL_SPECTROSCOPIC = STIS_GRATINGS + WFC3_GRISMS + ACS_GRISMS + COS_GRATINGS


def query_spectroscopic_obs(coord, radius=5.0, name="transient"):
    """Query MAST for all HST spectroscopic observations near a position.

    Parameters
    ----------
    coord : SkyCoord
        Target coordinates.
    radius : float
        Search radius in arcseconds.
    name : str
        Object name for logging.

    Returns
    -------
    pd.DataFrame
        Table of spectroscopic observations with metadata.
    """
    print(f"Querying MAST for spectroscopic obs of {name}...")

    obs = Observations.query_region(coord, radius=radius * u.arcsec)
    hst = obs[obs["obs_collection"] == "HST"]

    records = []
    for row in hst:
        filt = str(row["filters"])
        dtype = str(row["dataproduct_type"]).lower()
        instrument = str(row["instrument_name"])

        # Identify spectroscopic observations
        is_spec_type = "spectrum" in dtype or "spectr" in dtype
        is_grism = any(g in filt for g in ALL_SPECTROSCOPIC)

        if not (is_spec_type or is_grism):
            continue

        try:
            t_min = float(row["t_min"])
            t_max = float(row["t_max"])
        except (ValueError, TypeError):
            continue

        mjd_mid = (t_min + t_max) / 2.0
        exptime = float(row["t_exptime"]) if row["t_exptime"] else 0

        # Determine spectroscopic mode
        if any(g in filt for g in STIS_GRATINGS):
            mode = "STIS_slit"
        elif any(g in filt for g in COS_GRATINGS):
            mode = "COS_slit"
        elif any(g in filt for g in WFC3_GRISMS):
            mode = "WFC3_grism"
        elif any(g in filt for g in ACS_GRISMS):
            mode = "ACS_grism"
        else:
            mode = "unknown"

        records.append({
            "obs_id": str(row["obs_id"]),
            "mjd": mjd_mid,
            "grating": filt,
            "instrument": instrument,
            "mode": mode,
            "proposal_id": str(row["proposal_id"]),
            "exptime_s": exptime,
            "dataproduct_type": dtype,
        })

    df = pd.DataFrame(records).sort_values("mjd")
    print(f"  Found {len(df)} spectroscopic observations")
    if len(df) > 0:
        for _, row in df.iterrows():
            print(f"    {row['mode']:12s} {row['grating']:10s} "
                  f"MJD={row['mjd']:.1f} {row['instrument']:14s} "
                  f"prog={row['proposal_id']} {row['obs_id']}")
    return df


def download_extracted_spectra(obs_df, outdir, max_download=50):
    """Download pipeline-extracted 1D spectra (x1d, sx1, x1dsum files).

    Parameters
    ----------
    obs_df : pd.DataFrame
        Observation table from query_spectroscopic_obs.
    outdir : str
        Directory to save downloaded files.
    max_download : int
        Maximum number of products to download.

    Returns
    -------
    list of str
        Paths to downloaded spectral files.
    """
    os.makedirs(outdir, exist_ok=True)

    if len(obs_df) == 0:
        print("  No spectroscopic observations to download.")
        return []

    print(f"\nDownloading extracted spectra to {outdir}...")

    # Get observation IDs
    obs_ids = obs_df["obs_id"].tolist()

    # Query MAST for these specific observations
    obs_table = Observations.query_criteria(obs_id=obs_ids)

    # Get data products
    products = Observations.get_product_list(obs_table)

    # Filter to extracted 1D spectra (STIS/COS x1d, plus HAP grism extractions)
    spec_extensions = ["_x1d.fits", "_sx1.fits", "_x1dsum.fits",
                       "_x2d.fits", "_sx2.fits",
                       "_1d.fits", "_opt.fits"]  # HAP grism extractions
    spec_mask = np.array([
        any(ext in str(pn) for ext in spec_extensions)
        for pn in products["productFilename"]
    ])
    spec_products = products[spec_mask]

    print(f"  Found {len(spec_products)} extracted spectral products")

    if len(spec_products) == 0:
        # For grism data, try getting the direct + grism images
        print("  No pipeline-extracted spectra. Trying grism images...")
        grism_mask = np.array([
            any(ext in str(pn) for ext in ["_flt.fits", "_flc.fits", "_drz.fits"])
            for pn in products["productFilename"]
        ])
        spec_products = products[grism_mask]
        print(f"  Found {len(spec_products)} grism image products")

    if len(spec_products) > max_download:
        print(f"  Limiting to {max_download} products")
        spec_products = spec_products[:max_download]

    if len(spec_products) == 0:
        return []

    manifest = Observations.download_products(
        spec_products, download_dir=outdir, flat=True
    )

    downloaded = [str(row["Local Path"]) for _, row in manifest.to_pandas().iterrows()
                  if str(row["Status"]) == "COMPLETE"]
    print(f"  Downloaded {len(downloaded)} files")
    return downloaded


def read_x1d_spectrum(fpath):
    """Read a STIS/COS x1d.fits extracted spectrum.

    Parameters
    ----------
    fpath : str
        Path to x1d.fits file.

    Returns
    -------
    dict
        Spectrum record with wavelength, flux, error, and metadata.
    """
    fname = os.path.basename(fpath)

    with fits.open(fpath) as hdul:
        pri = hdul[0].header

        instrument = pri.get("INSTRUME", "UNKNOWN")
        grating = pri.get("OPT_ELEM", pri.get("FILTER", "UNKNOWN"))
        detector = pri.get("DETECTOR", "")
        exptime = float(pri.get("TEXPTIME", pri.get("EXPTIME", 0)))
        proposal = str(pri.get("PROPOSID", ""))
        target = pri.get("TARGNAME", "")

        # Get MJD: try EXPSTART from ext 1 first, then DATE-OBS from either
        ext1_header = hdul[1].header if len(hdul) > 1 else {}
        expstart = (ext1_header.get("EXPSTART") or pri.get("EXPSTART"))
        if expstart is not None:
            mjd = float(expstart)
        else:
            date_obs = (ext1_header.get("DATE-OBS") or
                        pri.get("DATE-OBS", ""))
            time_obs = (ext1_header.get("TIME-OBS") or
                        pri.get("TIME-OBS", "00:00:00"))
            try:
                t = Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")
                mjd = t.mjd
            except Exception:
                mjd = np.nan

        # x1d files have spectral data in extension 1
        if len(hdul) < 2:
            return None

        data = hdul[1].data
        if data is None or len(data) == 0:
            return None

        # x1d tables can have multiple spectral orders
        # Combine them or take the first/primary
        all_wave = []
        all_flux = []
        all_err = []

        for row in data:
            wave = row["WAVELENGTH"]
            flux = row["FLUX"]
            err = row["ERROR"]
            npts = len(wave)

            # Filter out bad data
            good = (wave > 0) & np.isfinite(flux) & np.isfinite(err)
            if np.sum(good) > 10:
                all_wave.append(wave[good])
                all_flux.append(flux[good])
                all_err.append(err[good])

        if not all_wave:
            return None

        # Concatenate and sort by wavelength
        wave = np.concatenate(all_wave)
        flux = np.concatenate(all_flux)
        err = np.concatenate(all_err)

        sort_idx = np.argsort(wave)
        wave = wave[sort_idx]
        flux = flux[sort_idx]
        err = err[sort_idx]

    return {
        "filename": fname,
        "mjd": mjd,
        "instrument": f"{instrument}/{detector}",
        "grating": grating,
        "exptime_s": exptime,
        "proposal_id": proposal,
        "target": target,
        "wavelength": wave,  # Angstroms
        "flux": flux,  # erg/s/cm^2/Angstrom
        "flux_err": err,
        "wave_min": wave.min(),
        "wave_max": wave.max(),
        "n_pixels": len(wave),
    }


def process_spectral_directory(specdir):
    """Process all x1d/sx1 files in a directory.

    Parameters
    ----------
    specdir : str
        Directory containing spectral FITS files.

    Returns
    -------
    list of dict
        List of spectrum records.
    """
    from glob import glob

    spec_files = []
    for ext in ["*_x1d.fits", "*_sx1.fits", "*_x1dsum.fits"]:
        spec_files.extend(glob(os.path.join(specdir, ext)))

    print(f"\nProcessing {len(spec_files)} extracted spectra in {specdir}")

    spectra = []
    for fpath in sorted(spec_files):
        fname = os.path.basename(fpath)
        try:
            spec = read_x1d_spectrum(fpath)
            if spec is not None:
                spectra.append(spec)
                print(f"  {fname}: {spec['grating']} "
                      f"MJD={spec['mjd']:.3f} "
                      f"{spec['wave_min']:.0f}-{spec['wave_max']:.0f}A "
                      f"({spec['n_pixels']} px)")
            else:
                print(f"  {fname}: no valid data")
        except Exception as e:
            print(f"  {fname}: ERROR {e}")

    return spectra


def save_spectra_hdf5(spectra, outpath, object_name="transient"):
    """Save extracted spectra to HDF5 (primary format).

    Each group contains wavelength, flux, and flux_err arrays
    plus metadata attributes.

    Parameters
    ----------
    spectra : list of dict
        Spectrum records from read_x1d_spectrum.
    outpath : str
        Output HDF5 file path.
    object_name : str
        Object name for metadata.
    """
    import h5py

    with h5py.File(outpath, "w") as f:
        f.attrs["object_name"] = object_name
        f.attrs["n_spectra"] = len(spectra)

        for i, spec in enumerate(spectra):
            grp = f.create_group(f"spec_{i:03d}")
            grp.create_dataset("wavelength", data=spec["wavelength"],
                               compression="gzip")
            grp.create_dataset("flux", data=spec["flux"],
                               compression="gzip")
            grp.create_dataset("flux_err", data=spec["flux_err"],
                               compression="gzip")

            grp.attrs["filename"] = spec["filename"]
            grp.attrs["mjd"] = spec["mjd"] if np.isfinite(spec["mjd"]) else 0.0
            grp.attrs["instrument"] = spec["instrument"]
            grp.attrs["grating"] = spec["grating"]
            grp.attrs["exptime_s"] = spec["exptime_s"]
            grp.attrs["proposal_id"] = spec["proposal_id"]
            grp.attrs["target"] = spec["target"]
            grp.attrs["wave_min"] = spec["wave_min"]
            grp.attrs["wave_max"] = spec["wave_max"]
            grp.attrs["n_pixels"] = spec["n_pixels"]
            grp.attrs["wavelength_unit"] = "Angstrom"
            grp.attrs["flux_unit"] = "erg/s/cm2/A"

    print(f"  Saved {len(spectra)} spectra to {outpath}")


def save_spectra_to_fits(spectra, outpath):
    """Save extracted spectra to a multi-extension FITS file (legacy format).

    Parameters
    ----------
    spectra : list of dict
        Spectrum records from read_x1d_spectrum.
    outpath : str
        Output FITS file path.
    """
    hdul = fits.HDUList([fits.PrimaryHDU()])

    for i, spec in enumerate(spectra):
        col_wave = fits.Column(name="WAVELENGTH", format="D",
                               unit="Angstrom", array=spec["wavelength"])
        col_flux = fits.Column(name="FLUX", format="D",
                               unit="erg/s/cm2/A", array=spec["flux"])
        col_err = fits.Column(name="FLUX_ERR", format="D",
                              unit="erg/s/cm2/A", array=spec["flux_err"])

        table = fits.BinTableHDU.from_columns([col_wave, col_flux, col_err])
        table.header["EXTNAME"] = f"SPEC_{i:03d}"
        table.header["FILENAME"] = spec["filename"]
        table.header["MJD"] = spec["mjd"] if np.isfinite(spec["mjd"]) else 0.0
        table.header["INSTRUME"] = spec["instrument"]
        table.header["GRATING"] = spec["grating"]
        table.header["EXPTIME"] = spec["exptime_s"]
        table.header["PROPOSID"] = spec["proposal_id"]
        table.header["TARGNAME"] = spec["target"]
        table.header["WAVEMIN"] = spec["wave_min"]
        table.header["WAVEMAX"] = spec["wave_max"]

        hdul.append(table)

    hdul.writeto(outpath, overwrite=True)
    print(f"  Saved {len(spectra)} spectra to {outpath}")


def save_spectra_to_csv(spectra, outdir, prefix="spectrum"):
    """Save each spectrum as a separate CSV file + a summary table.

    Parameters
    ----------
    spectra : list of dict
        Spectrum records.
    outdir : str
        Output directory.
    prefix : str
        Filename prefix.
    """
    os.makedirs(outdir, exist_ok=True)

    summary_records = []
    for i, spec in enumerate(spectra):
        # Save individual spectrum
        spec_df = pd.DataFrame({
            "wavelength_A": spec["wavelength"],
            "flux_erg_s_cm2_A": spec["flux"],
            "flux_err_erg_s_cm2_A": spec["flux_err"],
        })
        spec_fname = f"{prefix}_{i:03d}_{spec['grating']}_MJD{spec['mjd']:.1f}.csv"
        spec_df.to_csv(os.path.join(outdir, spec_fname), index=False)

        # Summary record
        summary_records.append({
            "filename": spec["filename"],
            "spectrum_file": spec_fname,
            "mjd": spec["mjd"],
            "instrument": spec["instrument"],
            "grating": spec["grating"],
            "exptime_s": spec["exptime_s"],
            "proposal_id": spec["proposal_id"],
            "wave_min_A": spec["wave_min"],
            "wave_max_A": spec["wave_max"],
            "n_pixels": spec["n_pixels"],
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(outdir, f"{prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary to {summary_path}")


def plot_spectra(spectra, mjd_ref=None, title="HST Spectra", outpath=None):
    """Plot all spectra, color-coded by epoch.

    Parameters
    ----------
    spectra : list of dict
        Spectrum records.
    mjd_ref : float, optional
        Reference MJD for computing delta_t labels.
    title : str
        Plot title.
    outpath : str, optional
        Output path. If None, displays interactively.
    """
    if not spectra:
        print("  No spectra to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    # Color by epoch
    mjds = [s["mjd"] for s in spectra]
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(mjds), max(mjds)) if len(set(mjds)) > 1 else None

    for spec in spectra:
        color = cmap(norm(spec["mjd"])) if norm else "C0"

        if mjd_ref:
            dt = spec["mjd"] - mjd_ref
            label = f"{spec['grating']} dt={dt:+.1f}d"
        else:
            label = f"{spec['grating']} MJD={spec['mjd']:.1f}"

        # Smooth slightly for display
        wave = spec["wavelength"]
        flux = spec["flux"]

        # Bin to ~5A resolution for cleaner plotting
        if len(wave) > 500:
            bin_size = max(1, len(wave) // 500)
            n_bins = len(wave) // bin_size
            wave = wave[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
            flux = flux[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)

        ax.plot(wave, flux, color=color, alpha=0.8, lw=0.8, label=label)

    ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=11)
    ax.set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.savefig(outpath.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath}")
    else:
        plt.show()


def plot_spectral_sequence(spectra, mjd_ref, title="Spectral Sequence",
                           outpath=None):
    """Plot spectra offset vertically by epoch — classic SN spectral sequence.

    Parameters
    ----------
    spectra : list of dict
        Spectrum records, sorted by MJD.
    mjd_ref : float
        Reference MJD (discovery/explosion).
    title : str
        Plot title.
    outpath : str
        Output path.
    """
    if not spectra:
        return

    spectra_sorted = sorted(spectra, key=lambda s: s["mjd"])

    fig, ax = plt.subplots(figsize=(7, max(3, len(spectra_sorted) * 0.8)))

    offset = 0
    for spec in spectra_sorted:
        dt = spec["mjd"] - mjd_ref
        wave = spec["wavelength"]
        flux = spec["flux"]

        # Normalize flux for display
        good = np.isfinite(flux) & (flux > 0)
        if np.sum(good) < 10:
            continue
        flux_norm = flux / np.median(flux[good])

        # Bin for display
        if len(wave) > 500:
            bin_size = max(1, len(wave) // 500)
            n_bins = len(wave) // bin_size
            wave = wave[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
            flux_norm = flux_norm[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)

        ax.plot(wave, flux_norm + offset, lw=0.8, alpha=0.9)
        ax.text(wave[-1] + 50, offset + 1.0,
                f"+{dt:.1f}d ({spec['grating']})",
                fontsize=7, va="center")

        offset += 2.0

    ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=11)
    ax.set_ylabel("Normalized Flux + offset", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.savefig(outpath.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath}")


# -----------------------------------------------------------------------
# Demo: AT2017gfo (GW170817)
# -----------------------------------------------------------------------

# AT2017gfo coordinates and reference epoch
AT2017GFO_RA = 197.45037
AT2017GFO_DEC = -23.38148
AT2017GFO_COORD = SkyCoord(ra=AT2017GFO_RA, dec=AT2017GFO_DEC, unit="deg")
MJD_GW170817 = 57982.529  # 2017-08-17 12:41:04 UTC


def demo_at2017gfo():
    """Demonstrate spectral extraction for AT2017gfo."""
    print("=" * 60)
    print("THATCH-SPECTRA Demo: AT2017gfo (GW170817)")
    print("=" * 60)

    specdir = os.path.join(DATADIR, "AT2017gfo", "spectra")
    os.makedirs(specdir, exist_ok=True)

    # Step 1: Query for spectroscopic observations
    print("\n--- Step 1: Query spectroscopic observations ---")
    try:
        obs_df = query_spectroscopic_obs(
            AT2017GFO_COORD, radius=5.0, name="AT2017gfo"
        )
        if len(obs_df) > 0:
            obs_df["delta_t_days"] = obs_df["mjd"] - MJD_GW170817
            log_path = os.path.join(specdir, "spectroscopic_obs_log.csv")
            obs_df.to_csv(log_path, index=False)
            print(f"  Saved observation log to {log_path}")
    except Exception as e:
        print(f"  MAST query failed: {e}")
        print("  Proceeding with local files...")
        obs_df = pd.DataFrame()

    # Step 2: Download extracted spectra
    if len(obs_df) > 0:
        print("\n--- Step 2: Download pipeline-extracted spectra ---")
        try:
            downloaded = download_extracted_spectra(obs_df, specdir)
        except Exception as e:
            print(f"  Download failed: {e}")
            downloaded = []
    else:
        downloaded = []

    # Step 3: Process whatever spectra we have locally
    print("\n--- Step 3: Process extracted spectra ---")
    spectra = process_spectral_directory(specdir)

    if not spectra:
        print("\n  No spectra available locally. To populate:")
        print("  1. Wait for MAST to be available, then re-run")
        print("  2. Or manually download x1d files from MAST to:")
        print(f"     {specdir}")
        return None

    # Step 4: Save standardized output
    print("\n--- Step 4: Save standardized spectra ---")
    save_spectra_to_csv(spectra, specdir, prefix="AT2017gfo")
    save_spectra_to_fits(
        spectra,
        os.path.join(specdir, "AT2017gfo_spectra.fits")
    )

    # Step 5: Plot
    print("\n--- Step 5: Plot spectra ---")
    plot_spectra(
        spectra,
        mjd_ref=MJD_GW170817,
        title="AT2017gfo — HST Spectra (THATCH)",
        outpath=os.path.join(FIGDIR, "AT2017gfo_spectra.pdf"),
    )
    plot_spectral_sequence(
        spectra,
        mjd_ref=MJD_GW170817,
        title="AT2017gfo — HST Spectral Sequence",
        outpath=os.path.join(FIGDIR, "AT2017gfo_spectral_sequence.pdf"),
    )

    # Summary
    print("\n--- Summary ---")
    print(f"  Spectra extracted: {len(spectra)}")
    for s in spectra:
        dt = s["mjd"] - MJD_GW170817
        print(f"    {s['grating']:8s} dt={dt:+7.1f}d  "
              f"{s['wave_min']:.0f}-{s['wave_max']:.0f}A  "
              f"{s['instrument']}")

    return spectra


# -----------------------------------------------------------------------
# Demo: SN 2011fe
# -----------------------------------------------------------------------

SN2011FE_RA = 210.77420
SN2011FE_DEC = 54.27370
SN2011FE_COORD = SkyCoord(ra=SN2011FE_RA, dec=SN2011FE_DEC, unit="deg")
MJD_BMAX_2011FE = 55814.5


def demo_sn2011fe():
    """Demonstrate spectral extraction for SN 2011fe."""
    print("\n" + "=" * 60)
    print("THATCH-SPECTRA Demo: SN 2011fe")
    print("=" * 60)

    specdir = os.path.join(DATADIR, "SN2011fe", "spectra")
    os.makedirs(specdir, exist_ok=True)

    # Step 1: Query
    print("\n--- Step 1: Query spectroscopic observations ---")
    try:
        obs_df = query_spectroscopic_obs(
            SN2011FE_COORD, radius=3.0, name="SN 2011fe"
        )
        if len(obs_df) > 0:
            obs_df["delta_t_days"] = obs_df["mjd"] - MJD_BMAX_2011FE
            log_path = os.path.join(specdir, "spectroscopic_obs_log.csv")
            obs_df.to_csv(log_path, index=False)
            print(f"  Saved observation log to {log_path}")
    except Exception as e:
        print(f"  MAST query failed: {e}")
        obs_df = pd.DataFrame()

    # Step 2: Download
    if len(obs_df) > 0:
        print("\n--- Step 2: Download pipeline-extracted spectra ---")
        try:
            downloaded = download_extracted_spectra(obs_df, specdir)
        except Exception as e:
            print(f"  Download failed: {e}")

    # Step 3: Process
    print("\n--- Step 3: Process extracted spectra ---")
    spectra = process_spectral_directory(specdir)

    if not spectra:
        print("  No spectra available locally.")
        return None

    # Step 4: Save
    print("\n--- Step 4: Save standardized spectra ---")
    save_spectra_to_csv(spectra, specdir, prefix="SN2011fe")
    save_spectra_to_fits(
        spectra,
        os.path.join(specdir, "SN2011fe_spectra.fits")
    )

    # Step 5: Plot
    print("\n--- Step 5: Plot spectra ---")
    plot_spectra(
        spectra,
        mjd_ref=MJD_BMAX_2011FE,
        title="SN 2011fe — HST/STIS Spectra (THATCH)",
        outpath=os.path.join(FIGDIR, "SN2011fe_spectra.pdf"),
    )
    plot_spectral_sequence(
        spectra,
        mjd_ref=MJD_BMAX_2011FE,
        title="SN 2011fe — HST Spectral Sequence",
        outpath=os.path.join(FIGDIR, "SN2011fe_spectral_sequence.pdf"),
    )

    return spectra


def main():
    spectra_gfo = demo_at2017gfo()
    spectra_11fe = demo_sn2011fe()


if __name__ == "__main__":
    main()
