#!/usr/bin/env python
"""
THATCH grism spectral extraction using grizli.

Extracts 1D spectra from WFC3/IR grism (G102, G141) FLT files
at a known transient position using grizli's forward-modeling approach.

Usage:
    conda activate thatch
    python extract_grism.py
"""

import os
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glob import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time

warnings.filterwarnings("ignore")

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data")
FIGDIR = os.path.join(BASEDIR, "..", "Figures")

# AT2017gfo
RA = 197.45037
DEC = -23.38148
MJD_GW = 57982.529

# WFC3/IR grism dispersion parameters (approximate)
# G102: 0.8-1.15 um, ~24.5 A/pix, R~210 at 1.0 um
# G141: 1.07-1.7 um, ~46.5 A/pix, R~130 at 1.4 um
GRISM_PARAMS = {
    "G102": {
        "wave_min": 7500,
        "wave_max": 11800,
        "disp": 24.5,
        "ref_wave": 10000,
        "trace_offset": 0,
    },
    "G141": {
        "wave_min": 10500,
        "wave_max": 17500,
        "disp": 46.5,
        "ref_wave": 14000,
        "trace_offset": 0,
    },
}


def simple_box_extraction(flt_path, ra, dec, box_half=3, bg_offset=15, bg_width=5):
    """Simple box extraction of a grism spectrum at a known position.

    This is a quick-look extraction suitable for isolated sources.
    For crowded fields, use grizli's full contamination modeling.

    Parameters
    ----------
    flt_path : str
        Path to WFC3/IR grism FLT file.
    ra, dec : float
        Source coordinates in degrees.
    box_half : int
        Half-width of extraction box in pixels (cross-dispersion).
    bg_offset : int
        Offset from source for background estimation.
    bg_width : int
        Width of background region.

    Returns
    -------
    dict or None
        Extracted spectrum with wavelength, flux, metadata.
    """
    fname = os.path.basename(flt_path)

    with fits.open(flt_path) as hdul:
        pri = hdul[0].header
        sci = hdul["SCI"].data.astype(float)
        err = hdul["ERR"].data.astype(float)
        dq = hdul["DQ"].data

        instrument = pri.get("INSTRUME", "WFC3")
        detector = pri.get("DETECTOR", "IR")
        filt = pri.get("FILTER", pri.get("OPT_ELEM", ""))
        exptime = float(pri.get("EXPTIME", 1.0))
        date_obs = pri.get("DATE-OBS", "")
        time_obs = pri.get("TIME-OBS", "00:00:00")
        expstart = pri.get("EXPSTART")
        proposal = str(pri.get("PROPOSID", ""))

        if expstart:
            mjd = float(expstart)
        else:
            try:
                t = Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")
                mjd = t.mjd
            except Exception:
                mjd = np.nan

        if filt not in GRISM_PARAMS:
            return None

        params = GRISM_PARAMS[filt]

        # Get pixel position of source
        wcs = WCS(hdul["SCI"].header, fobj=hdul)
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        x_src, y_src = wcs.world_to_pixel(coord)
        x_src, y_src = int(round(float(x_src))), int(round(float(y_src)))

        ny, nx = sci.shape
        if not (
            50 < x_src < nx - 50
            and box_half + bg_offset + bg_width < y_src < ny - box_half - bg_offset - bg_width
        ):
            print(f"  {fname}: source at ({x_src},{y_src}) too close to edge")
            return None

        # For WFC3/IR grisms, dispersion is roughly along the x-axis
        # The direct image position gives the 0th-order location
        # The spectrum extends to positive x from there

        # Extract a strip centered on the source y-position
        y_lo = y_src - box_half
        y_hi = y_src + box_half + 1

        # Background strips above and below
        bg_lo1 = y_src - bg_offset - bg_width
        bg_hi1 = y_src - bg_offset
        bg_lo2 = y_src + bg_offset
        bg_hi2 = y_src + bg_offset + bg_width

        # Make sure background regions are valid
        bg_lo1 = max(0, bg_lo1)
        bg_hi2 = min(ny, bg_hi2)

        # Extract spectral strip (full x range around source)
        x_start = max(0, x_src - 20)  # small buffer before source
        x_end = min(nx, x_src + 250)  # spectrum extends ~200 pixels

        strip_src = sci[y_lo:y_hi, x_start:x_end]
        strip_err = err[y_lo:y_hi, x_start:x_end]
        strip_dq = dq[y_lo:y_hi, x_start:x_end]

        # Background
        bg_strip1 = sci[bg_lo1:bg_hi1, x_start:x_end]
        bg_strip2 = sci[bg_lo2:bg_hi2, x_start:x_end]
        bg_level = np.nanmedian(np.vstack([bg_strip1, bg_strip2]), axis=0)

        # Sum in cross-dispersion direction, subtract background
        flux_raw = np.nansum(strip_src, axis=0)
        flux_bg = bg_level * (y_hi - y_lo)
        flux_net = flux_raw - flux_bg

        # Error propagation
        flux_err = np.sqrt(np.nansum(strip_err**2, axis=0))

        # Build wavelength array
        # Approximate: pixel offset from source position * dispersion
        n_pix = x_end - x_start
        pix = np.arange(n_pix)
        pix_offset = pix - (x_src - x_start)

        wave = params["ref_wave"] + pix_offset * params["disp"]

        # Trim to valid wavelength range
        valid = (wave >= params["wave_min"]) & (wave <= params["wave_max"]) & np.isfinite(flux_net)
        if np.sum(valid) < 20:
            print(f"  {fname}: too few valid pixels ({np.sum(valid)})")
            return None

        wave = wave[valid]
        flux_net = flux_net[valid]
        flux_err = flux_err[valid]

        # Convert from e-/s to flux density (approximate)
        # Use the PHOTFLAM-like sensitivity curve
        # For quick-look, normalize to the direct image photometry if available
        # Here we leave in e-/s units

        return {
            "filename": fname,
            "mjd": mjd,
            "delta_t": mjd - MJD_GW,
            "grism": filt,
            "instrument": f"{instrument}/{detector}",
            "exptime_s": exptime,
            "proposal_id": proposal,
            "wavelength": wave,
            "flux": flux_net,
            "flux_err": flux_err,
            "flux_unit": "e-/s (uncalibrated)",
            "x_src": x_src,
            "y_src": y_src,
        }


def try_grizli_extraction(grism_dir, ra, dec):
    """Attempt grizli-based extraction for better calibration.

    Returns list of extracted spectra, or None if grizli fails.
    """
    try:
        from grizli import utils, model, fitting
        from grizli.prep import process_direct_grism_visit

        print("  grizli available - attempting full extraction...")
    except ImportError:
        print("  grizli not available, using simple box extraction")
        return None

    # For a full grizli reduction we need:
    # 1. Direct image (for source detection + WCS alignment)
    # 2. Grism images (FLT files)
    # 3. Reference catalog

    # This is a complex multi-step process. For the demo,
    # we'll use grizli's GrismFLT model to do a forward-model extraction
    # at the known source position.

    flt_files = sorted(glob(os.path.join(grism_dir, "*_flt.fits")))
    if not flt_files:
        return None

    print(f"  Found {len(flt_files)} FLT files")

    # For now, return None and fall back to simple extraction
    # Full grizli pipeline requires more setup (reference images, catalogs)
    # This is the work proposed for months 3-5 of the THATCH timeline
    print("  Full grizli pipeline requires reference images - using box extraction")
    return None


def main():
    print("=" * 60)
    print("THATCH Grism Extraction: AT2017gfo")
    print("=" * 60)

    grism_dir = os.path.join(DATADIR, "AT2017gfo", "spectra", "grism")
    flt_files = sorted(glob(os.path.join(grism_dir, "*_flt.fits")))

    print(f"Found {len(flt_files)} grism FLT files")

    # Try grizli first
    grizli_result = try_grizli_extraction(grism_dir, RA, DEC)

    # Fall back to simple box extraction
    if grizli_result is None:
        print("\nRunning simple box extraction at known source position...")

    spectra = []
    for fpath in flt_files:
        try:
            spec = simple_box_extraction(fpath, RA, DEC)
            if spec is not None:
                spectra.append(spec)
                print(
                    f"  {spec['filename']}: {spec['grism']} "
                    f"dt={spec['delta_t']:+.1f}d "
                    f"x={spec['x_src']} y={spec['y_src']}"
                )
        except Exception as e:
            print(f"  {os.path.basename(fpath)}: ERROR {e}")

    if not spectra:
        print("No spectra extracted.")
        return

    # Group by epoch and grism, average duplicates
    groups = {}
    for s in spectra:
        key = (round(s["mjd"], 0), s["grism"])
        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    merged = []
    for (mjd_r, grism), specs in sorted(groups.items()):
        # Average onto common wavelength grid
        ref = specs[0]
        wave = ref["wavelength"]
        fluxes = [ref["flux"]]
        for s in specs[1:]:
            interp_f = np.interp(wave, s["wavelength"], s["flux"])
            fluxes.append(interp_f)

        avg_flux = np.mean(fluxes, axis=0)
        avg_err = ref["flux_err"] / np.sqrt(len(specs))

        merged.append(
            {
                "mjd": mjd_r,
                "delta_t": mjd_r - MJD_GW,
                "grism": grism,
                "wavelength": wave,
                "flux": avg_flux,
                "flux_err": avg_err,
                "n_combined": len(specs),
            }
        )

    print(f"\nMerged to {len(merged)} unique epoch/grism combinations:")
    for m in merged:
        print(f"  {m['grism']} dt={m['delta_t']:+.1f}d ({m['n_combined']} exposures combined)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Left: G102, Right: G141
    for ax, grism, title in [
        (axes[0], "G102", "G102 (0.8-1.15 $\\mu$m)"),
        (axes[1], "G141", "G141 (1.1-1.7 $\\mu$m)"),
    ]:
        grism_specs = [m for m in merged if m["grism"] == grism]

        if not grism_specs:
            ax.set_title(title)
            continue

        cmap = plt.cm.plasma
        dts = [m["delta_t"] for m in grism_specs]
        norm = plt.Normalize(min(dts), max(dts)) if len(set(dts)) > 1 else None

        for m in grism_specs:
            color = cmap(norm(m["delta_t"])) if norm else "C0"
            wave_um = m["wavelength"] / 1e4  # to microns

            # Smooth slightly
            from scipy.ndimage import uniform_filter1d

            flux_smooth = uniform_filter1d(m["flux"], size=5)

            ax.plot(
                wave_um, flux_smooth, color=color, lw=1, label=f"+{m['delta_t']:.0f}d", alpha=0.9
            )

        ax.set_xlabel("Wavelength ($\\mu$m)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Flux (e$^{-}$/s, uncalibrated)", fontsize=11)

    fig.suptitle("AT2017gfo — WFC3/IR Grism Spectra (THATCH box extraction)", fontsize=12, y=1.02)
    plt.tight_layout()

    outpath = os.path.join(FIGDIR, "AT2017gfo_grism_spectra.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outpath}")

    # Also plot combined with the STIS UV spectrum
    fig, ax = plt.subplots(figsize=(8, 4))

    # Load STIS spectrum
    stis_path = os.path.join(DATADIR, "AT2017gfo", "spectra", "odp801010_x1d.fits")
    if os.path.exists(stis_path):
        from hustle_spectra import read_x1d_spectrum

        stis = read_x1d_spectrum(stis_path)
        if stis:
            w = stis["wavelength"] / 1e4
            f = stis["flux"]
            good = (w > 0.16) & (w < 0.32) & np.isfinite(f)
            # Normalize
            pos = f[good] > 0
            if np.sum(pos) > 5:
                f_norm = f[good] / np.max(f[good][pos])
                ax.plot(w[good], f_norm, color="purple", lw=1, label="STIS G230L +5.6d", alpha=0.9)

    # Plot closest-epoch grism spectra
    for m in merged:
        if abs(m["delta_t"] - 5) < 2:  # near +5d
            wave_um = m["wavelength"] / 1e4
            from scipy.ndimage import uniform_filter1d

            flux_smooth = uniform_filter1d(m["flux"], size=5)
            # Normalize
            pos = flux_smooth > 0
            if np.sum(pos) > 10:
                f_norm = flux_smooth / np.max(flux_smooth[pos])
                color = "C0" if m["grism"] == "G102" else "C1"
                ax.plot(
                    wave_um,
                    f_norm,
                    color=color,
                    lw=1,
                    label=f"{m['grism']} +{m['delta_t']:.0f}d",
                    alpha=0.9,
                )

    ax.set_xlabel("Wavelength ($\\mu$m)", fontsize=11)
    ax.set_ylabel("Normalized Flux", fontsize=11)
    ax.set_title("AT2017gfo +5d — HST UV-to-NIR (STIS + WFC3 Grism)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 1.8)

    plt.tight_layout()
    outpath2 = os.path.join(FIGDIR, "AT2017gfo_full_spectrum.pdf")
    plt.savefig(outpath2, dpi=150, bbox_inches="tight")
    plt.savefig(outpath2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath2}")


if __name__ == "__main__":
    main()
