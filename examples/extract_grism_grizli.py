#!/usr/bin/env python
"""
THATCH grism spectral extraction using grizli's forward-modeling pipeline.

For each grism FLT file, pairs it with the contemporaneous direct image,
builds a contamination model, and extracts a flux-calibrated 1D spectrum
at the known transient position.

Usage:
    conda activate thatch
    python extract_grism_grizli.py
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glob import glob
from collections import defaultdict
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS

warnings.filterwarnings("ignore")

# Set grizli config path
os.environ.setdefault("GRIZLI", os.path.expanduser("~/grizli"))

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data")
FIGDIR = os.path.join(BASEDIR, "..", "Figures")

# AT2017gfo
RA = 197.45037
DEC = -23.38148
MJD_GW = 57982.529


def identify_visit_pairs(flt_dir):
    """Group FLT files into direct+grism visit pairs by visit ID.

    Returns dict of {visit_key: {"direct": [files], "grism": [files], ...}}
    """
    flt_files = sorted(glob(os.path.join(flt_dir, "*_flt.fits")))
    # Exclude HAP reprocessed files (start with hst_)
    flt_files = [f for f in flt_files if not os.path.basename(f).startswith("hst_")]

    visits = defaultdict(lambda: {"direct": [], "grism": [], "grism_filter": None,
                                   "direct_filter": None, "mjd": None, "program": None})

    for fpath in flt_files:
        h = fits.getheader(fpath, 0)
        filt = h.get("FILTER", "")
        rootname = os.path.basename(fpath).replace("_flt.fits", "")
        # Visit ID: first 6 chars of rootname (e.g., idp7g1)
        visit_id = rootname[:6]
        expstart = h.get("EXPSTART", 0)
        program = str(h.get("PROPOSID", ""))

        if filt in ["G102", "G141"]:
            visits[visit_id]["grism"].append(fpath)
            visits[visit_id]["grism_filter"] = filt
        else:
            visits[visit_id]["direct"].append(fpath)
            visits[visit_id]["direct_filter"] = filt

        visits[visit_id]["mjd"] = expstart
        visits[visit_id]["program"] = program

    # Pair grism visits with direct visits from the same program/epoch
    # Visit naming: idp7g1 (direct F110W) pairs with idp7g1 (grism G102)
    # But actually the visit IDs are different for direct vs grism
    # Let's pair by program + epoch proximity

    paired = []
    grism_visits = {k: v for k, v in visits.items() if v["grism"]}
    direct_visits = {k: v for k, v in visits.items() if v["direct"]}

    for gk, gv in grism_visits.items():
        # Find closest direct visit from same program
        best_dk = None
        best_dt = 999
        for dk, dv in direct_visits.items():
            if dv["program"] == gv["program"]:
                dt = abs(dv["mjd"] - gv["mjd"])
                if dt < best_dt:
                    best_dt = dt
                    best_dk = dk

        if best_dk and best_dt < 1.0:  # within 1 day
            paired.append({
                "grism_visit": gk,
                "direct_visit": best_dk,
                "grism_files": gv["grism"],
                "direct_files": direct_visits[best_dk]["direct"],
                "grism_filter": gv["grism_filter"],
                "direct_filter": direct_visits[best_dk]["direct_filter"],
                "mjd": gv["mjd"],
                "program": gv["program"],
            })

    return paired


def extract_with_grizli(grism_file, direct_file, ra, dec, seg_id=1):
    """Extract a grism spectrum using grizli's GrismFLT model.

    Parameters
    ----------
    grism_file : str
        Path to grism FLT file.
    direct_file : str
        Path to paired direct image FLT file.
    ra, dec : float
        Source coordinates.
    seg_id : int
        Segmentation map ID for the source.

    Returns
    -------
    dict or None
        Extracted spectrum.
    """
    from grizli import model as grizli_model
    from grizli import utils as grizli_utils

    fname = os.path.basename(grism_file)

    try:
        # Create a simple segmentation map with just our source
        with fits.open(direct_file) as hdul:
            sci_ext = None
            for i, hdu in enumerate(hdul):
                if hdu.name == "SCI":
                    sci_ext = i
                    break
            if sci_ext is None:
                sci_ext = 1

            direct_data = hdul[sci_ext].data
            direct_header = hdul[sci_ext].header
            direct_wcs = WCS(direct_header, fobj=hdul)

        # Get pixel position of source
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        x_src, y_src = direct_wcs.world_to_pixel(coord)
        x_src, y_src = int(round(float(x_src))), int(round(float(y_src)))

        ny, nx = direct_data.shape

        # Create segmentation map: circle around source
        seg = np.zeros((ny, nx), dtype=int)
        yy, xx = np.mgrid[:ny, :nx]
        dist = np.sqrt((xx - x_src)**2 + (yy - y_src)**2)
        seg[dist <= 8] = seg_id  # 8-pixel radius

        # Create a simple catalog
        cat = Table()
        cat["NUMBER"] = [seg_id]
        cat["X_WORLD"] = [ra]
        cat["Y_WORLD"] = [dec]
        cat["X_IMAGE"] = [x_src + 1]  # 1-indexed
        cat["Y_IMAGE"] = [y_src + 1]
        cat["MAG_AUTO"] = [24.0]  # approximate
        cat["FLUX_RADIUS"] = [5.0]
        cat["A_IMAGE"] = [3.0]
        cat["B_IMAGE"] = [3.0]
        cat["THETA_IMAGE"] = [0.0]

        # Write temp segmentation and catalog
        tmpdir = os.path.dirname(grism_file)
        seg_file = os.path.join(tmpdir, "temp_seg.fits")
        cat_file = os.path.join(tmpdir, "temp_cat.fits")

        fits.writeto(seg_file, seg, direct_header, overwrite=True)
        cat.write(cat_file, format="fits", overwrite=True)

        # Initialize GrismFLT
        # Determine grism from header
        grism_header = fits.getheader(grism_file, 0)
        grism_name = grism_header.get("FILTER", "G141")

        grism_flt = grizli_model.GrismFLT(
            grism_file=grism_file,
            direct_file=direct_file,
            seg_file=seg_file,
            ref_file=None,
            ref_ext=0,
            shrink_segimage=False,
            force_grism=grism_name,
            verbose=False,
        )

        # Compute model for our source
        grism_flt.compute_model_orders(id=seg_id, mag=24.0,
                                        compute_size=True, is_cgs=False,
                                        verbose=False)

        # Get disperser orders — structure is (status, spectrum, OrderedDict)
        disp = grism_flt.object_dispersers.get(seg_id)
        if disp is None:
            print(f"  {fname}: no disperser computed")
            return None

        orders = disp[2]  # OrderedDict of spectral orders
        if "A" not in orders:
            print(f"  {fname}: no first-order spectrum")
            return None

        # Create BeamCutout for first order
        beam_obj = orders["A"]
        beam = grizli_model.BeamCutout(flt=grism_flt, beam=beam_obj,
                                        conf=grism_flt.conf)

        # Optimal 1D extraction
        wave_1d = beam.beam.lam
        sens = beam.beam.sensitivity
        flux_2d = beam.grism.data["SCI"] - beam.contam
        err_2d = beam.grism.data["ERR"]

        model_2d = beam.beam.model
        if model_2d.sum() > 0:
            weights = model_2d / np.maximum(model_2d.sum(axis=0, keepdims=True), 1e-30)
            weights = np.nan_to_num(weights, 0)
            flux_1d = np.sum(flux_2d * weights, axis=0)
            var_1d = np.sum((err_2d * weights)**2, axis=0)
            err_1d = np.sqrt(np.maximum(var_1d, 0))
        else:
            flux_1d = np.sum(flux_2d, axis=0)
            err_1d = np.sqrt(np.sum(err_2d**2, axis=0))

        # Flux calibrate: divide by sensitivity
        # Require sensitivity > 10% of peak to avoid edge artifacts
        sens_thresh = 0.1 * np.max(sens)
        good = ((sens > sens_thresh) & (wave_1d > 0) &
                np.isfinite(flux_1d) & np.isfinite(err_1d))
        if np.sum(good) < 10:
            print(f"  {fname}: too few good pixels after calibration")
            return None

        flux_cal = np.zeros_like(flux_1d)
        err_cal = np.zeros_like(err_1d)
        flux_cal[good] = flux_1d[good] / sens[good]
        err_cal[good] = err_1d[good] / sens[good]

        # Get metadata
        h = fits.getheader(grism_file, 0)
        grism_filter = h.get("FILTER", "")
        expstart = h.get("EXPSTART", np.nan)

        # Clean up temp files
        for tf in [seg_file, cat_file]:
            if os.path.exists(tf):
                os.remove(tf)

        return {
            "filename": fname,
            "mjd": float(expstart),
            "delta_t": float(expstart) - MJD_GW,
            "grism": grism_filter,
            "wavelength": wave_1d[good],  # Angstroms
            "flux": flux_cal[good],  # erg/s/cm^2/A (flux-calibrated)
            "flux_err": err_cal[good],
            "flux_unit": "erg/s/cm2/A",
        }

    except Exception as e:
        print(f"  {fname}: grizli extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("THATCH Grism Extraction (grizli): AT2017gfo")
    print("=" * 60)

    flt_dir = os.path.join(DATADIR, "AT2017gfo", "spectra", "grism")

    # Identify visit pairs
    print("\nIdentifying direct+grism visit pairs...")
    pairs = identify_visit_pairs(flt_dir)
    print(f"Found {len(pairs)} grism visits with direct image pairs:")
    for p in pairs:
        print(f"  {p['grism_filter']:5s} MJD={p['mjd']:.3f} "
              f"dt={p['mjd']-MJD_GW:+.1f}d "
              f"prog={p['program']} "
              f"({len(p['grism_files'])}G + {len(p['direct_files'])}D)")

    # Extract spectra
    print("\nExtracting spectra with grizli...")
    all_spectra = []

    for pair in pairs:
        grism_filter = pair["grism_filter"]
        dt = pair["mjd"] - MJD_GW

        # Use first direct file as reference
        direct_file = pair["direct_files"][0]

        for grism_file in pair["grism_files"]:
            print(f"\n  Processing {os.path.basename(grism_file)} "
                  f"({grism_filter}, dt={dt:+.1f}d)...")
            spec = extract_with_grizli(grism_file, direct_file, RA, DEC)
            if spec is not None:
                all_spectra.append(spec)
                print(f"    OK: {len(spec['wavelength'])} pixels, "
                      f"{spec['wavelength'].min():.0f}-{spec['wavelength'].max():.0f} A")

    if not all_spectra:
        print("\nNo spectra extracted. Falling back to box extraction.")
        return

    # Group by epoch and grism, average
    groups = defaultdict(list)
    for s in all_spectra:
        key = (round(s["mjd"], 0), s["grism"])
        groups[key].append(s)

    merged = []
    for (mjd_r, grism), specs in sorted(groups.items()):
        ref = specs[0]
        wave = ref["wavelength"]
        fluxes = [ref["flux"]]
        for s in specs[1:]:
            interp_f = np.interp(wave, s["wavelength"], s["flux"])
            fluxes.append(interp_f)

        avg_flux = np.mean(fluxes, axis=0)
        avg_err = ref["flux_err"] / np.sqrt(len(specs))

        merged.append({
            "mjd": mjd_r,
            "delta_t": mjd_r - MJD_GW,
            "grism": grism,
            "wavelength": wave,
            "flux": avg_flux,
            "flux_err": avg_err,
            "n_combined": len(specs),
        })

    print(f"\n{'='*60}")
    print(f"Merged to {len(merged)} unique epoch/grism spectra:")
    for m in merged:
        print(f"  {m['grism']} dt={m['delta_t']:+.1f}d "
              f"({m['n_combined']} exposures) "
              f"{m['wavelength'].min():.0f}-{m['wavelength'].max():.0f} A")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, grism, title in [(axes[0], "G102", r"G102 (0.8-1.15 $\mu$m)"),
                              (axes[1], "G141", r"G141 (1.1-1.7 $\mu$m)")]:
        grism_specs = [m for m in merged if m["grism"] == grism]
        if not grism_specs:
            ax.set_title(title)
            continue

        cmap = plt.cm.plasma
        dts = [m["delta_t"] for m in grism_specs]
        norm_c = plt.Normalize(min(dts), max(dts)) if len(set(dts)) > 1 else None

        for m in grism_specs:
            color = cmap(norm_c(m["delta_t"])) if norm_c else "C0"
            wave_um = m["wavelength"] / 1e4

            from scipy.ndimage import uniform_filter1d
            flux_smooth = uniform_filter1d(m["flux"], size=3)

            ax.plot(wave_um, flux_smooth, color=color, lw=1,
                    label=f"+{m['delta_t']:.0f}d", alpha=0.9)

        ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"F$_\lambda$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)", fontsize=11)

    fig.suptitle("AT2017gfo — WFC3/IR Grism Spectra (grizli extraction)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    outpath = os.path.join(FIGDIR, "AT2017gfo_grism_grizli.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outpath}")

    # Combined UV-NIR plot with X-shooter
    fig, ax = plt.subplots(figsize=(8, 4))

    # Load STIS
    stis_path = os.path.join(DATADIR, "AT2017gfo", "spectra", "odp801010_x1d.fits")
    if os.path.exists(stis_path):
        sys.path.insert(0, BASEDIR)
        from hustle_spectra import read_x1d_spectrum
        stis = read_x1d_spectrum(stis_path)
        if stis:
            w = stis["wavelength"]
            f = stis["flux"]
            good = (w > 1600) & (w < 3100) & np.isfinite(f)
            ax.plot(w[good] / 1e4, f[good], color="purple", lw=1,
                    label="STIS G230L +5.6d", alpha=0.9)

    # Plot grism spectra near +5d
    for m in merged:
        if abs(m["delta_t"] - 5) < 3:
            from scipy.ndimage import uniform_filter1d
            flux_smooth = uniform_filter1d(m["flux"], size=3)
            color = "C0" if m["grism"] == "G102" else "C1"
            ax.plot(m["wavelength"] / 1e4, flux_smooth, color=color, lw=1,
                    label=f"{m['grism']} +{m['delta_t']:.0f}d", alpha=0.9)

    # Load closest X-shooter
    xsh_dir = os.path.join(DATADIR, "AT2017gfo", "spectra", "xshooter")
    xsh_file = os.path.join(xsh_dir, "XSGW0823_smooth.dat")
    if os.path.exists(xsh_file):
        data = np.loadtxt(xsh_file)
        w, f = data[:, 0], data[:, 1]
        good = (w > 3500) & (w < 24000) & (f > 0)
        # Bin
        w_g, f_g = w[good], f[good]
        bs = max(1, len(w_g) // 500)
        n = len(w_g) // bs
        w_b = w_g[:n*bs].reshape(n, bs).mean(axis=1)
        f_b = f_g[:n*bs].reshape(n, bs).mean(axis=1)
        ax.plot(w_b / 1e4, f_b, color="gray", lw=0.7, alpha=0.5,
                label="X-shooter +6d (ground)")

    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=11)
    ax.set_ylabel(r"F$_\lambda$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)", fontsize=11)
    ax.set_title("AT2017gfo — HST UV-to-NIR Spectrum (THATCH/grizli)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 1.8)

    plt.tight_layout()
    outpath2 = os.path.join(FIGDIR, "AT2017gfo_full_spectrum_grizli.pdf")
    plt.savefig(outpath2, dpi=150, bbox_inches="tight")
    plt.savefig(outpath2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath2}")


if __name__ == "__main__":
    main()
