#!/usr/bin/env python
"""
Plot spectral sequences for AT2017gfo (HST + X-shooter) and SN 2011fe (STIS).
Merges duplicate exposures at the same epoch/grating.

Usage:
    conda activate thatch
    python plot_spectral_sequences.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data")
FIGDIR = os.path.join(BASEDIR, "..", "Figures")

MJD_GW = 57982.529  # GW170817
MJD_BMAX = 55814.5  # SN 2011fe B-max


def load_xshooter_spectra(specdir):
    """Load X-shooter spectra from .dat files."""
    spectra = []
    # Map filenames to dates (MJD of Aug 17 = 57982.529)
    date_map = {
        "XSGW0818": 57983.5, "G298048_XSH_20170819": 57984.5,
        "XSGW0820": 57985.5, "G298048_XSH_20170821": 57986.5,
        "XSGW0822": 57987.5, "XSGW0823": 57988.5,
        "XSGW0824": 57989.5, "XSGW0825": 57990.5,
        "XSGW0826": 57991.5, "XSGW0827": 57992.5,
    }

    for fpath in sorted(glob(os.path.join(specdir, "*.dat"))):
        fname = os.path.basename(fpath)
        key = fname.replace("_smooth.dat", "").replace(".dat", "")

        mjd = date_map.get(key, None)
        if mjd is None:
            continue

        try:
            data = np.loadtxt(fpath)
            wave = data[:, 0]
            flux = data[:, 1]
            good = (wave > 3000) & (wave < 25000) & np.isfinite(flux)
            if np.sum(good) < 50:
                continue
            spectra.append({
                "wave": wave[good], "flux": flux[good],
                "mjd": mjd, "source": "X-shooter",
                "label": f"XSH +{mjd - MJD_GW:.1f}d",
            })
        except Exception:
            continue

    return sorted(spectra, key=lambda s: s["mjd"])


def load_hst_stis_spectra(specdir):
    """Load STIS x1d/sx1 spectra and merge duplicates."""
    from astropy.io import fits

    raw = []
    for ext in ["*_x1d.fits", "*_sx1.fits"]:
        for fpath in sorted(glob(os.path.join(specdir, ext))):
            try:
                with fits.open(fpath) as hdul:
                    pri = hdul[0].header
                    ext1 = hdul[1].header if len(hdul) > 1 else {}
                    grating = pri.get("OPT_ELEM", "?")

                    expstart = ext1.get("EXPSTART") or pri.get("EXPSTART")
                    if expstart:
                        mjd = float(expstart)
                    else:
                        continue

                    data = hdul[1].data
                    if data is None:
                        continue

                    all_w, all_f = [], []
                    for row in data:
                        w, f = row["WAVELENGTH"], row["FLUX"]
                        good = (w > 0) & np.isfinite(f)
                        if np.sum(good) > 10:
                            all_w.append(w[good])
                            all_f.append(f[good])

                    if not all_w:
                        continue

                    wave = np.concatenate(all_w)
                    flux = np.concatenate(all_f)
                    idx = np.argsort(wave)

                    raw.append({
                        "wave": wave[idx], "flux": flux[idx],
                        "mjd": mjd, "grating": grating,
                    })
            except Exception:
                continue

    # Merge duplicates: group by rounded MJD + grating, average fluxes
    if not raw:
        return []

    groups = {}
    for s in raw:
        key = (round(s["mjd"], 1), s["grating"])
        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    merged = []
    for (mjd_r, grating), specs in sorted(groups.items()):
        if len(specs) == 1:
            merged.append(specs[0])
        else:
            # Average onto common wavelength grid
            ref = specs[0]
            wave = ref["wave"]
            fluxes = [ref["flux"]]
            for s in specs[1:]:
                interp_flux = np.interp(wave, s["wave"], s["flux"])
                fluxes.append(interp_flux)
            avg_flux = np.mean(fluxes, axis=0)
            merged.append({
                "wave": wave, "flux": avg_flux,
                "mjd": mjd_r, "grating": grating,
            })

    return sorted(merged, key=lambda s: (s["mjd"], s["grating"]))


def plot_at2017gfo_sequence():
    """Plot AT2017gfo spectral sequence: X-shooter + HST/STIS UV."""
    print("Plotting AT2017gfo spectral sequence...")

    specdir_xsh = os.path.join(DATADIR, "AT2017gfo", "spectra", "xshooter")
    specdir_hst = os.path.join(DATADIR, "AT2017gfo", "spectra")

    xsh_spectra = load_xshooter_spectra(specdir_xsh)
    hst_spectra = load_hst_stis_spectra(specdir_hst)

    print(f"  X-shooter spectra: {len(xsh_spectra)}")
    print(f"  HST spectra: {len(hst_spectra)}")

    if not xsh_spectra and not hst_spectra:
        print("  No spectra found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.RdYlBu_r
    all_dt = [s["mjd"] - MJD_GW for s in xsh_spectra]
    if all_dt:
        norm = plt.Normalize(min(all_dt), max(all_dt))
    else:
        norm = plt.Normalize(0, 10)

    offset = 0
    offset_step = 1.2

    # Plot X-shooter spectra
    for spec in xsh_spectra:
        dt = spec["mjd"] - MJD_GW
        color = cmap(norm(dt))
        wave = spec["wave"]
        flux = spec["flux"]

        # Normalize to peak
        good = (wave > 4000) & (wave < 20000) & (flux > 0)
        if np.sum(good) < 10:
            continue
        flux_norm = flux / np.max(flux[good])

        # Bin for cleaner display
        if len(wave) > 1000:
            bs = max(1, len(wave) // 800)
            n = len(wave) // bs
            wave = wave[:n*bs].reshape(n, bs).mean(axis=1)
            flux_norm = flux_norm[:n*bs].reshape(n, bs).mean(axis=1)

        ax.plot(wave, flux_norm + offset, color=color, lw=0.7, alpha=0.9)
        ax.text(wave[wave < 22000][-1] + 300, offset + 0.5,
                f"+{dt:.1f}d XSH", fontsize=6, va="center", color=color)
        offset += offset_step

    # Overlay HST/STIS UV spectrum
    for spec in hst_spectra:
        dt = spec["mjd"] - MJD_GW
        wave = spec["wave"]
        flux = spec["flux"]

        good = np.isfinite(flux) & (flux > 0) & (wave > 1600) & (wave < 3100)
        if np.sum(good) < 10:
            continue

        flux_norm = flux / np.max(flux[good])

        # Find the closest XSH epoch to overlay
        closest_offset = 0
        for i, xs in enumerate(xsh_spectra):
            if abs(xs["mjd"] - spec["mjd"]) < 1.0:
                closest_offset = i * offset_step
                break

        ax.plot(wave[good], flux_norm[good] + closest_offset,
                color="purple", lw=1.2, alpha=0.9)
        ax.text(1500, closest_offset + 0.5,
                f"+{dt:.1f}d HST/STIS", fontsize=6, va="center",
                color="purple", fontweight="bold")

    ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=11)
    ax.set_ylabel("Normalized Flux + offset", fontsize=11)
    ax.set_title("AT2017gfo — Spectral Sequence (X-shooter + HST/STIS UV)",
                 fontsize=11)
    ax.set_xlim(1400, 24000)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, "AT2017gfo_spectral_sequence.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


def plot_sn2011fe_sequence():
    """Plot SN 2011fe STIS spectral sequence, merging duplicate exposures."""
    print("\nPlotting SN 2011fe spectral sequence...")

    specdir = os.path.join(DATADIR, "SN2011fe", "spectra")
    spectra = load_hst_stis_spectra(specdir)

    print(f"  Merged spectra: {len(spectra)}")

    if not spectra:
        print("  No spectra found.")
        return

    # Combine gratings at same epoch into single spectrum
    epoch_groups = {}
    for s in spectra:
        epoch_key = round(s["mjd"], 0)
        if epoch_key not in epoch_groups:
            epoch_groups[epoch_key] = []
        epoch_groups[epoch_key].append(s)

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.viridis
    all_dt = [round(s["mjd"], 0) - MJD_BMAX for s in spectra]
    unique_epochs = sorted(set(all_dt))
    norm = plt.Normalize(min(unique_epochs), max(unique_epochs))

    offset = 0
    offset_step = 2.5
    plotted_epochs = set()

    for epoch_mjd in sorted(epoch_groups.keys()):
        dt = epoch_mjd - MJD_BMAX
        color = cmap(norm(dt))

        grp = sorted(epoch_groups[epoch_mjd], key=lambda s: s["wave"].min())

        for spec in grp:
            wave = spec["wave"]
            flux = spec["flux"]

            good = np.isfinite(flux) & (wave > 1600) & (wave < 10300)
            if np.sum(good) < 10:
                continue

            flux_g = flux[good]
            wave_g = wave[good]

            # Normalize
            pos = flux_g > 0
            if np.sum(pos) < 10:
                continue
            flux_norm = flux_g / np.max(flux_g[pos])

            # Bin for display
            if len(wave_g) > 300:
                bs = max(1, len(wave_g) // 300)
                n = len(wave_g) // bs
                wave_g = wave_g[:n*bs].reshape(n, bs).mean(axis=1)
                flux_norm = flux_norm[:n*bs].reshape(n, bs).mean(axis=1)

            ax.plot(wave_g, flux_norm + offset, color=color, lw=0.8, alpha=0.9)

        # Label once per epoch
        if epoch_mjd not in plotted_epochs:
            ax.text(10400, offset + 0.8, f"{dt:+.0f}d",
                    fontsize=7, va="center", color=color)
            plotted_epochs.add(epoch_mjd)

        offset += offset_step

    ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=11)
    ax.set_ylabel("Normalized Flux + offset", fontsize=11)
    ax.set_title("SN 2011fe — HST/STIS Spectral Sequence (THATCH)", fontsize=11)
    ax.set_xlim(1500, 11000)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, "SN2011fe_spectral_sequence.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


if __name__ == "__main__":
    plot_at2017gfo_sequence()
    plot_sn2011fe_sequence()
