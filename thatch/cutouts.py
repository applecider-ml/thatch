"""
thatch-cutouts: Extract image cutouts centered on transient positions.

Generates multi-band science image cutouts from HST drizzled images,
suitable for ML classifier ingestion (vision transformers, CNNs).
"""

import os
import warnings
import numpy as np
from glob import glob

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

warnings.filterwarnings("ignore")


def extract_cutout(fits_path, ra, dec, size_arcsec=5.0, size_pixels=None):
    """Extract a square cutout from a FITS image at a given position.

    Parameters
    ----------
    fits_path : str
        Path to FITS image (drz/drc).
    ra, dec : float
        Target coordinates in degrees.
    size_arcsec : float
        Cutout size in arcseconds (used if size_pixels is None).
    size_pixels : int or None
        Cutout size in pixels. Overrides size_arcsec if given.

    Returns
    -------
    dict or None
        Cutout data with image array, WCS, and metadata.
    """
    fname = os.path.basename(fits_path)

    with fits.open(fits_path) as hdul:
        # Find science extension
        sci_ext = None
        for i, hdu in enumerate(hdul):
            if hdu.name == "SCI" or (
                hasattr(hdu, "data") and hdu.data is not None
                and hdu.data.ndim == 2 and i > 0
            ):
                sci_ext = i
                break
        if sci_ext is None and hdul[0].data is not None and hdul[0].data.ndim == 2:
            sci_ext = 0
        if sci_ext is None:
            return None

        header = hdul[sci_ext].header
        data = hdul[sci_ext].data.astype(float)
        pri = hdul[0].header

        # Get WCS
        wcs = WCS(header, naxis=2)
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        # Determine pixel scale
        try:
            pscale = np.abs(header.get("CD1_1", header.get("CDELT1", 1e-5))) * 3600
        except (ValueError, TypeError):
            pscale = 0.05  # default

        if size_pixels is None:
            size_pixels = int(round(size_arcsec / pscale))

        size_pixels = max(size_pixels, 11)  # minimum 11x11

        # Extract cutout
        try:
            cutout = Cutout2D(data, coord, (size_pixels, size_pixels), wcs=wcs)
        except Exception:
            return None

        # Get metadata
        filt = (header.get("FILTER") or header.get("FILTER1") or
                pri.get("FILTER") or pri.get("FILTER1") or "UNKNOWN")
        filt2 = header.get("FILTER2", pri.get("FILTER2", ""))
        if filt in ["CLEAR1L", "CLEAR1S", "CLEAR", "N/A"] and filt2:
            filt = filt2

        instrument = f"{pri.get('INSTRUME', '')}/{pri.get('DETECTOR', '')}"
        exptime = float(header.get("EXPTIME", pri.get("EXPTIME", 0)))

        expstart = pri.get("EXPSTART")
        if expstart:
            mjd = float(expstart)
        else:
            from astropy.time import Time
            date_obs = pri.get("DATE-OBS", header.get("DATE-OBS", ""))
            time_obs = pri.get("TIME-OBS", header.get("TIME-OBS", "00:00:00"))
            try:
                t = Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")
                mjd = t.mjd
            except Exception:
                mjd = np.nan

        return {
            "filename": fname,
            "data": cutout.data,
            "wcs": cutout.wcs,
            "shape": cutout.shape,
            "filter": filt,
            "instrument": instrument,
            "exptime_s": exptime,
            "mjd": mjd,
            "pixel_scale": pscale,
            "proposal_id": str(pri.get("PROPOSID", "")),
        }


def extract_cutouts_for_object(data_dir, ra, dec, size_arcsec=5.0):
    """Extract cutouts from all FITS images in a directory.

    Parameters
    ----------
    data_dir : str
        Directory containing drz/drc FITS files.
    ra, dec : float
        Target coordinates.
    size_arcsec : float
        Cutout size.

    Returns
    -------
    list of dict
        List of cutout records.
    """
    fits_files = []
    for ext in ["*_drz.fits", "*_drc.fits"]:
        fits_files.extend(glob(os.path.join(data_dir, ext)))

    cutouts = []
    for fpath in sorted(fits_files):
        try:
            c = extract_cutout(fpath, ra, dec, size_arcsec=size_arcsec)
            if c is not None:
                cutouts.append(c)
        except Exception:
            continue

    return cutouts


def save_cutouts_hdf5(cutouts, outpath, object_name="transient"):
    """Save cutouts to an HDF5 file.

    Parameters
    ----------
    cutouts : list of dict
        Cutout records from extract_cutout.
    outpath : str
        Output HDF5 file path.
    object_name : str
        Object name for metadata.
    """
    import h5py

    with h5py.File(outpath, "w") as f:
        f.attrs["object_name"] = object_name
        f.attrs["n_cutouts"] = len(cutouts)

        for i, c in enumerate(cutouts):
            grp = f.create_group(f"cutout_{i:04d}")
            grp.create_dataset("data", data=c["data"], compression="gzip")
            grp.attrs["filename"] = c["filename"]
            grp.attrs["filter"] = c["filter"]
            grp.attrs["instrument"] = c["instrument"]
            grp.attrs["exptime_s"] = c["exptime_s"]
            grp.attrs["mjd"] = c["mjd"] if np.isfinite(c["mjd"]) else 0.0
            grp.attrs["pixel_scale"] = c["pixel_scale"]
            grp.attrs["proposal_id"] = c["proposal_id"]
            grp.attrs["shape"] = c["shape"]

    print(f"  Saved {len(cutouts)} cutouts to {outpath}")


def plot_cutout_mosaic(cutouts, title="", outpath=None, ncols=6):
    """Plot a mosaic of cutouts, one per filter/epoch.

    Parameters
    ----------
    cutouts : list of dict
        Cutout records.
    title : str
        Plot title.
    outpath : str
        Output path for figure.
    ncols : int
        Number of columns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not cutouts:
        return

    n = len(cutouts)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, c in enumerate(cutouts):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        # Display with zscale-like stretch
        data = c["data"]
        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 99)

        ax.imshow(data, origin="lower", cmap="gray_r",
                  vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{c['filter']}\nMJD {c['mjd']:.0f}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty panels
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.savefig(outpath.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
