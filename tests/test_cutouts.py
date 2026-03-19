"""Tests for thatch.cutouts."""

import numpy as np
import pytest

from astropy.io import fits


@pytest.fixture
def sample_fits(tmp_path):
    """Create a sample FITS image for testing."""
    ny, nx = 200, 200
    data = np.random.normal(100, 10, (ny, nx)).astype(np.float32)

    # Add a point source at center
    data[100, 100] += 500

    # Use PrimaryHDU with data directly to avoid header ordering issues
    pri = fits.PrimaryHDU(data=data)
    pri.header["CRPIX1"] = 100
    pri.header["CRPIX2"] = 100
    pri.header["CRVAL1"] = 197.45
    pri.header["CRVAL2"] = -23.38
    pri.header["CD1_1"] = -1.3889e-5  # 0.05"/pix
    pri.header["CD1_2"] = 0.0
    pri.header["CD2_1"] = 0.0
    pri.header["CD2_2"] = 1.3889e-5
    pri.header["CTYPE1"] = "RA---TAN"
    pri.header["CTYPE2"] = "DEC--TAN"
    pri.header["FILTER"] = "F110W"
    pri.header["INSTRUME"] = "WFC3"
    pri.header["DETECTOR"] = "IR"
    pri.header["EXPTIME"] = 300.0
    pri.header["DATE-OBS"] = "2017-08-22"
    pri.header["TIME-OBS"] = "12:00:00"
    pri.header["PROPOSID"] = 99999

    hdul = fits.HDUList([pri])
    fpath = str(tmp_path / "test_drz.fits")
    hdul.writeto(fpath, overwrite=True)
    return fpath


def test_extract_cutout(sample_fits):
    """extract_cutout should return a valid cutout dict."""
    from thatch.cutouts import extract_cutout

    result = extract_cutout(sample_fits, ra=197.45, dec=-23.38, size_arcsec=2.0)
    assert result is not None
    assert "data" in result
    assert result["data"].ndim == 2
    assert result["filter"] == "F110W"


def test_extract_cutout_offimage(sample_fits):
    """extract_cutout should return None for position off the image."""
    from thatch.cutouts import extract_cutout

    result = extract_cutout(sample_fits, ra=0.0, dec=0.0, size_arcsec=2.0)
    assert result is None


def test_save_cutouts_hdf5(sample_fits, tmp_path):
    """save_cutouts_hdf5 should create a valid HDF5 file."""
    from thatch.cutouts import extract_cutout, save_cutouts_hdf5
    import h5py

    cutout = extract_cutout(sample_fits, ra=197.45, dec=-23.38, size_arcsec=2.0)
    outpath = str(tmp_path / "test_cutouts.hdf5")
    save_cutouts_hdf5([cutout], outpath, object_name="TestObj")

    with h5py.File(outpath, "r") as f:
        assert f.attrs["object_name"] == "TestObj"
        assert f.attrs["n_cutouts"] == 1
        assert "cutout_0000" in f
        assert f["cutout_0000"]["data"].shape == cutout["data"].shape
