# THATCH

**Transient Harvester for Archival Training and Classification from Hubble**

THATCH is an open-source pipeline for extracting ML-ready transient light curves, image cutouts, and spectra from the Hubble Space Telescope archive. It bridges the domain gap between ground-based surveys (ZTF, Rubin/LSST) and space-based missions (Roman) by providing empirical space-based training data for transient classifiers.

## Why THATCH?

The Nancy Grace Roman Space Telescope's High Latitude Time Domain Survey (HLTDS) will discover ~12,000 Type Ia supernovae and thousands of other transients. Current classifiers are trained on ground-based data, but ground-based and space-based observations differ fundamentally in PSF, backgrounds, pixel scale, and filter bandpasses.

**HST is the only existing space-based observatory with 30+ years of transient photometry and spectroscopy** — making it the ideal bridge between ground-based training data and Roman deployment.

## Features

| Module | Description |
|--------|-------------|
| `thatch.crossmatch` | Cross-match transient catalogs against the HST archive |
| `thatch.harvest` | Download calibrated HST images from MAST |
| `thatch.photometry` | Aperture photometry with HST aperture corrections |
| `thatch.spectra` | STIS slit and WFC3 grism spectral extraction |
| `thatch.cutouts` | Image cutout extraction to HDF5 |
| `thatch.catalog` | Unified catalog builder (Parquet/HDF5) |
| `thatch.tracker` | Pipeline job tracker for batch processing |
| `thatch.data` | Upload/download from HuggingFace |

## Data Access

The THATCH dataset is hosted on HuggingFace:

```python
from thatch.data import load_catalog

catalog = load_catalog()  # Downloads from HuggingFace
print(f"{len(catalog)} measurements across {catalog['object'].nunique()} transients")
```

[:fontawesome-solid-database: Browse Dataset on HuggingFace](https://huggingface.co/datasets/applecider-ml/thatch){ .md-button }

## Quick Start

```bash
pip install thatch
```

```python
from thatch.crossmatch import check_hst_coverage

result = check_hst_coverage({
    "name": "AT2017gfo", "ra": 197.45037, "dec": -23.38148
})
print(f"{result['n_imaging']} imaging + {result['n_spectroscopy']} spectroscopy")
```

## Photometric Accuracy

THATCH achieves **0.02 mag RMS** agreement with published HST photometry, validated against Cowperthwaite+2017 and Lyman+2018 measurements of the kilonova AT2017gfo.
