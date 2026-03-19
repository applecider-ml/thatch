# THATCH

**Transient Harvester for Archival Training and Classification from Hubble**

[![CI](https://github.com/applecider-ml/thatch/actions/workflows/ci.yml/badge.svg)](https://github.com/applecider-ml/thatch/actions/workflows/ci.yml)
[![Docs](https://github.com/applecider-ml/thatch/actions/workflows/docs.yml/badge.svg)](https://applecider-ml.github.io/thatch/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/applecider-ml/thatch)

**[Documentation](https://applecider-ml.github.io/thatch/)** | **[Dataset](https://huggingface.co/datasets/applecider-ml/thatch)** | **[Paper (in prep)]()**

THATCH is an open-source pipeline for extracting ML-ready transient light curves, image cutouts, and spectra from the Hubble Space Telescope archive. It bridges the domain gap between ground-based surveys (ZTF, Rubin/LSST) and space-based missions (Roman) by providing empirical space-based training data for transient classifiers.

## Features

- **thatch-harvest**: Query MAST for HST observations of known transients, download calibrated images
- **thatch-photometry**: Aperture photometry with HST-specific aperture corrections, validated to 0.02 mag RMS
- **thatch-spectra**: Extract STIS slit spectra and WFC3/IR grism spectra (via grizli)
- **thatch-cutouts**: Generate image cutouts in HDF5 format for vision transformer ingestion
- **thatch-catalog**: Build unified multi-format catalogs (CSV, Parquet, FITS)
- **thatch-crossmatch**: Cross-match transient catalogs (TNS, OSC) against the HST archive

## Installation

```bash
pip install thatch
```

For grism extraction support:
```bash
pip install "thatch[grism]"
```

## Quick Start

```python
from thatch.crossmatch import check_hst_coverage
from thatch.cutouts import extract_cutouts_for_object, save_cutouts_hdf5
from thatch.catalog import build_catalog

# Check if a transient has HST data
result = check_hst_coverage({"name": "AT2017gfo", "ra": 197.45037, "dec": -23.38148})
print(f"{result['n_imaging']} imaging + {result['n_spectroscopy']} spectroscopy observations")

# Extract cutouts
cutouts = extract_cutouts_for_object("data/AT2017gfo/", ra=197.45037, dec=-23.38148)
save_cutouts_hdf5(cutouts, "AT2017gfo_cutouts.hdf5", object_name="AT2017gfo")

# Build catalog from processed data
catalog = build_catalog("data/")
catalog.to_parquet("thatch_catalog.parquet")
```

## CLI

```bash
thatch crossmatch                          # Find HST-observed transients
thatch cutouts AT2017gfo --ra 197.45 --dec -23.38 --datadir data/AT2017gfo/
thatch catalog data/ --format all          # Build catalog in CSV/Parquet/FITS
```

## Demonstrated Objects

| Object | Type | HST Obs | Filters | Spectroscopy |
|--------|------|---------|---------|-------------|
| SN 1987A | SN II-pec | 1,937 | 10 | STIS, COS |
| SN Refsdal | Lensed SN | 1,430 | 7 | WFC3 grism |
| SN 2005cs | SN IIP | 724 | 15+ | STIS |
| SN 2014J | SN Ia | 611 | 15 | STIS |
| SN 2011fe | SN Ia | 292 | 10+ | STIS (43 spectra) |
| AT2017gfo | Kilonova | 229 | 8 | STIS + WFC3 grism |
| SN 1993J | SN IIb | 170 | 10+ | — |
| ASASSN-14li | TDE | 113 | 5+ | STIS |
| AT2018cow | FBOT | 81 | 5+ | — |

## Photometric Accuracy

THATCH aperture photometry with HST-standard aperture corrections achieves **0.02 mag RMS** agreement with published values, validated against Cowperthwaite+2017 and Lyman+2018 measurements of AT2017gfo.

## Citation

If you use THATCH in your research, please cite:

```
Coughlin et al. (in prep). THATCH: Transient Harvester for Archival Training
and Classification from Hubble.
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
