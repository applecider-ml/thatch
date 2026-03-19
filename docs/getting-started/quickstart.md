# Quick Start

## 1. Find HST-Observed Transients

```python
from thatch.crossmatch import check_hst_coverage

result = check_hst_coverage({
    "name": "AT2017gfo",
    "ra": 197.45037,
    "dec": -23.38148,
    "type": "Kilonova",
})
print(f"{result['n_imaging']} imaging + {result['n_spectroscopy']} spectroscopy")
# 198 imaging + 31 spectroscopy
```

## 2. Extract Photometry

```python
# After downloading images to data/AT2017gfo/
from thatch.cutouts import extract_cutouts_for_object, save_cutouts_hdf5

cutouts = extract_cutouts_for_object(
    "data/AT2017gfo/", ra=197.45037, dec=-23.38148
)
save_cutouts_hdf5(cutouts, "AT2017gfo_cutouts.hdf5", object_name="AT2017gfo")
```

## 3. Build a Catalog

```python
from thatch.catalog import build_catalog, save_catalog, catalog_summary

catalog = build_catalog("data/")
catalog_summary(catalog)
save_catalog(catalog, "thatch_catalog.parquet")
```

## 4. Load from HuggingFace

```python
from thatch.data import load_catalog

catalog = load_catalog()  # Downloads automatically
```

## CLI Usage

```bash
# Cross-match known transients against HST archive
thatch crossmatch

# Build catalog from processed data
thatch catalog data/ --format parquet

# Check pipeline status
python -m thatch.tracker status tracker.parquet
```

## Batch Processing

```bash
# Initialize tracker from cross-match results
python -m thatch.tracker init thatch_hst_transients.csv

# Run pipeline (resumable)
python -m thatch.tracker run thatch_tracker.parquet --datadir data/
```
