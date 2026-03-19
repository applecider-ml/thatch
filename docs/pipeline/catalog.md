# Catalog

The `thatch.catalog` module merges per-object photometry into a unified catalog in Parquet format.

## Building the Catalog

```python
from thatch.catalog import build_catalog, save_catalog, catalog_summary

catalog = build_catalog("data/")
catalog_summary(catalog)
save_catalog(catalog, "thatch_catalog.parquet")
```

Output:
```
THATCH Catalog Summary
  Total measurements: 584
  Objects: 12
  Valid detections: 449

  By object:
    AT2017gfo           :   18 total,    6 detections, 3 filters
    SN1987A             :  160 total,  143 detections, 10 filters
    SN2011fe            :   37 total,   11 detections, 6 filters
    SN2014J             :   25 total,   20 detections, 11 filters
    ...
```

## Schema

The catalog uses a standardized column set:

| Column | Type | Description |
|--------|------|-------------|
| `object` | str | Transient name |
| `mjd` | float | Modified Julian Date |
| `delta_t_days` | float | Days from reference epoch |
| `filter` | str | HST filter name (e.g., F110W) |
| `instrument` | str | Instrument/detector (e.g., WFC3/IR) |
| `ab_mag` | float | Calibrated AB magnitude |
| `ab_mag_err` | float | Magnitude uncertainty |
| `count_rate` | float | Aperture-corrected count rate (e-/s) |
| `photflam` | float | Inverse sensitivity (erg/cm²/Å/e-) |

## Data Formats

- **Primary**: Parquet (fast columnar reads, HuggingFace compatible)
- **Secondary**: HDF5 for cutouts and spectra (array data)

## Loading from HuggingFace

```python
from thatch.data import load_catalog

catalog = load_catalog()  # Downloads from applecider-ml/thatch
```
