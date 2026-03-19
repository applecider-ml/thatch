# Pipeline Overview

THATCH processes transients through a multi-stage pipeline:

```
Cross-match → Download → Photometry → Spectroscopy → Cutouts → Catalog
```

Each stage is tracked independently and can be retried on failure.

## Stages

### 1. Cross-Match (`thatch.crossmatch`)
Query known transient catalogs (TNS, OSC) against the MAST HST archive to identify which transients have HST observations.

### 2. Download (`thatch.harvest`)
Download calibrated HST images (\_drz, \_drc) from MAST for each transient.

### 3. Photometry (`thatch.photometry`)
Perform aperture photometry at the known transient position with HST-specific aperture corrections from the instrument handbooks. Validated to 0.02 mag RMS.

### 4. Spectroscopy (`thatch.spectra`)
Extract 1D spectra from STIS slit observations (pipeline x1d/sx1 files) and WFC3/IR grism data (via grizli forward-modeling).

### 5. Cutouts (`thatch.cutouts`)
Generate multi-band image cutouts centered on transient positions, stored as HDF5 for ML ingestion.

### 6. Catalog (`thatch.catalog`)
Merge all products into a unified Parquet catalog with standardized schema.

## Data Formats

| Data Type | Format | Description |
|-----------|--------|-------------|
| Photometry | Parquet | Per-object and unified catalogs |
| Spectra | HDF5 | Wavelength, flux, error arrays + metadata |
| Cutouts | HDF5 | Image arrays + WCS + metadata |

## Job Tracking

The `thatch.tracker` module maintains a Parquet-based status table for batch processing. It tracks each object through the pipeline stages and supports resumable execution.
