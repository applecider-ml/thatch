# Installation

## Basic Install

```bash
pip install thatch
```

## With Optional Dependencies

```bash
# For grism extraction via grizli
pip install "thatch[grism]"

# For HuggingFace data access
pip install "thatch[hf]"

# Everything
pip install "thatch[all]"

# Development
pip install "thatch[dev]"
```

## From Source

```bash
git clone https://github.com/mcoughlin/thatch.git
cd thatch
pip install -e ".[dev]"
```

## Dependencies

**Required:**

- Python >= 3.10
- numpy, pandas, astropy, astroquery, matplotlib, scipy, photutils

**Optional:**

- `grizli` — WFC3/IR grism spectral extraction
- `huggingface_hub` + `pyarrow` — dataset upload/download
- `h5py` — HDF5 cutout/spectra storage
