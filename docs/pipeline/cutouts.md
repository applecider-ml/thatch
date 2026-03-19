# Cutouts

THATCH generates multi-band image cutouts centered on transient positions, stored in HDF5 for direct ingestion by ML classifiers (vision transformers, CNNs).

## Method

For each drizzled HST image:

1. Read the WCS and compute the pixel position of the transient
2. Extract a square cutout using `astropy.nddata.Cutout2D`
3. Store the cutout array, WCS, and metadata in an HDF5 group

The cutout size defaults to 5" (adjustable), which corresponds to different pixel sizes per instrument:

| Instrument | Pixel Scale | 5" Cutout |
|-----------|-------------|-----------|
| WFC3/UVIS | 0.04"/pix | 125×125 px |
| WFC3/IR | 0.13"/pix | 38×38 px |
| ACS/WFC | 0.05"/pix | 100×100 px |

## Usage

```python
from thatch.cutouts import extract_cutouts_for_object, save_cutouts_hdf5

cutouts = extract_cutouts_for_object("data/AT2017gfo/", ra=197.45, dec=-23.38)
save_cutouts_hdf5(cutouts, "AT2017gfo_cutouts.hdf5", object_name="AT2017gfo")
```

## Example: AT2017gfo

18 cutouts across 8 filters (F110W, F160W, F153M, F606W, F475W, F336W, F275W, F225W):

![AT2017gfo cutouts](../images/AT2017gfo_cutouts.png)

*Multi-band HST cutouts of AT2017gfo. The kilonova is clearly detected in the NIR (F110W, F160W) and fades in the UV.*

## HDF5 Schema

```
object_cutouts.hdf5
├── attrs: object_name, n_cutouts
├── cutout_0000/
│   ├── data          # 2D float array
│   └── attrs: filter, instrument, mjd, exptime_s, pixel_scale, proposal_id
├── cutout_0001/
│   └── ...
```
