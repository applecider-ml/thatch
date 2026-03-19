# Cross-Matching

The first step in THATCH is identifying which known transients have HST observations.

## How It Works

For each transient, `thatch.crossmatch` queries the MAST archive within a configurable search radius (default 5"), filters to HST, and separates imaging from spectroscopy (STIS slit, WFC3/ACS grism, COS).

```python
from thatch.crossmatch import check_hst_coverage

result = check_hst_coverage({
    "name": "AT2017gfo", "ra": 197.45037, "dec": -23.38148
})
print(f"{result['n_imaging']} imaging + {result['n_spectroscopy']} spectroscopy")
# 198 imaging + 31 spectroscopy
```

## Results

A cross-match of 37 known transients finds **17 with HST data**, totaling **6,133 observations** (5,020 imaging + 1,109 spectroscopic):

| Object | Type | Obs | Imaging | Spectroscopy | Programs |
|--------|------|-----|---------|-------------|----------|
| SN 1987A | SN II-pec | 1,937 | 1,240 | 693 | 76 |
| SN Refsdal | Lensed SN | 1,430 | 1,289 | 141 | 61 |
| SN 2005cs | SN IIP | 724 | 664 | 60 | 30 |
| SN 2014J | SN Ia | 611 | 564 | 47 | 35 |
| SN 2011fe | SN Ia | 292 | 217 | 75 | 25 |
| AT2017gfo | Kilonova | 229 | 198 | 31 | 23 |
| SN 1993J | SN IIb | 170 | 170 | 0 | 22 |
| ASASSN-14li | TDE | 113 | 66 | 47 | 11 |
| AT2018cow | FBOT | 81 | 81 | 0 | 5 |

## Output

The cross-match produces a Parquet table with `name`, `ra`, `dec`, `type`, `n_imaging`, `n_spectroscopy`, `filters_imaging`, `instruments`, `programs`, `mjd_min`, `mjd_max`.
