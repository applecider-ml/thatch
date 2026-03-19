"""
thatch-catalog: Curated multimodal transient database.

Merges photometry, cutouts, spectra, and metadata into a standardized
schema. Primary formats: Parquet (tabular) and HDF5 (arrays).
"""

import os
import numpy as np
import pandas as pd
from glob import glob


STANDARD_COLUMNS = [
    "object",
    "filename",
    "mjd",
    "delta_t_days",
    "filter",
    "instrument",
    "exptime_s",
    "count_rate",
    "count_rate_err",
    "ab_mag",
    "ab_mag_err",
    "x_pix",
    "y_pix",
    "photflam",
    "photplam",
    "bg_median",
    "bg_std",
]


def build_catalog(data_dir, objects=None):
    """Build a unified THATCH catalog from processed object directories.

    Reads per-object photometry files (Parquet preferred, CSV fallback)
    and combines into a single standardized DataFrame.

    Parameters
    ----------
    data_dir : str
        Root data directory containing per-object subdirectories.
    objects : list of str, optional
        List of object names. If None, autodiscover from directory.

    Returns
    -------
    pd.DataFrame
        Combined photometry catalog with standardized columns.
    """
    if objects is None:
        # Autodiscover — prefer Parquet, fall back to CSV
        parquets = glob(os.path.join(data_dir, "*", "*_photometry.parquet"))
        csvs = glob(os.path.join(data_dir, "*", "*_photometry.csv"))
        found = {os.path.basename(os.path.dirname(p)): p for p in csvs}
        found.update({os.path.basename(os.path.dirname(p)): p for p in parquets})
        objects_files = found
    else:
        objects_files = {}
        for obj in objects:
            pq = os.path.join(data_dir, obj, f"{obj}_photometry.parquet")
            csv = os.path.join(data_dir, obj, f"{obj}_photometry.csv")
            if os.path.exists(pq):
                objects_files[obj] = pq
            elif os.path.exists(csv):
                objects_files[obj] = csv

    all_phot = []
    for obj, fpath in sorted(objects_files.items()):
        if fpath.endswith(".parquet"):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath)
        df["object"] = obj
        all_phot.append(df)

    if not all_phot:
        return pd.DataFrame()

    catalog = pd.concat(all_phot, ignore_index=True)

    # Standardize columns
    for col in STANDARD_COLUMNS:
        if col not in catalog.columns:
            catalog[col] = np.nan

    # Keep standard columns plus any extras
    cols = STANDARD_COLUMNS + [c for c in catalog.columns if c not in STANDARD_COLUMNS]
    return catalog[cols]


def save_catalog(catalog, outpath):
    """Save catalog to Parquet (primary format).

    Parameters
    ----------
    catalog : pd.DataFrame
        THATCH catalog.
    outpath : str
        Output path. Should end in .parquet.
    """
    catalog.to_parquet(outpath, index=False)
    print(f"  Saved catalog ({len(catalog)} rows) to {outpath}")


def load_catalog(path):
    """Load a THATCH catalog from Parquet.

    Parameters
    ----------
    path : str
        Path to .parquet file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_parquet(path)


def catalog_summary(catalog):
    """Print summary statistics for a THATCH catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        THATCH photometry catalog.
    """
    print("THATCH Catalog Summary")
    print(f"  Total measurements: {len(catalog)}")
    print(f"  Objects: {catalog['object'].nunique()}")

    valid = catalog[catalog["ab_mag"].notna() & (catalog["ab_mag"] < 30)]
    print(f"  Valid detections: {len(valid)}")

    print("\n  By object:")
    for obj, grp in catalog.groupby("object"):
        v = grp[grp["ab_mag"].notna() & (grp["ab_mag"] < 30)]
        filters = sorted(v["filter"].unique()) if len(v) > 0 else []
        print(f"    {obj:20s}: {len(grp):4d} total, {len(v):4d} detections, {len(filters)} filters")

    print("\n  By filter:")
    for filt, grp in valid.groupby("filter"):
        print(f"    {filt:10s}: {len(grp):4d} detections across {grp['object'].nunique()} objects")


def main():
    """Build catalog from default data directory."""
    import sys

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Building THATCH catalog from {data_dir}")
    catalog = build_catalog(data_dir)

    if len(catalog) == 0:
        print("No photometry found.")
        return

    catalog_summary(catalog)

    outpath = os.path.join(data_dir, "thatch_catalog.parquet")
    save_catalog(catalog, outpath)


if __name__ == "__main__":
    main()
