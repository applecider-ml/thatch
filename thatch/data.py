"""
thatch.data: Upload and download THATCH data products from Hugging Face.

Data is stored at: https://huggingface.co/datasets/applecider-ml/thatch
"""

import os
from pathlib import Path

HF_ORG = "applecider-ml"
HF_REPO = f"{HF_ORG}/thatch"


def upload_catalog(data_dir, token=None):
    """Upload THATCH catalog files to Hugging Face.

    Parameters
    ----------
    data_dir : str
        Directory containing thatch_catalog.{csv,parquet,fits} and
        per-object subdirectories.
    token : str, optional
        HF API token. If None, uses cached login.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Ensure dataset repo exists
    try:
        api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Could not create repo {HF_REPO}: {e}")
        return

    data_path = Path(data_dir)

    # Upload catalog (Parquet primary, CSV secondary)
    for ext in ["parquet", "csv"]:
        fpath = data_path / f"thatch_catalog.{ext}"
        if fpath.exists():
            print(f"  Uploading {fpath.name}...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"catalog/thatch_catalog.{ext}",
                repo_id=HF_REPO,
                repo_type="dataset",
            )

    # Upload cross-match results
    xmatch = data_path / "thatch_hst_transients.csv"
    if xmatch.exists():
        print(f"  Uploading {xmatch.name}...")
        api.upload_file(
            path_or_fileobj=str(xmatch),
            path_in_repo="catalog/thatch_hst_transients.csv",
            repo_id=HF_REPO,
            repo_type="dataset",
        )

    # Upload per-object photometry
    for phot_csv in sorted(data_path.glob("*/*_photometry.csv")):
        obj_name = phot_csv.parent.name
        print(f"  Uploading {obj_name} photometry...")
        api.upload_file(
            path_or_fileobj=str(phot_csv),
            path_in_repo=f"photometry/{obj_name}/{phot_csv.name}",
            repo_id=HF_REPO,
            repo_type="dataset",
        )

    # Upload per-object cutouts (HDF5)
    for hdf5 in sorted(data_path.glob("*/*_cutouts.hdf5")):
        obj_name = hdf5.parent.name
        print(f"  Uploading {obj_name} cutouts...")
        api.upload_file(
            path_or_fileobj=str(hdf5),
            path_in_repo=f"cutouts/{obj_name}/{hdf5.name}",
            repo_id=HF_REPO,
            repo_type="dataset",
        )

    # Upload per-object spectra (HDF5 primary, FITS secondary)
    for spec_file in sorted(data_path.glob("*/spectra/*_spectra.hdf5")):
        obj_name = spec_file.parent.parent.name
        print(f"  Uploading {obj_name} spectra...")
        api.upload_file(
            path_or_fileobj=str(spec_file),
            path_in_repo=f"spectra/{obj_name}/{spec_file.name}",
            repo_id=HF_REPO,
            repo_type="dataset",
        )

    print(f"\nDone. Data available at: https://huggingface.co/datasets/{HF_REPO}")


def download_catalog(outdir=".", token=None):
    """Download the THATCH catalog from Hugging Face.

    Parameters
    ----------
    outdir : str
        Directory to save downloaded files.
    token : str, optional
        HF API token for private repos.

    Returns
    -------
    str
        Path to downloaded catalog CSV.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(outdir, exist_ok=True)

    fpath = hf_hub_download(
        repo_id=HF_REPO,
        filename="catalog/thatch_catalog.parquet",
        repo_type="dataset",
        local_dir=outdir,
        token=token,
    )
    print(f"Downloaded catalog to {fpath}")
    return fpath


def download_object(object_name, outdir=".", token=None, include_cutouts=True):
    """Download all data for a specific object.

    Parameters
    ----------
    object_name : str
        Object name (e.g., "AT2017gfo").
    outdir : str
        Output directory.
    token : str, optional
        HF API token.
    include_cutouts : bool
        Whether to download HDF5 cutouts (can be large).

    Returns
    -------
    dict
        Paths to downloaded files.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(outdir, exist_ok=True)
    downloaded = {}

    # Photometry
    try:
        fpath = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"photometry/{object_name}/{object_name}_photometry.csv",
            repo_type="dataset",
            local_dir=outdir,
            token=token,
        )
        downloaded["photometry"] = fpath
        print(f"  Downloaded photometry: {fpath}")
    except Exception as e:
        print(f"  No photometry for {object_name}: {e}")

    # Spectra
    try:
        fpath = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"spectra/{object_name}/{object_name}_spectra.fits",
            repo_type="dataset",
            local_dir=outdir,
            token=token,
        )
        downloaded["spectra"] = fpath
        print(f"  Downloaded spectra: {fpath}")
    except Exception:
        pass

    # Cutouts
    if include_cutouts:
        try:
            fpath = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"cutouts/{object_name}/{object_name}_cutouts.hdf5",
                repo_type="dataset",
                local_dir=outdir,
                token=token,
            )
            downloaded["cutouts"] = fpath
            print(f"  Downloaded cutouts: {fpath}")
        except Exception:
            pass

    return downloaded


def load_catalog(token=None):
    """Load the THATCH catalog directly as a pandas DataFrame.

    Downloads from HF if not cached locally.

    Returns
    -------
    pd.DataFrame
        THATCH photometry catalog.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    fpath = hf_hub_download(
        repo_id=HF_REPO,
        filename="catalog/thatch_catalog.parquet",
        repo_type="dataset",
        token=token,
    )
    return pd.read_parquet(fpath)
