"""
thatch.tracker: Pipeline job tracker for batch processing.

Maintains a Parquet-based status table tracking the download and
processing state of every transient in the THATCH catalog.

Status flow per object:
  queued -> downloading -> downloaded -> photometry -> spectra -> cutouts -> complete

Each stage is independent and can be retried on failure.
"""

import os
import time
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd


class Status(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    DOWNLOAD_FAILED = "download_failed"
    PHOTOMETRY = "photometry"
    PHOTOMETRY_DONE = "photometry_done"
    PHOTOMETRY_FAILED = "photometry_failed"
    SPECTRA = "spectra"
    SPECTRA_DONE = "spectra_done"
    SPECTRA_FAILED = "spectra_failed"
    CUTOUTS = "cutouts"
    CUTOUTS_DONE = "cutouts_done"
    CUTOUTS_FAILED = "cutouts_failed"
    COMPLETE = "complete"
    SKIPPED = "skipped"


TRACKER_COLUMNS = [
    "name", "ra", "dec", "type", "z", "discovery_mjd",
    "n_hst_imaging", "n_hst_spectroscopy", "n_hst_programs",
    "status",
    "n_images_downloaded", "n_spectra_downloaded",
    "n_photometry_measurements", "n_valid_detections",
    "n_cutouts", "n_spectra_extracted",
    "download_started", "download_completed",
    "photometry_started", "photometry_completed",
    "spectra_started", "spectra_completed",
    "cutouts_started", "cutouts_completed",
    "last_error", "last_updated",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_tracker(crossmatch_path, tracker_path):
    """Initialize a tracker from a cross-match results CSV.

    Parameters
    ----------
    crossmatch_path : str
        Path to thatch_hst_transients.csv from crossmatch.
    tracker_path : str
        Path to save the tracker Parquet file.

    Returns
    -------
    pd.DataFrame
        Initialized tracker.
    """
    xmatch = pd.read_csv(crossmatch_path)

    records = []
    for _, row in xmatch.iterrows():
        records.append({
            "name": row["name"],
            "ra": row["ra"],
            "dec": row["dec"],
            "type": row["type"],
            "z": row.get("z", np.nan),
            "discovery_mjd": row.get("discovery_mjd", np.nan),
            "n_hst_imaging": int(row.get("n_imaging", 0)),
            "n_hst_spectroscopy": int(row.get("n_spectroscopy", 0)),
            "n_hst_programs": int(row.get("n_programs", 0)),
            "status": Status.QUEUED.value,
            "n_images_downloaded": 0,
            "n_spectra_downloaded": 0,
            "n_photometry_measurements": 0,
            "n_valid_detections": 0,
            "n_cutouts": 0,
            "n_spectra_extracted": 0,
            "download_started": None,
            "download_completed": None,
            "photometry_started": None,
            "photometry_completed": None,
            "spectra_started": None,
            "spectra_completed": None,
            "cutouts_started": None,
            "cutouts_completed": None,
            "last_error": None,
            "last_updated": _now(),
        })

    tracker = pd.DataFrame(records)
    tracker.to_parquet(tracker_path, index=False)
    print(f"Initialized tracker with {len(tracker)} objects -> {tracker_path}")
    return tracker


def load_tracker(tracker_path):
    """Load tracker from Parquet."""
    return pd.read_parquet(tracker_path)


def save_tracker(tracker, tracker_path):
    """Save tracker to Parquet."""
    tracker.to_parquet(tracker_path, index=False)


def update_status(tracker, name, status, tracker_path=None, **kwargs):
    """Update the status and metadata for an object.

    Parameters
    ----------
    tracker : pd.DataFrame
        Tracker DataFrame.
    name : str
        Object name.
    status : Status or str
        New status.
    tracker_path : str, optional
        If given, save after updating.
    **kwargs
        Additional columns to update (e.g., n_images_downloaded=42).

    Returns
    -------
    pd.DataFrame
        Updated tracker.
    """
    mask = tracker["name"] == name
    if not mask.any():
        print(f"  Warning: {name} not in tracker")
        return tracker

    tracker.loc[mask, "status"] = status.value if isinstance(status, Status) else str(status)
    tracker.loc[mask, "last_updated"] = _now()

    for key, val in kwargs.items():
        if key in tracker.columns:
            tracker.loc[mask, key] = val

    if tracker_path:
        save_tracker(tracker, tracker_path)

    return tracker


def get_next_objects(tracker, status=Status.QUEUED, n=1):
    """Get the next objects to process for a given status.

    Parameters
    ----------
    tracker : pd.DataFrame
        Tracker.
    status : Status
        Filter by this status.
    n : int
        Number of objects to return.

    Returns
    -------
    pd.DataFrame
        Subset of tracker rows.
    """
    val = status.value if isinstance(status, Status) else str(status)
    pending = tracker[tracker["status"] == val]
    return pending.head(n)


def print_summary(tracker):
    """Print a summary of pipeline progress."""
    print("THATCH Pipeline Tracker")
    print(f"  Total objects: {len(tracker)}")
    print()

    for status in Status:
        count = (tracker["status"] == status.value).sum()
        if count > 0:
            print(f"  {status.value:25s}: {count}")

    print()

    # Stats
    total_img = tracker["n_images_downloaded"].sum()
    total_phot = tracker["n_photometry_measurements"].sum()
    total_det = tracker["n_valid_detections"].sum()
    total_spec = tracker["n_spectra_extracted"].sum()
    total_cut = tracker["n_cutouts"].sum()
    print(f"  Images downloaded: {total_img}")
    print(f"  Photometry measurements: {total_phot} ({total_det} valid detections)")
    print(f"  Spectra extracted: {total_spec}")
    print(f"  Cutouts: {total_cut}")


def run_pipeline(tracker_path, data_dir, stages=None, max_objects=None,
                 max_download=50, delay_between=2.0):
    """Run the full THATCH pipeline on queued objects.

    Parameters
    ----------
    tracker_path : str
        Path to tracker Parquet file.
    data_dir : str
        Root data directory.
    stages : list of str, optional
        Which stages to run. Default: all.
        Options: "download", "photometry", "spectra", "cutouts"
    max_objects : int, optional
        Maximum objects to process in this run.
    max_download : int
        Max images to download per object.
    delay_between : float
        Seconds to wait between objects (MAST rate limiting).
    """
    from glob import glob
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astroquery.mast import Observations

    if stages is None:
        stages = ["download", "photometry", "spectra", "cutouts"]

    tracker = load_tracker(tracker_path)

    # Determine which objects to process
    actionable = tracker[tracker["status"].isin([
        Status.QUEUED.value,
        Status.DOWNLOADED.value,
        Status.PHOTOMETRY_DONE.value,
        Status.SPECTRA_DONE.value,
    ])]

    if max_objects:
        actionable = actionable.head(max_objects)

    print(f"Processing {len(actionable)} objects...")

    for idx, row in actionable.iterrows():
        name = row["name"]
        ra, dec = row["ra"], row["dec"]
        obj_dir = os.path.join(data_dir, name)
        os.makedirs(obj_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{name} ({row['type']}) — status: {row['status']}")
        print(f"{'='*60}")

        # --- DOWNLOAD ---
        if "download" in stages and row["status"] == Status.QUEUED.value:
            tracker = update_status(tracker, name, Status.DOWNLOADING,
                                    tracker_path, download_started=_now())
            try:
                coord = SkyCoord(ra=ra, dec=dec, unit="deg")
                obs = Observations.query_region(coord, radius=5.0 * u.arcsec)
                hst = obs[obs["obs_collection"] == "HST"]

                # Filter to imaging
                import numpy as np_local
                img_mask = np_local.array(["image" in str(r["dataproduct_type"]).lower()
                                           for r in hst])
                img_obs = hst[img_mask]

                if len(img_obs) > 0:
                    products = Observations.get_product_list(img_obs)
                    drz_mask = np_local.array([
                        any(ext in str(pn) for ext in ["_drz.fits", "_drc.fits"])
                        for pn in products["productFilename"]
                    ])
                    drz = products[drz_mask]
                    if len(drz) > max_download:
                        drz = drz[:max_download]

                    if len(drz) > 0:
                        manifest = Observations.download_products(
                            drz, download_dir=obj_dir, flat=True)
                        n_dl = len([f for f in manifest["Local Path"]
                                    if os.path.exists(str(f))])
                    else:
                        n_dl = 0
                else:
                    n_dl = 0

                n_fits = len(glob(os.path.join(obj_dir, "*.fits")))
                tracker = update_status(tracker, name, Status.DOWNLOADED,
                                        tracker_path,
                                        n_images_downloaded=n_fits,
                                        download_completed=_now())
                print(f"  Downloaded {n_fits} images")

            except Exception as e:
                tracker = update_status(tracker, name, Status.DOWNLOAD_FAILED,
                                        tracker_path, last_error=str(e)[:200])
                print(f"  Download failed: {e}")

            time.sleep(delay_between)

        # --- PHOTOMETRY ---
        if "photometry" in stages and row["status"] in [
            Status.DOWNLOADED.value, Status.QUEUED.value
        ]:
            # Check if images exist
            n_fits = len(glob(os.path.join(obj_dir, "*.fits")))
            if n_fits == 0:
                continue

            tracker = update_status(tracker, name, Status.PHOTOMETRY,
                                    tracker_path, photometry_started=_now())
            try:
                from thatch.photometry import process_one_target
                target = {"name": name, "ra": ra, "dec": dec, "type": row["type"]}
                process_one_target(target)

                # Count results
                phot_file = os.path.join(obj_dir, f"{name}_photometry.csv")
                if os.path.exists(phot_file):
                    df = pd.read_csv(phot_file)
                    n_meas = len(df)
                    n_det = len(df[df["ab_mag"].notna() & (df["ab_mag"] < 30)])
                else:
                    n_meas, n_det = 0, 0

                tracker = update_status(tracker, name, Status.PHOTOMETRY_DONE,
                                        tracker_path,
                                        n_photometry_measurements=n_meas,
                                        n_valid_detections=n_det,
                                        photometry_completed=_now())
                print(f"  Photometry: {n_meas} measurements, {n_det} detections")

            except Exception as e:
                tracker = update_status(tracker, name, Status.PHOTOMETRY_FAILED,
                                        tracker_path, last_error=str(e)[:200])
                print(f"  Photometry failed: {e}")

        # --- CUTOUTS ---
        if "cutouts" in stages and tracker.loc[
            tracker["name"] == name, "status"
        ].values[0] in [Status.PHOTOMETRY_DONE.value, Status.SPECTRA_DONE.value]:
            tracker = update_status(tracker, name, Status.CUTOUTS,
                                    tracker_path, cutouts_started=_now())
            try:
                from thatch.cutouts import extract_cutouts_for_object, save_cutouts_hdf5
                cutouts = extract_cutouts_for_object(obj_dir, ra, dec,
                                                      size_arcsec=5.0)
                if cutouts:
                    hdf5_path = os.path.join(obj_dir, f"{name}_cutouts.hdf5")
                    save_cutouts_hdf5(cutouts, hdf5_path, object_name=name)

                tracker = update_status(tracker, name, Status.COMPLETE,
                                        tracker_path,
                                        n_cutouts=len(cutouts) if cutouts else 0,
                                        cutouts_completed=_now())
                print(f"  Cutouts: {len(cutouts) if cutouts else 0}")

            except Exception as e:
                tracker = update_status(tracker, name, Status.CUTOUTS_FAILED,
                                        tracker_path, last_error=str(e)[:200])
                print(f"  Cutouts failed: {e}")

    # Final summary
    tracker = load_tracker(tracker_path)
    print(f"\n{'='*60}")
    print_summary(tracker)

    return tracker


def main():
    """CLI entry point for the tracker."""
    import argparse

    parser = argparse.ArgumentParser(description="THATCH Pipeline Tracker")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="Initialize tracker from cross-match CSV")
    p_init.add_argument("crossmatch_csv", help="Cross-match results CSV")
    p_init.add_argument("--output", "-o", default="thatch_tracker.parquet")

    p_status = sub.add_parser("status", help="Print pipeline status")
    p_status.add_argument("tracker", help="Tracker Parquet file")

    p_run = sub.add_parser("run", help="Run pipeline on queued objects")
    p_run.add_argument("tracker", help="Tracker Parquet file")
    p_run.add_argument("--datadir", default="data", help="Data directory")
    p_run.add_argument("--max-objects", type=int, help="Max objects to process")
    p_run.add_argument("--max-download", type=int, default=50)
    p_run.add_argument("--stages", nargs="+",
                       default=["download", "photometry", "cutouts"],
                       help="Pipeline stages to run")

    args = parser.parse_args()

    if args.cmd == "init":
        init_tracker(args.crossmatch_csv, args.output)
    elif args.cmd == "status":
        tracker = load_tracker(args.tracker)
        print_summary(tracker)
    elif args.cmd == "run":
        run_pipeline(args.tracker, args.datadir,
                     stages=args.stages,
                     max_objects=args.max_objects,
                     max_download=args.max_download)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
