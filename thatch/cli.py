"""
THATCH command-line interface.

Provides unified CLI entry points for all THATCH modules.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="thatch",
        description="THATCH: Transient Harvester for Archival Training and Classification from Hubble",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # crossmatch
    p_xmatch = subparsers.add_parser(
        "crossmatch", help="Cross-match transients against HST archive"
    )
    p_xmatch.add_argument(
        "--output", "-o", default="thatch_hst_transients.csv", help="Output CSV path"
    )
    p_xmatch.add_argument(
        "--radius", type=float, default=5.0, help="Search radius in arcseconds (default: 5.0)"
    )

    # harvest
    p_harvest = subparsers.add_parser("harvest", help="Download HST images for a transient")
    p_harvest.add_argument("name", help="Transient name")
    p_harvest.add_argument("--ra", type=float, required=True, help="RA in degrees")
    p_harvest.add_argument("--dec", type=float, required=True, help="Dec in degrees")
    p_harvest.add_argument("--radius", type=float, default=5.0, help="Search radius in arcseconds")
    p_harvest.add_argument("--outdir", "-o", default=".", help="Output directory")
    p_harvest.add_argument("--max-download", type=int, default=50, help="Max products to download")

    # photometry
    p_phot = subparsers.add_parser(
        "photometry", help="Run aperture photometry on downloaded images"
    )
    p_phot.add_argument("name", help="Transient name")
    p_phot.add_argument("--ra", type=float, required=True, help="RA in degrees")
    p_phot.add_argument("--dec", type=float, required=True, help="Dec in degrees")
    p_phot.add_argument("--datadir", default=".", help="Directory with FITS images")
    p_phot.add_argument("--aperture", type=int, default=5, help="Aperture radius in pixels")

    # spectra
    p_spec = subparsers.add_parser("spectra", help="Extract spectra for a transient")
    p_spec.add_argument("name", help="Transient name")
    p_spec.add_argument("--ra", type=float, required=True, help="RA in degrees")
    p_spec.add_argument("--dec", type=float, required=True, help="Dec in degrees")
    p_spec.add_argument("--outdir", "-o", default=".", help="Output directory")

    # cutouts
    p_cut = subparsers.add_parser("cutouts", help="Extract image cutouts")
    p_cut.add_argument("name", help="Transient name")
    p_cut.add_argument("--ra", type=float, required=True, help="RA in degrees")
    p_cut.add_argument("--dec", type=float, required=True, help="Dec in degrees")
    p_cut.add_argument("--datadir", default=".", help="Directory with FITS images")
    p_cut.add_argument("--size", type=float, default=5.0, help="Cutout size in arcsec")
    p_cut.add_argument("--output", "-o", help="Output HDF5 path")

    # catalog
    p_cat = subparsers.add_parser("catalog", help="Build unified catalog from processed data")
    p_cat.add_argument("datadir", help="Root data directory")
    p_cat.add_argument(
        "--format",
        choices=["parquet", "csv", "fits"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    # batch
    p_batch = subparsers.add_parser("batch", help="Run full pipeline on a list of transients")
    p_batch.add_argument("input_csv", help="CSV with columns: name, ra, dec, type")
    p_batch.add_argument("--outdir", "-o", default="data", help="Output directory")
    p_batch.add_argument("--max-download", type=int, default=50, help="Max images per object")
    p_batch.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip objects with existing photometry",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "crossmatch":
        from thatch.crossmatch import main as xmatch_main

        xmatch_main()

    elif args.command == "harvest":
        from thatch.harvest import harvest_object

        harvest_object(
            args.name,
            args.ra,
            args.dec,
            radius=args.radius,
            outdir=args.outdir,
            max_download=args.max_download,
        )

    elif args.command == "photometry":
        print(f"Running photometry for {args.name} at ({args.ra}, {args.dec})")
        print("Use run_photometry_batch.py for batch processing")

    elif args.command == "spectra":
        from thatch.spectra import query_spectroscopic_obs, download_extracted_spectra
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra=args.ra, dec=args.dec, unit="deg")
        obs_df = query_spectroscopic_obs(coord, name=args.name)
        if len(obs_df) > 0:
            download_extracted_spectra(obs_df, args.outdir)

    elif args.command == "cutouts":
        from thatch.cutouts import extract_cutouts_for_object, save_cutouts_hdf5

        cutouts = extract_cutouts_for_object(args.datadir, args.ra, args.dec, size_arcsec=args.size)
        if cutouts and args.output:
            save_cutouts_hdf5(cutouts, args.output, object_name=args.name)
        print(f"Extracted {len(cutouts)} cutouts")

    elif args.command == "catalog":
        from thatch.catalog import build_catalog, catalog_summary, save_catalog

        catalog = build_catalog(args.datadir)
        catalog_summary(catalog)
        if args.format == "parquet":
            save_catalog(catalog, f"{args.datadir}/thatch_catalog.parquet")
        elif args.format == "csv":
            catalog.to_csv(f"{args.datadir}/thatch_catalog.csv", index=False)
        elif args.format == "fits":
            from astropy.table import Table

            Table.from_pandas(catalog).write(
                f"{args.datadir}/thatch_catalog.fits", format="fits", overwrite=True
            )

    elif args.command == "batch":
        print(f"Batch processing from {args.input_csv}")
        print("Use run_photometry_batch.py directly for now")


if __name__ == "__main__":
    main()
