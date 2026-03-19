#!/usr/bin/env python
"""
THATCH-HARVEST: Query MAST for HST observations of known transients.

This script:
1. Gathers transient coordinates from multiple catalogs (TNS, OSC, hardcoded showcase objects)
2. Cross-matches against the HST archive via astroquery.mast
3. Produces a summary of HST-observed transients with observation details
4. Generates example light curve plots for showcase objects

Usage:
    conda activate thatch
    python hustle_harvest.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Observations

warnings.filterwarnings("ignore")

# Output directory
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTDIR, exist_ok=True)
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "Figures")
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================
# 1. SHOWCASE TRANSIENTS WITH KNOWN HST OBSERVATIONS
# ============================================================
# These are transients we KNOW have extensive HST data,
# ideal for demonstration and pipeline validation.

SHOWCASE_TRANSIENTS = [
    # Kilonovae
    {
        "name": "AT2017gfo",
        "ra": 197.45037,
        "dec": -23.38148,
        "type": "Kilonova",
        "redshift": 0.00980,
        "notes": "GW170817 counterpart; gold-standard kilonova",
    },
    # Type Ia SNe with extensive HST coverage
    {
        "name": "SN2011fe",
        "ra": 210.77420,
        "dec": 54.27370,
        "type": "SN Ia",
        "redshift": 0.000804,
        "notes": "M101; 10 STIS epochs, 11.5yr baseline",
    },
    {
        "name": "SN2014J",
        "ra": 148.92554,
        "dec": 69.67387,
        "type": "SN Ia",
        "redshift": 0.000677,
        "notes": "M82; 7 WFC3 + 10 STIS epochs",
    },
    # Lensed SNe
    {
        "name": "SN Refsdal",
        "ra": 177.39792,
        "dec": 22.39569,
        "type": "SN II (lensed)",
        "redshift": 1.489,
        "notes": "First multiply-lensed SN; MACS J1149",
    },
    {
        "name": "iPTF16geu",
        "ra": 333.17054,
        "dec": 16.74564,
        "type": "SN Ia (lensed)",
        "redshift": 0.409,
        "notes": "First resolved strongly-lensed SN Ia",
    },
    # Jetted TDEs
    {
        "name": "Sw J1644+57",
        "ra": 251.20529,
        "dec": 57.58089,
        "type": "TDE (jetted)",
        "redshift": 0.354,
        "notes": "First jetted TDE; extensive HST astrometry",
    },
    {
        "name": "AT2022cmc",
        "ra": 207.07004,
        "dec": 33.77079,
        "type": "TDE (jetted)",
        "redshift": 1.193,
        "notes": "Optically-discovered jetted TDE",
    },
    # SLSNe
    {
        "name": "SN2015bn",
        "ra": 174.09413,
        "dec": 0.22619,
        "type": "SLSN-I",
        "redshift": 0.1136,
        "notes": "Well-studied SLSN with HST imaging",
    },
    # GRB-SNe
    {
        "name": "SN1998bw",
        "ra": 290.96671,
        "dec": -52.63558,
        "type": "SN Ic-BL",
        "redshift": 0.00867,
        "notes": "GRB 980425 counterpart; prototype GRB-SN",
    },
    # SN 1987A
    {
        "name": "SN1987A",
        "ra": 83.86667,
        "dec": -69.26972,
        "type": "SN II-pec",
        "redshift": 0.000927,
        "notes": "LMC; 25+ years HST monitoring",
    },
]


def query_hst_observations(ra, dec, radius_arcsec=10.0):
    """Query MAST for all HST observations near a position."""
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    radius_deg = radius_arcsec / 3600.0

    try:
        obs_table = Observations.query_region(
            coord,
            radius=radius_deg * u.deg,
        )
    except Exception as e:
        print(f"  MAST query failed: {e}")
        return None

    if obs_table is None or len(obs_table) == 0:
        return None

    # Filter to HST only
    hst_mask = obs_table["obs_collection"] == "HST"
    hst_obs = obs_table[hst_mask]

    if len(hst_obs) == 0:
        return None

    return hst_obs


def summarize_hst_obs(hst_obs):
    """Summarize HST observations into a compact dict."""
    if hst_obs is None:
        return None

    def safe_unique(col):
        """Get unique non-masked values from a column."""
        vals = []
        for v in col:
            try:
                s = str(v)
                if s and s != "--" and s != "N/A":
                    vals.append(s)
            except Exception:
                continue
        return list(set(vals))

    instruments = safe_unique(hst_obs["instrument_name"])
    filters = safe_unique(hst_obs["filters"])
    n_obs = len(hst_obs)
    programs = safe_unique(hst_obs["proposal_id"])

    # Get time range
    try:
        t_min = float(np.nanmin(hst_obs["t_min"]))
        t_max = float(np.nanmax(hst_obs["t_max"]))
    except (ValueError, TypeError):
        t_min, t_max = None, None

    return {
        "n_observations": n_obs,
        "instruments": instruments,
        "filters": filters,
        "programs": programs,
        "t_min_mjd": t_min,
        "t_max_mjd": t_max,
    }


def query_tns_recent(limit=100):
    """
    Query the Transient Name Server for recent classified transients.
    Note: TNS requires API key for bulk access. This fetches from the
    public search page as a demonstration.
    For full THATCH deployment, register for TNS API access.
    """
    print("\n--- Querying TNS (public search, limited) ---")
    # TNS public API requires authentication for bulk queries.
    # For the demonstration, we use the OSC instead.
    print("  TNS requires API key for bulk access.")
    print("  For full deployment, register at https://www.wis-tns.org/")
    return []


def query_osc(limit=500):
    """
    Query the Open Supernova Catalog (sne.space) for classified SNe
    with coordinates.
    """
    print("\n--- Querying Open Supernova Catalog ---")

    url = "https://sne.space/astrocats/astrocats/supernovae/output/catalog.min.json"
    alt_url = "https://sne.space/sne/catalog.json"

    # The OSC API allows querying individual objects or the full catalog.
    # For demonstration, query a curated list of well-known SNe.
    # The full catalog is ~300MB; for the proposal demo we use targeted queries.

    well_known_sne = [
        "SN2011fe",
        "SN2014J",
        "SN2012cg",
        "SN2013dy",
        "SN2005cf",
        "SN2003du",
        "SN2012fr",
        "SN2017cbv",
        "SN2018oh",
        "SN2019ein",
        "SN2011by",
        "SN2016coj",
        "SN2017erp",
        "SN2013aa",
        "SN2012ht",
        "SN2009ig",
        "SN2005df",
        "SN2015F",
    ]

    results = []
    for sn_name in well_known_sne:
        try:
            resp = requests.get(
                f"https://sne.space/sne/{sn_name}.json",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if sn_name in data:
                    obj = data[sn_name]
                    ra = obj.get("ra", [{}])[0].get("value") if obj.get("ra") else None
                    dec = obj.get("dec", [{}])[0].get("value") if obj.get("dec") else None
                    z = obj.get("redshift", [{}])[0].get("value") if obj.get("redshift") else None
                    claimedtype = (
                        obj.get("claimedtype", [{}])[0].get("value")
                        if obj.get("claimedtype")
                        else None
                    )
                    if ra and dec:
                        # Convert RA from HMS to degrees if needed
                        try:
                            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                            results.append(
                                {
                                    "name": sn_name,
                                    "ra": coord.ra.deg,
                                    "dec": coord.dec.deg,
                                    "type": claimedtype or "Unknown",
                                    "redshift": float(z) if z else None,
                                    "notes": "From OSC",
                                }
                            )
                        except Exception:
                            pass
        except Exception as e:
            print(f"  Failed to query {sn_name}: {e}")
            continue

    print(f"  Retrieved {len(results)} SNe from OSC")
    return results


def build_transient_catalog():
    """Build combined transient catalog from all sources."""
    all_transients = list(SHOWCASE_TRANSIENTS)

    # Add OSC transients
    osc_transients = query_osc()
    # Deduplicate by name
    existing_names = {t["name"] for t in all_transients}
    for t in osc_transients:
        if t["name"] not in existing_names:
            all_transients.append(t)
            existing_names.add(t["name"])

    print(f"\nTotal transients in catalog: {len(all_transients)}")
    return all_transients


def cross_match_with_hst(transients):
    """Cross-match all transients with the HST archive."""
    print("\n=== Cross-matching with HST archive via MAST ===\n")

    results = []
    for i, t in enumerate(transients):
        name = t["name"]
        ra, dec = t["ra"], t["dec"]
        print(f"[{i + 1}/{len(transients)}] Querying {name} (RA={ra:.5f}, Dec={dec:.5f})...")

        hst_obs = query_hst_observations(ra, dec, radius_arcsec=5.0)
        summary = summarize_hst_obs(hst_obs)

        if summary and summary["n_observations"] > 0:
            print(
                f"  FOUND: {summary['n_observations']} HST obs, "
                f"{len(summary['programs'])} programs, "
                f"instruments: {summary['instruments']}"
            )
            t["hst_summary"] = summary
            t["hst_obs_table"] = hst_obs
            results.append(t)
        else:
            print("  No HST observations found")

    print(f"\n=== {len(results)}/{len(transients)} transients have HST data ===")
    return results


def get_hst_product_list(obs_table):
    """Get downloadable data products for a set of observations."""
    try:
        products = Observations.get_product_list(obs_table)
        return products
    except Exception as e:
        print(f"  Failed to get product list: {e}")
        return None


def extract_photometry_from_obs(obs_table, name):
    """
    Extract photometric measurements from HST observation metadata.
    This uses the observation-level metadata (MJD, filter, exposure time)
    as a proxy for a light curve. Full photometry extraction would require
    downloading and measuring images.
    """
    if obs_table is None or len(obs_table) == 0:
        return None

    # Filter to imaging observations (not spectroscopy)
    imaging_mask = np.array(
        [
            "IMAGE" in str(dt).upper() or "image" in str(dt).lower()
            for dt in obs_table["dataproduct_type"]
        ]
    )
    img_obs = obs_table[imaging_mask] if np.any(imaging_mask) else obs_table

    records = []
    for row in img_obs:
        try:
            records.append(
                {
                    "name": name,
                    "mjd_start": float(row["t_min"]) if row["t_min"] else None,
                    "mjd_end": float(row["t_max"]) if row["t_max"] else None,
                    "filter": str(row["filters"]),
                    "instrument": str(row["instrument_name"]),
                    "proposal_id": str(row["proposal_id"]),
                    "exposure_time": float(row["t_exptime"]) if row["t_exptime"] else None,
                    "target_name": str(row["target_name"]),
                    "obs_id": str(row["obs_id"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not records:
        return None

    df = pd.DataFrame(records)
    df = df.dropna(subset=["mjd_start"])
    df = df.sort_values("mjd_start")
    return df


def plot_observation_timeline(transient, outdir=None):
    """Plot the HST observation timeline for a transient."""
    if outdir is None:
        outdir = FIGDIR

    name = transient["name"]
    obs_table = transient.get("hst_obs_table")
    if obs_table is None:
        return

    df = extract_photometry_from_obs(obs_table, name)
    if df is None or len(df) == 0:
        return

    # Color by instrument
    instrument_colors = {
        "ACS/WFC": "C0",
        "ACS/HRC": "C1",
        "ACS/SBC": "C2",
        "WFC3/UVIS": "C3",
        "WFC3/IR": "C4",
        "STIS/CCD": "C5",
        "STIS/NUV-MAMA": "C6",
        "STIS/FUV-MAMA": "C7",
        "WFPC2": "C8",
        "NICMOS": "C9",
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Get unique filters and assign y-positions
    unique_filters = sorted(df["filter"].unique())
    filter_ypos = {f: i for i, f in enumerate(unique_filters)}

    for _, row in df.iterrows():
        instrument = row["instrument"]
        color = instrument_colors.get(instrument, "gray")
        y = filter_ypos[row["filter"]]
        ax.scatter(
            row["mjd_start"],
            y,
            c=color,
            s=40,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_yticks(range(len(unique_filters)))
    ax.set_yticklabels(unique_filters, fontsize=8)
    ax.set_xlabel("MJD", fontsize=12)
    ax.set_title(f"HST Observation Timeline: {name}", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Legend for instruments
    from matplotlib.lines import Line2D

    present_instruments = df["instrument"].unique()
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=instrument_colors.get(inst, "gray"),
            markersize=8,
            label=inst,
        )
        for inst in present_instruments
        if inst in instrument_colors
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(outdir, f"timeline_{name.replace(' ', '_')}.pdf")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    return fname


def create_summary_table(results):
    """Create a summary table of all HST-observed transients."""
    rows = []
    for t in results:
        s = t.get("hst_summary", {})
        rows.append(
            {
                "Name": t["name"],
                "Type": t["type"],
                "Redshift": t.get("redshift"),
                "N_HST_obs": s.get("n_observations", 0),
                "N_programs": len(s.get("programs", [])),
                "Instruments": "; ".join(s.get("instruments", [])),
                "Filters": "; ".join(sorted(s.get("filters", []))),
                "Programs": "; ".join(sorted(s.get("programs", []))),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("N_HST_obs", ascending=False)

    outfile = os.path.join(OUTDIR, "hst_transient_summary.csv")
    df.to_csv(outfile, index=False)
    print(f"\nSummary table saved to: {outfile}")
    print(f"\n{df.to_string(index=False)}")
    return df


def main():
    print("=" * 60)
    print("THATCH-HARVEST: HST Transient Light Curve Extraction")
    print("=" * 60)

    # Step 1: Build transient catalog
    transients = build_transient_catalog()

    # Step 2: Cross-match with HST archive
    hst_transients = cross_match_with_hst(transients)

    # Step 3: Create summary table
    summary_df = create_summary_table(hst_transients)

    # Step 4: Generate observation timeline plots for showcase objects
    print("\n--- Generating observation timeline plots ---")
    showcase_names = {t["name"] for t in SHOWCASE_TRANSIENTS}
    for t in hst_transients:
        if t["name"] in showcase_names:
            plot_observation_timeline(t)

    # Step 5: Save detailed observation tables
    print("\n--- Saving detailed observation tables ---")
    for t in hst_transients:
        name = t["name"]
        df = extract_photometry_from_obs(t.get("hst_obs_table"), name)
        if df is not None and len(df) > 0:
            fname = os.path.join(OUTDIR, f"obs_{name.replace(' ', '_')}.csv")
            df.to_csv(fname, index=False)

    print("\n" + "=" * 60)
    print("THATCH-HARVEST complete!")
    print(f"Results in: {OUTDIR}")
    print(f"Figures in: {FIGDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
