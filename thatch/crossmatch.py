#!/usr/bin/env python
"""
thatch-crossmatch: Find all HST-observed transients by cross-matching
known transient catalogs against the MAST HST archive.

Queries the Transient Name Server (TNS) and Open Supernova Catalog (OSC)
for known transients, then checks which have HST observations within
a configurable search radius.

Usage:
    conda activate hustle
    python thatch_crossmatch.py
"""

import os
import time
import warnings
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Observations

warnings.filterwarnings("ignore")

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "..", "data")
os.makedirs(DATADIR, exist_ok=True)

# Known transient samples to cross-match
# Start with a curated list of well-known HST-observed transients
# This will be extended to full TNS/OSC queries

KNOWN_TRANSIENTS = [
    # Type Ia supernovae
    {
        "name": "SN2011fe",
        "ra": 210.77420,
        "dec": 54.27370,
        "type": "SN Ia",
        "z": 0.001,
        "discovery_mjd": 55796.7,
    },
    {
        "name": "SN2014J",
        "ra": 148.92550,
        "dec": 69.67390,
        "type": "SN Ia",
        "z": 0.001,
        "discovery_mjd": 56681.0,
    },
    {
        "name": "SN2012cg",
        "ra": 186.80290,
        "dec": 9.42030,
        "type": "SN Ia",
        "z": 0.001,
        "discovery_mjd": 56066.0,
    },
    {
        "name": "SN2012fr",
        "ra": 54.23870,
        "dec": -36.12620,
        "type": "SN Ia",
        "z": 0.005,
        "discovery_mjd": 56233.0,
    },
    {
        "name": "SN2005cf",
        "ra": 219.34460,
        "dec": -7.41290,
        "type": "SN Ia",
        "z": 0.006,
        "discovery_mjd": 53519.0,
    },
    {
        "name": "SN2021aefx",
        "ra": 64.63280,
        "dec": -44.53910,
        "type": "SN Ia",
        "z": 0.005,
        "discovery_mjd": 59534.0,
    },
    {
        "name": "SN2018gv",
        "ra": 124.29540,
        "dec": -9.58240,
        "type": "SN Ia",
        "z": 0.005,
        "discovery_mjd": 58135.0,
    },
    {
        "name": "SN1994D",
        "ra": 186.63710,
        "dec": 7.70340,
        "type": "SN Ia",
        "z": 0.001,
        "discovery_mjd": 49427.0,
    },
    {
        "name": "SN1972E",
        "ra": 196.32920,
        "dec": -43.37580,
        "type": "SN Ia",
        "z": 0.002,
        "discovery_mjd": 41433.0,
    },
    {
        "name": "SN2006X",
        "ra": 185.72458,
        "dec": 15.80889,
        "type": "SN Ia",
        "z": 0.005,
        "discovery_mjd": 53770.0,
    },
    # SN Refsdal (lensed SN Ia-pec)
    {
        "name": "SN_Refsdal",
        "ra": 177.39840,
        "dec": 22.39530,
        "type": "SN Ia-pec (lensed)",
        "z": 1.49,
        "discovery_mjd": 56988.0,
    },
    # Core-collapse supernovae
    {
        "name": "SN1987A",
        "ra": 83.86670,
        "dec": -69.26960,
        "type": "SN II-pec",
        "z": 0.001,
        "discovery_mjd": 46849.0,
    },
    {
        "name": "SN1993J",
        "ra": 148.85420,
        "dec": 69.01710,
        "type": "SN IIb",
        "z": 0.001,
        "discovery_mjd": 49074.0,
    },
    {
        "name": "SN2005cs",
        "ra": 202.46990,
        "dec": 47.17630,
        "type": "SN IIP",
        "z": 0.002,
        "discovery_mjd": 53549.0,
    },
    {
        "name": "SN2008bk",
        "ra": 359.44750,
        "dec": -32.56370,
        "type": "SN IIP",
        "z": 0.001,
        "discovery_mjd": 54543.0,
    },
    {
        "name": "SN2013ej",
        "ra": 24.17420,
        "dec": 15.75680,
        "type": "SN IIP",
        "z": 0.002,
        "discovery_mjd": 56497.0,
    },
    {
        "name": "SN2017eaw",
        "ra": 205.43090,
        "dec": 58.03150,
        "type": "SN IIP",
        "z": 0.001,
        "discovery_mjd": 57887.0,
    },
    {
        "name": "SN2023ixf",
        "ra": 210.91075,
        "dec": 54.31169,
        "type": "SN II",
        "z": 0.001,
        "discovery_mjd": 60083.0,
    },
    # Ic-BL / GRB-SNe
    {
        "name": "SN1998bw",
        "ra": 290.85330,
        "dec": -23.76970,
        "type": "SN Ic-BL",
        "z": 0.009,
        "discovery_mjd": 50930.0,
    },
    {
        "name": "SN2003dh",
        "ra": 163.07458,
        "dec": 21.53139,
        "type": "SN Ic-BL",
        "z": 0.169,
        "discovery_mjd": 52727.0,
    },
    # Superluminous supernovae
    {
        "name": "SN2015bn",
        "ra": 175.11250,
        "dec": 0.84000,
        "type": "SLSN-I",
        "z": 0.110,
        "discovery_mjd": 57058.0,
    },
    {
        "name": "SN2018bsz",
        "ra": 243.71250,
        "dec": -33.87580,
        "type": "SLSN-I",
        "z": 0.027,
        "discovery_mjd": 58257.0,
    },
    {
        "name": "iPTF16eh",
        "ra": 132.39375,
        "dec": 48.72444,
        "type": "SLSN-I",
        "z": 0.427,
        "discovery_mjd": 57400.0,
    },
    # Kilonovae
    {
        "name": "AT2017gfo",
        "ra": 197.45037,
        "dec": -23.38148,
        "type": "Kilonova",
        "z": 0.010,
        "discovery_mjd": 57982.5,
    },
    {
        "name": "GRB130603B",
        "ra": 172.20083,
        "dec": 17.06253,
        "type": "Kilonova candidate",
        "z": 0.356,
        "discovery_mjd": 56447.0,
    },
    # Tidal Disruption Events
    {
        "name": "Sw_J1644+57",
        "ra": 251.20530,
        "dec": 57.58090,
        "type": "TDE (jetted)",
        "z": 0.354,
        "discovery_mjd": 55648.0,
    },
    {
        "name": "AT2022cmc",
        "ra": 203.18590,
        "dec": 33.77020,
        "type": "TDE (jetted)",
        "z": 1.193,
        "discovery_mjd": 59621.0,
    },
    {
        "name": "ASASSN-14li",
        "ra": 192.06340,
        "dec": 17.77460,
        "type": "TDE",
        "z": 0.021,
        "discovery_mjd": 56983.0,
    },
    {
        "name": "AT2018dyb",
        "ra": 99.63833,
        "dec": -16.28694,
        "type": "TDE",
        "z": 0.018,
        "discovery_mjd": 58300.0,
    },
    # Luminous Red Novae / Intermediate Luminosity Transients
    {
        "name": "NGC4490-OT",
        "ra": 187.65170,
        "dec": 41.64250,
        "type": "LRN",
        "z": 0.002,
        "discovery_mjd": 56782.0,
    },
    # Ca-rich transients
    {
        "name": "SN2005E",
        "ra": 55.05458,
        "dec": -4.79622,
        "type": "Ca-rich",
        "z": 0.011,
        "discovery_mjd": 53386.0,
    },
    # FBOTs
    {
        "name": "AT2018cow",
        "ra": 244.00092,
        "dec": 22.26800,
        "type": "FBOT",
        "z": 0.014,
        "discovery_mjd": 58285.0,
    },
    # More Type Ia from large HST programs (SHOES, RAISIN, See Change)
    {
        "name": "SN2007af",
        "ra": 210.52790,
        "dec": -0.39070,
        "type": "SN Ia",
        "z": 0.005,
        "discovery_mjd": 54165.0,
    },
    {
        "name": "SN2011by",
        "ra": 175.31400,
        "dec": 55.32770,
        "type": "SN Ia",
        "z": 0.003,
        "discovery_mjd": 55680.0,
    },
    {
        "name": "SN2012ht",
        "ra": 180.47610,
        "dec": 3.66410,
        "type": "SN Ia",
        "z": 0.004,
        "discovery_mjd": 56281.0,
    },
    {
        "name": "SN2015F",
        "ra": 112.01500,
        "dec": -69.49140,
        "type": "SN Ia",
        "z": 0.005,
        "discovery_mjd": 57094.0,
    },
    {
        "name": "SN2019np",
        "ra": 154.98380,
        "dec": 29.31240,
        "type": "SN Ia",
        "z": 0.005,
        "discovery_mjd": 58493.0,
    },
]


def check_hst_coverage(transient, search_radius_arcsec=5.0, max_retries=3):
    """Check if a transient has HST observations in MAST.

    Parameters
    ----------
    transient : dict
        Transient info with ra, dec, name.
    search_radius_arcsec : float
        Search radius in arcseconds.

    Returns
    -------
    dict or None
        Summary of HST observations, or None if no HST data.
    """
    coord = SkyCoord(ra=transient["ra"], dec=transient["dec"], unit="deg")

    for attempt in range(max_retries):
        try:
            obs = Observations.query_region(coord, radius=search_radius_arcsec * u.arcsec)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    MAST error, retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    MAST failed after {max_retries} attempts: {e}")
                return None

    hst = obs[obs["obs_collection"] == "HST"]
    if len(hst) == 0:
        return None

    # Count imaging vs spectroscopy
    n_imaging = 0
    n_spectroscopy = 0
    filters_imaging = set()
    filters_spectroscopy = set()
    instruments = set()
    programs = set()
    mjd_min, mjd_max = 1e6, 0

    grism_names = [
        "G102",
        "G141",
        "G130M",
        "G140L",
        "G230L",
        "G230LB",
        "G430L",
        "G750L",
        "G800L",
        "G280",
    ]

    for row in hst:
        filt = str(row["filters"])
        dtype = str(row["dataproduct_type"]).lower()
        inst = str(row["instrument_name"])

        is_spec = "spectrum" in dtype or "spectr" in dtype
        is_grism = any(g in filt for g in grism_names)

        if is_spec or is_grism:
            n_spectroscopy += 1
            filters_spectroscopy.add(filt)
        elif "image" in dtype:
            # Skip grism from imaging count
            if not any(g in filt for g in grism_names):
                n_imaging += 1
                filters_imaging.add(filt)

        instruments.add(inst)
        programs.add(str(row["proposal_id"]))

        try:
            t_min = float(row["t_min"])
            t_max = float(row["t_max"])
            mjd_min = min(mjd_min, t_min)
            mjd_max = max(mjd_max, t_max)
        except (ValueError, TypeError):
            pass

    return {
        "name": transient["name"],
        "ra": transient["ra"],
        "dec": transient["dec"],
        "type": transient["type"],
        "z": transient.get("z", None),
        "discovery_mjd": transient.get("discovery_mjd", None),
        "n_hst_total": len(hst),
        "n_imaging": n_imaging,
        "n_spectroscopy": n_spectroscopy,
        "n_filters_imaging": len(filters_imaging),
        "n_filters_spectroscopy": len(filters_spectroscopy),
        "filters_imaging": ";".join(sorted(filters_imaging)),
        "filters_spectroscopy": ";".join(sorted(filters_spectroscopy)),
        "instruments": ";".join(sorted(instruments)),
        "n_programs": len(programs),
        "programs": ";".join(sorted(programs)),
        "mjd_min": mjd_min if mjd_min < 1e6 else None,
        "mjd_max": mjd_max if mjd_max > 0 else None,
    }


def main():
    print("=" * 70)
    print("THATCH Cross-Match: Finding HST-Observed Transients")
    print("=" * 70)

    print(f"\nChecking {len(KNOWN_TRANSIENTS)} known transients against MAST...")

    results = []
    for i, t in enumerate(KNOWN_TRANSIENTS):
        print(
            f"  [{i + 1}/{len(KNOWN_TRANSIENTS)}] {t['name']:20s} ({t['type']})...",
            end="",
            flush=True,
        )
        result = check_hst_coverage(t)
        if result:
            results.append(result)
            print(
                f" {result['n_imaging']}img + {result['n_spectroscopy']}spec "
                f"in {result['n_programs']} programs"
            )
        else:
            print(" no HST data")

        # Be polite to MAST
        time.sleep(0.5)

    # Save results
    df = pd.DataFrame(results)
    outpath = os.path.join(DATADIR, "thatch_hst_transients.csv")
    df.to_csv(outpath, index=False)
    print(f"\nSaved {len(df)} HST-observed transients to {outpath}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("THATCH Cross-Match Summary")
    print(f"{'=' * 70}")
    print(f"  Transients checked: {len(KNOWN_TRANSIENTS)}")
    print(f"  With HST data: {len(df)}")
    print(f"  Total HST observations: {df['n_hst_total'].sum()}")
    print(f"  Total imaging obs: {df['n_imaging'].sum()}")
    print(f"  Total spectroscopic obs: {df['n_spectroscopy'].sum()}")

    # By type
    print("\n  By transient type:")
    for ttype, grp in df.groupby("type"):
        print(
            f"    {ttype:25s}: {len(grp)} objects, "
            f"{grp['n_imaging'].sum()} imaging, "
            f"{grp['n_spectroscopy'].sum()} spectroscopy"
        )

    # Top objects by number of observations
    print("\n  Top 10 most-observed transients:")
    top = df.nlargest(10, "n_hst_total")
    for _, row in top.iterrows():
        print(
            f"    {row['name']:20s} {row['type']:20s} "
            f"{row['n_hst_total']:4d} obs ({row['n_imaging']}img + {row['n_spectroscopy']}spec) "
            f"in {row['n_programs']} programs"
        )


if __name__ == "__main__":
    main()
