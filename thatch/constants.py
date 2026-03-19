"""
Shared constants for THATCH: aperture corrections, filter properties, etc.
"""

# Aperture corrections: encircled energy fraction within r=5 pixels.
# From WFC3 Instrument Handbook Table 7.7 (UVIS, 0.04"/pix)
# and Table 7.8 (IR, 0.13"/pix), and ACS Instrument Handbook.
APERTURE_CORRECTIONS_5PX = {
    # WFC3/UVIS (r=5px = 0.20")
    "F225W": 0.800, "F275W": 0.810, "F336W": 0.830,
    "F438W": 0.845, "F475W": 0.850, "F555W": 0.855,
    "F600LP": 0.855, "F606W": 0.855, "F625W": 0.855,
    "F775W": 0.850, "F814W": 0.845,
    # WFC3/IR (r=5px = 0.65")
    "F105W": 0.945, "F110W": 0.950, "F125W": 0.950,
    "F140W": 0.950, "F153M": 0.945, "F160W": 0.945,
    # ACS/WFC (r=5px = 0.25")
    "F435W": 0.835,
}

# Aperture corrections for r=4 pixels (used for SN-in-galaxy cases)
APERTURE_CORRECTIONS_4PX = {
    # WFC3/UVIS (r=4px = 0.16")
    "F225W": 0.740, "F275W": 0.755, "F336W": 0.775,
    "F438W": 0.795, "F467M": 0.800, "F469N": 0.800,
    "F475W": 0.800, "F475X": 0.800, "F547M": 0.810,
    "F555W": 0.810, "F600LP": 0.810, "F606W": 0.810,
    "F625W": 0.810, "F775W": 0.800, "F814W": 0.795,
    # WFC3/IR (r=4px = 0.52")
    "F105W": 0.915, "F110W": 0.920, "F125W": 0.920,
    "F140W": 0.920, "F153M": 0.915, "F160W": 0.910,
    # ACS/WFC (r=4px = 0.20")
    "F435W": 0.785, "F475W": 0.795, "F555W": 0.805,
    "F606W": 0.810, "F625W": 0.810, "F775W": 0.800,
    "F814W": 0.795, "F850LP": 0.790,
    # WFPC2/PC (r=4px = 0.18")
    "F439W": 0.770, "F555W": 0.790, "F606W": 0.795,
    "F814W": 0.780, "F1042M": 0.760,
}

# Spectroscopic configurations
STIS_GRATINGS = [
    "G140L", "G140M", "G230L", "G230LB", "G230MB",
    "G430L", "G430M", "G750L", "G750M",
]
WFC3_GRISMS = ["G102", "G141", "G280"]
ACS_GRISMS = ["G800L"]
COS_GRATINGS = ["G130M", "G160M", "G230L", "G140L"]
ALL_SPECTROSCOPIC = STIS_GRATINGS + WFC3_GRISMS + ACS_GRISMS + COS_GRATINGS
