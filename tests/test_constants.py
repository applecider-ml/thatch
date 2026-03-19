"""Tests for thatch.constants."""

from thatch.constants import (
    APERTURE_CORRECTIONS_5PX,
    APERTURE_CORRECTIONS_4PX,
    ALL_SPECTROSCOPIC,
)


def test_aperture_corrections_5px_values():
    """Aperture corrections should be between 0 and 1."""
    for filt, ee in APERTURE_CORRECTIONS_5PX.items():
        assert 0.5 < ee < 1.0, f"{filt}: {ee} out of range"


def test_aperture_corrections_4px_values():
    """4px aperture corrections should be smaller than 5px."""
    for filt in APERTURE_CORRECTIONS_4PX:
        ee4 = APERTURE_CORRECTIONS_4PX[filt]
        assert 0.5 < ee4 < 1.0, f"{filt}: {ee4} out of range"
        if filt in APERTURE_CORRECTIONS_5PX:
            ee5 = APERTURE_CORRECTIONS_5PX[filt]
            assert ee4 <= ee5, f"{filt}: 4px EE ({ee4}) > 5px EE ({ee5})"


def test_spectroscopic_list_nonempty():
    """Spectroscopic grating list should have entries."""
    assert len(ALL_SPECTROSCOPIC) > 10


def test_known_filters_present():
    """Common HST filters should be in the aperture correction tables."""
    for filt in ["F110W", "F160W", "F606W", "F814W"]:
        assert filt in APERTURE_CORRECTIONS_5PX, f"{filt} missing from 5px table"
