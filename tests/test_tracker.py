"""Tests for thatch.tracker."""
import pandas as pd
import pytest

from thatch.tracker import (
    init_tracker, load_tracker, update_status, get_next_objects,
    print_summary, Status,
)


@pytest.fixture
def sample_crossmatch(tmp_path):
    """Create a sample cross-match CSV."""
    df = pd.DataFrame([
        {"name": "SN2011fe", "ra": 210.774, "dec": 54.274, "type": "SN Ia",
         "z": 0.001, "discovery_mjd": 55797, "n_imaging": 217,
         "n_spectroscopy": 75, "n_programs": 25},
        {"name": "AT2017gfo", "ra": 197.450, "dec": -23.381, "type": "Kilonova",
         "z": 0.010, "discovery_mjd": 57983, "n_imaging": 198,
         "n_spectroscopy": 31, "n_programs": 23},
    ])
    csv_path = str(tmp_path / "xmatch.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, tmp_path


def test_init_tracker(sample_crossmatch):
    csv_path, tmp_path = sample_crossmatch
    tracker_path = str(tmp_path / "tracker.parquet")
    tracker = init_tracker(csv_path, tracker_path)

    assert len(tracker) == 2
    assert all(tracker["status"] == Status.QUEUED.value)
    assert "last_updated" in tracker.columns


def test_load_tracker(sample_crossmatch):
    csv_path, tmp_path = sample_crossmatch
    tracker_path = str(tmp_path / "tracker.parquet")
    init_tracker(csv_path, tracker_path)

    loaded = load_tracker(tracker_path)
    assert len(loaded) == 2


def test_update_status(sample_crossmatch):
    csv_path, tmp_path = sample_crossmatch
    tracker_path = str(tmp_path / "tracker.parquet")
    tracker = init_tracker(csv_path, tracker_path)

    tracker = update_status(tracker, "SN2011fe", Status.DOWNLOADING,
                            tracker_path, n_images_downloaded=42)

    assert tracker.loc[tracker["name"] == "SN2011fe", "status"].values[0] == Status.DOWNLOADING.value
    assert tracker.loc[tracker["name"] == "SN2011fe", "n_images_downloaded"].values[0] == 42

    # AT2017gfo should still be queued
    assert tracker.loc[tracker["name"] == "AT2017gfo", "status"].values[0] == Status.QUEUED.value


def test_get_next_objects(sample_crossmatch):
    csv_path, tmp_path = sample_crossmatch
    tracker_path = str(tmp_path / "tracker.parquet")
    tracker = init_tracker(csv_path, tracker_path)

    next_objs = get_next_objects(tracker, Status.QUEUED, n=1)
    assert len(next_objs) == 1


def test_print_summary(sample_crossmatch, capsys):
    csv_path, tmp_path = sample_crossmatch
    tracker_path = str(tmp_path / "tracker.parquet")
    tracker = init_tracker(csv_path, tracker_path)

    print_summary(tracker)
    captured = capsys.readouterr()
    assert "THATCH" in captured.out
    assert "queued" in captured.out
