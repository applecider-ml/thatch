"""Tests for thatch.catalog."""

import pandas as pd
import pytest

from thatch.catalog import build_catalog, catalog_summary, save_catalog, load_catalog


@pytest.fixture
def sample_data(tmp_path):
    """Create sample photometry CSVs for testing."""
    obj_dir = tmp_path / "TestSN"
    obj_dir.mkdir()

    df = pd.DataFrame(
        {
            "filename": ["test1.fits", "test2.fits", "test3.fits"],
            "mjd": [55000.0, 55001.0, 55002.0],
            "delta_t_days": [0.0, 1.0, 2.0],
            "filter": ["F110W", "F160W", "F110W"],
            "instrument": ["WFC3/IR", "WFC3/IR", "WFC3/IR"],
            "exptime_s": [300.0, 300.0, 300.0],
            "count_rate": [10.0, 8.0, 5.0],
            "count_rate_err": [0.5, 0.4, 0.3],
            "ab_mag": [24.0, 24.5, 25.0],
            "ab_mag_err": [0.05, 0.06, 0.07],
            "x_pix": [500.0, 500.0, 500.0],
            "y_pix": [500.0, 500.0, 500.0],
            "photflam": [1.5e-20, 1.9e-20, 1.5e-20],
            "photplam": [11534.0, 15369.0, 11534.0],
            "bg_median": [1.0, 1.0, 1.0],
            "bg_std": [0.1, 0.1, 0.1],
        }
    )
    df.to_csv(obj_dir / "TestSN_photometry.csv", index=False)
    return tmp_path


def test_build_catalog(sample_data):
    """build_catalog should find and load photometry CSVs."""
    catalog = build_catalog(str(sample_data))
    assert len(catalog) == 3
    assert "object" in catalog.columns
    assert catalog["object"].iloc[0] == "TestSN"


def test_build_catalog_empty(tmp_path):
    """build_catalog should return empty DataFrame for empty directory."""
    catalog = build_catalog(str(tmp_path))
    assert len(catalog) == 0


def test_catalog_columns(sample_data):
    """Catalog should have standardized columns."""
    catalog = build_catalog(str(sample_data))
    required = ["object", "mjd", "filter", "ab_mag", "ab_mag_err"]
    for col in required:
        assert col in catalog.columns, f"Missing column: {col}"


def test_catalog_summary(sample_data, capsys):
    """catalog_summary should print without errors."""
    catalog = build_catalog(str(sample_data))
    catalog_summary(catalog)
    captured = capsys.readouterr()
    assert "THATCH" in captured.out
    assert "TestSN" in captured.out


def test_save_load_parquet(sample_data, tmp_path):
    """Catalog should round-trip through Parquet."""
    catalog = build_catalog(str(sample_data))
    outpath = str(tmp_path / "test.parquet")
    save_catalog(catalog, outpath)

    loaded = load_catalog(outpath)
    assert len(loaded) == len(catalog)
    assert set(loaded.columns) == set(catalog.columns)
