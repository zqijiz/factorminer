"""Tests for the library_io utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import export_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def empty_library():
    return FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)

def _make_factor(
    name="test",
    formula="Neg($close)",
    ic=0.06,
    signals=None,
    rng=None,
    M=20,
    T=60,
):
    """Helper to create a Factor with random signals."""
    if signals is None and rng is not None:
        signals = rng.normal(0, 1, (M, T))
    return Factor(
        id=0,
        name=name,
        formula=formula,
        category="test",
        ic_mean=ic,
        icir=1.0,
        ic_win_rate=0.6,
        max_correlation=0.0,
        batch_number=1,
        signals=signals,
        admission_date="2024-01-01 00:00:00",
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_export_csv(empty_library, tmp_path, rng):
    """Test exporting factors to CSV."""
    # Populate the library with some factors
    f1 = _make_factor(name="factor_1", ic=0.05, rng=rng)
    f2 = _make_factor(name="factor_2", formula="Add($open, $close)", ic=0.08, rng=rng)

    empty_library.admit_factor(f1)
    empty_library.admit_factor(f2)

    # Export to CSV
    csv_path = tmp_path / "factors.csv"
    export_csv(empty_library, csv_path)

    # Read the CSV back and verify
    assert csv_path.exists()

    with open(csv_path, newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames

        expected_fieldnames = [
            "ID", "Name", "Formula", "Category", "IC_Mean", "ICIR",
            "IC_Win_Rate", "Max_Correlation", "Batch", "Admission_Date"
        ]
        assert fieldnames == expected_fieldnames

        rows = list(reader)
        assert len(rows) == empty_library.size

        # Spot check the first row
        assert int(rows[0]["ID"]) == f1.id
        assert rows[0]["Name"] == "factor_1"
        assert rows[0]["Formula"] == "Neg($close)"
        assert float(rows[0]["IC_Mean"]) == pytest.approx(f1.ic_mean)
        assert int(rows[0]["Batch"]) == f1.batch_number
        assert rows[0]["Admission_Date"] == f1.admission_date
