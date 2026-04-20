"""Tests for library IO operations."""

from pathlib import Path

from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import export_formulas


def test_export_formulas(tmp_path: Path):
    """Test exporting formulas to a text file."""
    # Create a small mock library
    library = FactorLibrary()

    # Create some mock factors
    f1 = Factor(
        id=1,
        name="Test Factor 1",
        formula="Neg($close)",
        category="test",
        ic_mean=0.05,
        icir=1.0,
        ic_win_rate=0.55,
        max_correlation=0.1,
        batch_number=1,
        signals=None,
    )
    f2 = Factor(
        id=42,
        name="Test Factor 2",
        formula="CsRank($volume)",
        category="test",
        ic_mean=0.06,
        icir=1.2,
        ic_win_rate=0.6,
        max_correlation=0.2,
        batch_number=1,
        signals=None,
    )

    # Add directly to factors dict to avoid needing signals for admission
    library.factors[f1.id] = f1
    library.factors[f2.id] = f2

    # Path for exporting (inside a subdirectory to test mkdir)
    export_path = tmp_path / "exports" / "formulas.txt"

    # Run the export function
    export_formulas(library, export_path)

    # Verify the file was created
    assert export_path.exists()
    assert export_path.is_file()

    # Read the contents and verify
    content = export_path.read_text()
    lines = content.splitlines()

    # Check headers
    assert lines[0] == "# FactorMiner Library Formulas"
    assert lines[1] == "# ID | Name | Formula"
    assert lines[2] == f"# Total: {library.size} factors"
    assert lines[3].startswith("#---")

    # Check factor lines
    # library.list_factors() returns factors sorted by ID
    assert len(lines) == 6
    assert lines[4] == "0001 | Test Factor 1 | Neg($close)"
    assert lines[5] == "0042 | Test Factor 2 | CsRank($volume)"

def test_export_formulas_empty_library(tmp_path: Path):
    """Test exporting formulas when the library is empty."""
    library = FactorLibrary()

    export_path = tmp_path / "empty_formulas.txt"
    export_formulas(library, export_path)

    assert export_path.exists()
    content = export_path.read_text()
    lines = content.splitlines()

    # Only headers should exist
    assert len(lines) == 4
    assert lines[2] == "# Total: 0 factors"
