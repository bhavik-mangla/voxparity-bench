from pathlib import Path

from typer.testing import CliRunner

from voxparity.cli import app, load_item

ITEMS_DIR = Path(__file__).parent.parent / "items"


def test_shipped_items_are_valid():
    files = sorted((ITEMS_DIR / "pilot").rglob("*.yaml"))
    assert files, "no pilot items found"
    for f in files:
        item = load_item(f)
        assert item.id == f.stem


def test_cli_validate_passes_on_shipped_items():
    result = CliRunner().invoke(app, ["items", "validate", str(ITEMS_DIR)])
    assert result.exit_code == 0, result.output


def test_cli_stats_runs():
    result = CliRunner().invoke(app, ["items", "stats", str(ITEMS_DIR)])
    assert result.exit_code == 0, result.output
    assert "items: " in result.output
