from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def output_path(path):
    """Reject output paths or symlinks escaping this checkout."""
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = ROOT / result
    result = result.resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Output must be inside {ROOT}: {result}")
    return result
