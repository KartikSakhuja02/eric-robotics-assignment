from pathlib import Path


def fixture_root():
    root = Path(__file__).resolve().parents[1] / "outputs/verification"
    root.mkdir(parents=True, exist_ok=True)
    return root
