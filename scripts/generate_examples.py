"""Write the clickable synthetic demo images into examples/.

Run: .venv/bin/python scripts/generate_examples.py
These images are for demonstration/testing only, not training data.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspection.synth import EXAMPLES, example_png

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for kind, spec in EXAMPLES.items():
        for variant in range(spec["variants"]):
            target = OUT / f"{kind}_{variant}.png"
            target.write_bytes(example_png(kind, variant))
            print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
