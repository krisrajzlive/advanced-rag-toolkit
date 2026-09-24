"""Download Anthropic's "Detecting and Countering Misuse" threat report into data/pdf/.

    python scripts/download_anthropic_pdf.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://www-cdn.anthropic.com/e50be2e51e7695dc4b1366a37a245a597377d3b5/"
    "Anthropic-Detecting-and-countering-091026.pdf"
)
DEST = Path(__file__).resolve().parent.parent / "data" / "pdf" / (
    "anthropic-detecting-countering-misuse-2025-09.pdf"
)


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} -> {DEST}")
    urlretrieve(URL, DEST)
    print(f"Saved {DEST.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
