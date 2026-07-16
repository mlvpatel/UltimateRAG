"""Load the bundled sample documents and images into rag-multimodal-2026.

Run this after the stack is up so anyone can try the system on the included
sample data. For a fully local, no cost run:

    make db-up
    ollama serve &
    ollama pull nomic-embed-text
    ollama pull moondream
    EMBEDDING_PROVIDER=ollama python scripts/load_sample_data.py

Text files are chunked and embedded; images are captioned by the vision model
and stored, exactly as an upload through the UI would be.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.db_utils import init_db  # noqa: E402
from src.worker.tasks import process_document  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
IMAGE_DIR = SAMPLE_DIR / "images"


def main() -> None:
    files = sorted(SAMPLE_DIR.glob("*.txt"))
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files += sorted(IMAGE_DIR.glob(pattern))
    if not files:
        print(f"No sample files found in {SAMPLE_DIR}")
        return
    init_db()
    print(
        f"Loading {len(files)} sample files from {SAMPLE_DIR.name}/. "
        "Images are captioned by the vision model, which can take a moment."
    )
    for path in files:
        result = process_document(str(path), path.name)
        print(f"  {path.name}: {result}")
    print(
        "Done. Open the UI and ask a question, see sample_data/README.md for examples."
    )


if __name__ == "__main__":
    main()
