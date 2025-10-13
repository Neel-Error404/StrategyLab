from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import Dict, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)


def write_csv(frame: pd.DataFrame, output_path: pathlib.Path) -> None:
    """
    Write the canonical CSV atomically by using a temporary file swap.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_path.parent) as handle:
        temp_path = pathlib.Path(handle.name)
        frame.to_csv(handle, index=False)

    temp_path.replace(output_path)
    LOGGER.info("Dataset written to %s (%s rows).", output_path, len(frame))


def write_metadata(metadata: Dict[str, object], output_path: pathlib.Path) -> None:
    """
    Persist simple metadata alongside the CSV (`<output>.metadata.json`).
    """
    import json

    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
    LOGGER.info("Metadata written to %s.", metadata_path)
