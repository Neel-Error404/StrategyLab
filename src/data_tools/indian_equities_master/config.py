from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, Iterable, List, Optional

import yaml


DEFAULT_CONFIG_PATH = pathlib.Path("config/indian_equities_master.yaml")


@dataclasses.dataclass
class ScreenerConfig:
    region: str = "IN"
    quote_types: List[str] = dataclasses.field(
        default_factory=lambda: ["EQUITY", "ETF"]
    )
    exchanges: List[str] = dataclasses.field(default_factory=lambda: ["NSE", "BSE"])
    page_size: int = 250
    request_timeout: int = 15
    max_pages: int = 200  # safety guard (~50k records)


@dataclasses.dataclass
class EnrichConfig:
    batch_size: int = 25
    price_history_days: int = 5
    price_history_interval: str = "1d"
    enable_options_lookup: bool = False
    request_pause_seconds: float = 1.0
    max_retries: int = 3


@dataclasses.dataclass
class StorageConfig:
    output_csv: pathlib.Path = pathlib.Path("data/indian_equities_master.csv")
    tmp_dir: pathlib.Path = pathlib.Path("data/tmp")


@dataclasses.dataclass
class PipelineConfig:
    screener: ScreenerConfig = dataclasses.field(default_factory=ScreenerConfig)
    enrich: EnrichConfig = dataclasses.field(default_factory=EnrichConfig)
    storage: StorageConfig = dataclasses.field(default_factory=StorageConfig)


def load_config(config_path: Optional[pathlib.Path]) -> PipelineConfig:
    """
    Load pipeline configuration from YAML. Falls back to defaults if the file is missing.
    """
    base = PipelineConfig()

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return base

    with config_path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle) or {}

    _apply_overrides(base, raw)
    return base


def _apply_overrides(target: PipelineConfig, overrides: Dict[str, Any]) -> None:
    for section, values in overrides.items():
        if not hasattr(target, section):
            continue
        section_obj = getattr(target, section)
        if dataclasses.is_dataclass(section_obj) and isinstance(values, dict):
            for key, value in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
        elif isinstance(values, dict):
            for key, value in values.items():
                setattr(section_obj, key, value)

    # Normalise path types
    target.storage.output_csv = pathlib.Path(target.storage.output_csv)
    target.storage.tmp_dir = pathlib.Path(target.storage.tmp_dir)
