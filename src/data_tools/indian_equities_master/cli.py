from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys
import time
from typing import Optional

import pandas as pd

from . import __version__
from .config import PipelineConfig, load_config
from .discovery import fetch_listings
from .enrichment import build_dataset
from .io_utils import write_csv, write_metadata
from .qa import run_validations
from yfinance import cache as yf_cache


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    config = load_config(pathlib.Path(args.config) if args.config else None)
    if args.output:
        config.storage.output_csv = pathlib.Path(args.output)
    if args.enable_options_lookup:
        config.enrich.enable_options_lookup = True

    try:
        _run_update(config, emit_metadata=not args.skip_metadata)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Aborted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Update failed: %s", exc)
        return 1

    return 0


def _run_update(config: PipelineConfig, emit_metadata: bool) -> None:
    start = time.time()
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("Indian Equities Master update started (v%s).", __version__)
    logger.info("="*70)
    logger.info("Output path: %s", config.storage.output_csv)

    cache_dir = config.storage.tmp_dir / "yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf_cache.set_cache_location(str(cache_dir))
        logger.debug("Configured yfinance cache at %s", cache_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unable to relocate yfinance cache to %s: %s. Using default location.",
            cache_dir,
            exc,
        )
    else:
        try:
            cookie_cache = yf_cache._CookieCache()
            cookie_cache.dummy = True
            yf_cache._CookieCacheManager._Cookie_cache = cookie_cache
            logger.debug("Disabled yfinance cookie cache persistence (in-memory only).")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Unable to disable cookie cache persistence: %s", exc)

    # Step 1: Discovery
    logger.info("")
    logger.info("STEP 1: Discovery - Fetching listings from Yahoo Finance screener")
    logger.info("-"*70)
    listings = fetch_listings(config.screener)
    logger.info("Discovery completed: %s listings found", len(listings))

    # Log sample symbols
    if listings:
        sample_symbols = [l.get('symbol') for l in listings[:10]]
        logger.info("Sample symbols (first 10): %s", sample_symbols)

    # Step 2: Enrichment
    logger.info("")
    logger.info("STEP 2: Enrichment - Fetching detailed data for each symbol")
    logger.info("-"*70)
    dataset = build_dataset(listings, config.enrich)

    # Step 3: Validation
    logger.info("")
    logger.info("STEP 3: Validation - Running quality checks")
    logger.info("-"*70)
    dataset = run_validations(dataset)

    # Step 4: Output Summary
    logger.info("")
    logger.info("STEP 4: Summary - Dataset statistics")
    logger.info("-"*70)
    logger.info("Total records: %s", len(dataset))
    logger.info("Unique symbols: %s", dataset['symbol'].nunique())

    if 'exchange' in dataset.columns:
        exchange_counts = dataset['exchange'].value_counts().to_dict()
        logger.info("Exchanges breakdown:")
        for exchange, count in exchange_counts.items():
            logger.info("  - %s: %s", exchange, count)

    if 'data_quality_score' in dataset.columns:
        avg_quality = dataset['data_quality_score'].mean()
        logger.info("Average data quality score: %.3f", avg_quality)

    # Step 5: Write output
    logger.info("")
    logger.info("STEP 5: Writing output")
    logger.info("-"*70)
    write_csv(dataset, config.storage.output_csv)
    logger.info("CSV written to: %s", config.storage.output_csv)

    if emit_metadata:
        metadata = {
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "record_count": len(dataset),
            "unique_symbols": int(dataset['symbol'].nunique()),
            "columns": list(dataset.columns),
            "output_csv": str(config.storage.output_csv),
            "pipeline_version": __version__,
        }

        # Add exchange breakdown if available
        if 'exchange' in dataset.columns:
            metadata["exchanges"] = dataset['exchange'].value_counts().to_dict()

        # Add quality score if available
        if 'data_quality_score' in dataset.columns:
            metadata["avg_quality_score"] = float(dataset['data_quality_score'].mean())

        write_metadata(metadata, config.storage.output_csv)
        logger.info("Metadata written to: %s", str(config.storage.output_csv) + ".meta.json")

    duration = time.time() - start
    logger.info("")
    logger.info("="*70)
    logger.info("Run completed successfully in %.2fs", duration)
    logger.info("="*70)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indian-equities-master",
        description="Fetch and normalise Indian equities master data from Yahoo Finance.",
    )
    parser.add_argument(
        "command",
        choices={"update"},
        nargs="?",
        default="update",
        help="Pipeline action to execute.",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Optional path to YAML config (defaults to config/indian_equities_master.yaml).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Override output CSV path.",
    )
    parser.add_argument(
        "--enable-options-lookup",
        action="store_true",
        help="Attempt to query option chain availability for each symbol (slower).",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not emit sidecar metadata JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging verbosity.",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
