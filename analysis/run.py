#!/usr/bin/env python3
"""
Unified Analysis Runner
=======================

Executes generic analysis and portfolio construction modules defined in the
YAML configuration. Modules are orchestrated with dependency awareness and
outputs are recorded in per-run Markdown logs.

Usage:
    python analysis/run.py --config analysis/configs/mse_run.yaml
    python analysis/run.py --config config.yaml --targets generic,portfolio
    python analysis/run.py --config config.yaml --only basic_eda,cascade_analysis
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repository root (two levels up from this file)
ROOT = Path(__file__).resolve().parent.parent

# Add repository root to Python path for imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.generic.modules import (
    load_config,
    resolve_paths,
    get_module_spec,
    resolve_artifact_path,
)

# Ensure stdout/stderr can emit UTF-8 even when Windows defaults to cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULE_REGISTRY: Dict[str, Dict[str, Dict[str, str]]] = {
    "generic": {
        "basic_eda": {"script": "analysis/generic/scripts/01_basic_eda.py"},
        "trade_type_analysis": {"script": "analysis/generic/scripts/02_trade_type_analysis.py"},
        "stop_loss_simulation": {"script": "analysis/generic/scripts/04_stop_loss_simulation.py"},
        "cascade_analysis": {"script": "analysis/generic/scripts/03_cascade_analysis.py"},
        "ticker_ranking": {"script": "analysis/generic/scripts/05_ticker_ranking.py"},
        "risk_adjusted_patterns": {"script": "analysis/generic/scripts/06_risk_adjusted_patterns.py"},
        "validation_check": {"script": "analysis/generic/scripts/09_validation_check.py"},
        "top50_vs_overall": {"script": "analysis/generic/scripts/07_top50_vs_overall.py"},
        "top50_pattern_breakdown": {"script": "analysis/generic/scripts/08_top50_pattern_breakdown.py"},
    },
    "portfolio": {
        "ticker_ranking": {"script": "analysis/portfolio_construction/scripts/00_foundation_cascade_vs_anticascade_analysis.py"},
        "anti_cascade_filter": {"script": "analysis/portfolio_construction/scripts/01_corrected_anti_cascading_subset.py"},
        "sector_classification": {"script": "analysis/portfolio_construction/scripts/02_corrected_sector_classification_correlation.py"},
        "combination_generator": {"script": "analysis/portfolio_construction/scripts/03_corrected_intelligent_combination_generation.py"},
        "portfolio_optimizer": {"script": "analysis/portfolio_construction/scripts/04_portfolio_optimization_engine.py"},
        "pypfopt_weights": {"script": "analysis/portfolio_construction/scripts/05_pypfopt_optimal_weights.py"},
        "equity_curves": {"script": "analysis/portfolio_construction/scripts/06_equity_curve_generator.py"},
    },
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class ModuleExecutionResult:
    def __init__(
        self,
        module: str,
        category: str,
        status: str,
        message: str = "",
        outputs: Optional[List[str]] = None,
        stdout: str = "",
        stderr: str = "",
        duration_seconds: float = 0.0,
    ) -> None:
        self.module = module
        self.category = category
        self.status = status
        self.message = message
        self.outputs = outputs or []
        self.stdout = stdout
        self.stderr = stderr
        self.duration_seconds = duration_seconds

    def status_icon(self) -> str:
        mapping = {
            "success": "[OK]",
            "skipped-disabled": "[SKIP-disabled]",
            "skipped-missing-script": "[SKIP-missing]",
            "skipped-dependency": "[SKIP-dependency]",
            "error": "[ERROR]",
        }
        return mapping.get(self.status, "[SKIP]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis modules based on YAML config")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument(
        "--targets",
        help="Comma-separated list of targets to run (e.g. generic,portfolio)",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated list of specific modules to run (category inferred from config)",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip automatic merge even if merged file is missing",
    )
    return parser.parse_args()


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def flatten_strategy_specific_categories(config: Dict[str, any]) -> Dict[str, Tuple[str, Dict]]:
    categories: Dict[str, Tuple[str, Dict]] = {}
    strategy_specific = config.get("analysis", {}).get("strategy_specific", {})
    for strategy_name, strategy_block in strategy_specific.items():
        if not isinstance(strategy_block, dict):
            continue
        if strategy_block.get("enabled"):
            categories[f"strategy_specific:{strategy_name}"] = (strategy_name, strategy_block)
    return categories


def list_enabled_modules(config: Dict[str, Any], category: str) -> Dict[str, Dict[str, Any]]:
    analysis_block = config.get("analysis", {})
    if category.startswith("strategy_specific:"):
        strategy_name = category.split(":", 1)[1]
        strategy_block = analysis_block.get("strategy_specific", {}).get(strategy_name, {})
        defaults = strategy_block.get("modules", {})
        return {name: spec for name, spec in defaults.items() if spec.get("enabled", False)}

    category_block = analysis_block.get(category, {})
    modules = category_block.get("modules", {})
    return {name: spec for name, spec in modules.items() if spec.get("enabled", False)}


def build_dependency_graph(modules: Dict[str, Dict[str, Any]], category: str) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {module: [] for module in modules}
    for module, spec in modules.items():
        depends_on = spec.get("depends_on", [])
        for dep in depends_on:
            dep_cat, dep_mod = dep.split(":", 1) if ":" in dep else (category, dep)
            if dep_cat == category and dep_mod in modules:
                graph[module].append(dep_mod)
    return graph


def topological_sort(modules: Dict[str, Dict[str, Any]], category: str) -> List[str]:
    graph = build_dependency_graph(modules, category)
    order: List[str] = []
    visited: Dict[str, str] = {}

    def dfs(node: str) -> None:
        state = visited.get(node)
        if state == "temp":
            raise ValueError(f"Cyclic dependency detected in {category}: {node}")
        if state == "perm":
            return
        visited[node] = "temp"
        for neighbour in graph.get(node, []):
            dfs(neighbour)
        visited[node] = "perm"
        order.append(node)

    for module in modules:
        if visited.get(module) != "perm":
            dfs(module)

    # `order` already appends nodes after their dependencies have been visited
    return order


def ensure_merge_file(config_path: Path, config: Dict[str, Any], paths: Dict[str, str], skip_merge: bool) -> None:
    schema = config.get("_schema")
    if schema != "modular":
        return

    merged_path = Path(paths["merged_trades_file"])
    merge_cfg = config.get("merge", {})
    if merged_path.exists() or not merge_cfg.get("auto_generate", True):
        return

    if skip_merge:
        print("[WARN] Skipping auto-merge because --skip-merge flag was provided.")
        return

    print(f"[INFO] Merged trades file missing. Running merge script to create {merged_path} ...")
    merge_script = ROOT / "utils" / "merge_trades.py"
    cmd = [sys.executable, str(merge_script), "--config", str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError("Merge script failed. Aborting analysis run.")


def run_module_script(script_path: Path, config_path: Path) -> Tuple[int, str, str, float]:
    start = dt.datetime.now()
    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    module_root = script_path.parent.parent  # e.g., analysis/generic
    python_path_parts = []
    if module_root.exists():
        python_path_parts.append(str(module_root))
    if script_path.parent.exists():
        python_path_parts.append(str(script_path.parent))
    if python_path_parts:
        existing = env.get("PYTHONPATH", "")
        combined = os.pathsep.join(
            [p for p in python_path_parts if p] + ([existing] if existing else [])
        )
        env["PYTHONPATH"] = combined
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    duration = (dt.datetime.now() - start).total_seconds()
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode, result.stdout, result.stderr, duration


def compute_expected_outputs(config: Dict[str, Any], module_name: str, category: str) -> List[str]:
    module_spec = get_module_spec(config, module_name, category)
    outputs = []
    for output in module_spec.get("outputs", []):
        artifact = output.get("artifact")
        artifact_type = output.get("type", "csv")
        if not artifact:
            continue
        try:
            outputs.append(resolve_artifact_path(config, module_name, artifact, category=category, artifact_type=artifact_type))
        except Exception as exc:  # pragma: no cover
            outputs.append(f"[unresolved:{artifact}] ({exc})")
    return outputs


def execute_category(
    config_path: Path,
    config: Dict[str, Any],
    category: str,
    global_results: Dict[str, Dict[str, ModuleExecutionResult]],
) -> List[ModuleExecutionResult]:
    modules = list_enabled_modules(config, category)
    if not modules:
        print(f"[INFO] No modules enabled for category '{category}'.")
        return []

    registry = MODULE_REGISTRY.get(category, {})
    order = topological_sort(modules, category)

    results: List[ModuleExecutionResult] = []
    global_results.setdefault(category, {})

    for module in order:
        module_spec = modules[module]

        # Check cross-category dependencies
        blocked = False
        reason = ""
        for dep in module_spec.get("depends_on", []):
            dep_cat, dep_mod = dep.split(":", 1) if ":" in dep else (category, dep)
            dep_status = global_results.get(dep_cat, {}).get(dep_mod)
            if dep_cat == category:
                continue  # Handled in topological order
            if not dep_status or dep_status.status != "success":
                blocked = True
                reason = f"Dependency '{dep}' not satisfied."
                break
        if blocked:
            result = ModuleExecutionResult(
                module=module,
                category=category,
                status="skipped-dependency",
                message=reason,
            )
            results.append(result)
            global_results[category][module] = result
            print(f"[SKIP] Skipping {category}:{module} -> {reason}")
            continue

        script_info = registry.get(module)
        if not script_info:
            result = ModuleExecutionResult(
                module=module,
                category=category,
                status="skipped-missing-script",
                message="Module not registered in runner.",
            )
            results.append(result)
            global_results[category][module] = result
            print(f"[WARN] No runner entry for {category}:{module}. Skipping.")
            continue

        script_path = ROOT / script_info['script']
        if not script_path.exists():
            result = ModuleExecutionResult(
                module=module,
                category=category,
                status="skipped-missing-script",
                message=f"Script not found at {script_path}",
            )
            results.append(result)
            global_results[category][module] = result
            print(f"[WARN] Script missing for {category}:{module} ({script_path}).")
            continue

        print(f"\n{'=' * 80}")
        print(f"[RUN] Running {category}:{module}")
        print(f"{'=' * 80}")

        returncode, stdout, stderr, duration = run_module_script(script_path, config_path)

        expected_outputs = compute_expected_outputs(config, module, category)

        status = "success" if returncode == 0 else "error"
        message = "Completed successfully." if status == "success" else f"Exited with code {returncode}"

        result = ModuleExecutionResult(
            module=module,
            category=category,
            status=status,
            message=message,
            outputs=expected_outputs,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )
        results.append(result)
        global_results[category][module] = result

    return results


def write_run_log(config: Dict[str, Any], category: str, results: List[ModuleExecutionResult], paths: Dict[str, str]) -> None:
    run_cfg = config.get("run", {})
    log_root = Path(paths['run_logs_dir'])
    log_dir = log_root / run_cfg.get("strategy", "unknown_strategy") / run_cfg.get("run_id", "unknown_run")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{category.replace(':', '_')}_run.md"

    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [
        f"# {category.title()} Run Report",
        "",
        f"- **Timestamp**: {timestamp}",
        f"- **Strategy**: {run_cfg.get('strategy')}",
        f"- **Run ID**: {run_cfg.get('run_id')}",
    ]
    if run_cfg.get("date_range"):
        log_lines.append(f"- **Date Range**: {run_cfg['date_range']}")
    if run_cfg.get("label"):
        log_lines.append(f"- **Label**: {run_cfg['label']}")
    log_lines.append("")

    log_lines.append("| Module | Status | Duration | Outputs | Notes |")
    log_lines.append("|---|---|---|---|---|")

    for result in results:
        outputs = "<br>".join(result.outputs) if result.outputs else "-"
        note = result.message or "-"
        log_lines.append(
            f"| `{result.module}` | {result.status_icon()} | {result.duration_seconds:.1f}s | {outputs} | {note} |"
        )

    log_lines.append("")
    log_lines.append("## Diagnostics")
    for result in results:
        if result.stdout.strip():
            log_lines.append(f"### {result.module} stdout")
            log_lines.append("```text")
            log_lines.append(result.stdout.strip())
            log_lines.append("```")
        if result.stderr.strip():
            log_lines.append(f"### {result.module} stderr")
            log_lines.append("```text")
            log_lines.append(result.stderr.strip())
            log_lines.append("```")

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[INFO] Wrote run log -> {log_path}")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    config = load_config(str(config_path))
    paths = resolve_paths(config)

    try:
        ensure_merge_file(config_path, config, paths, skip_merge=args.skip_merge)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    available_categories = []
    analysis_block = config.get("analysis", {})
    if analysis_block.get("generic", {}).get("enabled", False):
        available_categories.append("generic")
    if analysis_block.get("portfolio", {}).get("enabled", False):
        available_categories.append("portfolio")
    strategy_categories = flatten_strategy_specific_categories(config)
    available_categories.extend(strategy_categories.keys())

    targets = split_csv(args.targets) if args.targets else available_categories
    only_modules = set(split_csv(args.only))

    execution_results: Dict[str, Dict[str, ModuleExecutionResult]] = {}
    exit_code = 0

    for category in targets:
        if category not in available_categories:
            print(f"[WARN] Target '{category}' not available or not enabled in config. Skipping.")
            continue

        category_results = execute_category(config_path, config, category, execution_results)

        # Filter to only requested modules if --only provided
        if only_modules:
            category_results = [
                result for result in category_results if result.module in only_modules
            ]

        write_run_log(config, category, category_results, paths)

        if any(result.status == "error" for result in category_results):
            exit_code = 1

    print("\n[OK] Run complete." if exit_code == 0 else "\n[ERROR] Run completed with errors.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
