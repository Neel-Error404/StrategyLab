#!/usr/bin/env python3
"""
Config Loader Module
====================

Loads and validates YAML configuration files for analysis scripts.

Usage (modular schema):
    from modules.config_loader import load_config, resolve_paths

    config = load_config('analysis/configs/mse_run.yaml')
    paths = resolve_paths(config)

    merged_file = paths['merged_trades_file']
    base_data_dir = paths['base_data_dir']
    output_dir = get_output_dir(config, module_name='basic_eda', category='generic')
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _PatternDict(dict):
    """Safe formatter that leaves unknown tokens untouched."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _detect_schema(config: Dict[str, Any]) -> str:
    if 'data_sources' in config or 'analysis' in config and 'generic' in config:
        return 'modular'
    return 'legacy'


def _validate_modular_config(config: Dict[str, Any]) -> None:
    required_sections = ['run', 'data_sources', 'output', 'analysis']
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    run_required = ['run_id', 'strategy']
    missing_run = [field for field in run_required if field not in config['run']]
    if missing_run:
        raise ValueError(f"Config 'run' section missing fields: {missing_run}")

    data_sources = config['data_sources']
    for field in ['strategy_trades_dir', 'base_data_dir']:
        if field not in data_sources or not data_sources[field]:
            raise ValueError(f"'data_sources.{field}' is required.")

    merge_cfg = config.get('merge', {})
    if merge_cfg.get('auto_generate', True):
        merge_required = ['trade_source', 'output_filename']
        missing_merge = [field for field in merge_required if field not in merge_cfg]
        if missing_merge:
            raise ValueError(f"Config 'merge' section missing fields: {missing_merge}")

    output_required = ['root_dir', 'reports_root_dir', 'run_logs_dir']
    missing_output = [field for field in output_required if field not in config['output']]
    if missing_output:
        raise ValueError(f"Config 'output' section missing fields: {missing_output}")


def _validate_legacy_config(config: Dict[str, Any]) -> None:
    required_sections = ['run', 'output']
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    required_run_fields = ['run_id', 'strategy', 'date_range', 'trade_source']
    missing_run = [f for f in required_run_fields if f not in config['run']]
    if missing_run:
        raise ValueError(f"Config 'run' section missing fields: {missing_run}")


def _build_context(config: Dict[str, Any], category: Optional[str] = None, module: Optional[str] = None,
                   artifact: Optional[str] = None) -> Dict[str, Any]:
    run_cfg = config.get('run', {})
    output_cfg = config.get('output', {})

    context = {
        'root': output_cfg.get('root_dir', output_cfg.get('analysis_output_dir', 'analysis/output')),
        'reports_root': output_cfg.get('reports_root_dir', output_cfg.get('reports_dir', 'analysis/reports')),
        'run_logs_root': output_cfg.get('run_logs_dir', 'analysis/run_logs'),
        'strategy': run_cfg.get('strategy', 'unknown_strategy'),
        'run_id': run_cfg.get('run_id', 'unknown_run'),
        'date_range': run_cfg.get('date_range', ''),
        'label': run_cfg.get('label', ''),
        'category': category or '',
        'module': module or '',
        'artifact': artifact or '',
    }
    return context


def _render_pattern(pattern: str, context: Dict[str, Any]) -> str:
    return pattern.format_map(_PatternDict(context))


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected at: {config_file.absolute()}\n"
            f"Create it by copying config_template.yaml"
        )

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config: {e}")

    schema = _detect_schema(config)
    if schema == 'modular':
        _validate_modular_config(config)
    else:
        _validate_legacy_config(config)

    config['_schema'] = schema

    run_cfg = config.get('run', {})
    print(f"✅ Loaded config from: {config_path}")
    print(f"   Run ID: {run_cfg.get('run_id')}")
    print(f"   Strategy: {run_cfg.get('strategy')}")
    if run_cfg.get('date_range'):
        print(f"   Date Range: {run_cfg.get('date_range')}")

    if schema == 'modular':
        ds = config['data_sources']
        print(f"   Strategy trades dir: {ds['strategy_trades_dir']}")
        print(f"   Base data dir: {ds['base_data_dir']}")
    else:
        print(f"   Trade Source: {run_cfg.get('trade_source')}")

    return config

def resolve_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Resolve all paths from configuration.

    Constructs full paths from config variables:
        {run_id}, {strategy}, {date_range}

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with resolved paths:
            - merged_trades_file: Path to merged trades CSV
            - base_data_dir: Directory with base data files
            - outputs_base: Base output directory
            - strategy_trades_dir: Strategy trades directory
            - risk_approved_trades_dir: Risk approved trades directory
    """
    schema = config.get('_schema', _detect_schema(config))

    if schema == 'modular':
        context = _build_context(config)
        output_cfg = config['output']
        data_cfg = config['data_sources']
        merge_cfg = config.get('merge', {})

        data_dir_pattern = output_cfg.get('data_dir', "{root}/{strategy}/{run_id}/data")
        data_dir = Path(_render_pattern(data_dir_pattern, context))
        _ensure_directory(data_dir)

        merged_source = data_cfg.get('merged_trades_dir')
        merged_filename = merge_cfg.get('output_filename', 'all_trades_merged.csv')

        if merged_source:
            merged_path = Path(merged_source)
            if merged_path.is_dir():
                merged_file = merged_path / merged_filename
            else:
                merged_file = merged_path
        else:
            merged_file = data_dir / merged_filename

        paths: Dict[str, str] = {
            'strategy_trades_dir': str(data_cfg['strategy_trades_dir']),
            'base_data_dir': str(data_cfg['base_data_dir']),
            'merged_trades_file': str(merged_file),
            'data_dir': str(data_dir),
            'root_dir': str(Path(context['root'])),
            'reports_root_dir': str(Path(context['reports_root'])),
            'run_logs_dir': str(Path(context['run_logs_root'])),
        }

        # Warn user about missing merged file if auto_generate disabled
        if not Path(paths['merged_trades_file']).exists():
            print(f"\n⚠️  WARNING: Merged trades file not found!")
            print(f"   Expected: {paths['merged_trades_file']}")
            if merge_cfg.get('auto_generate', True):
                print(f"   It will be created automatically when the runner executes merge_trades.\n")
            else:
                print(f"   Run merge script first or provide 'data_sources.merged_trades_dir'.\n")

        return paths

    # Legacy fallback
    run_id = config['run']['run_id']
    strategy = config['run']['strategy']
    date_range = config['run']['date_range']
    outputs_base = Path(f"outputs/{run_id}/{strategy}/{date_range}")

    if 'paths' in config and config['paths']:
        paths_section = config['paths']
        paths = {
            'merged_trades_file': paths_section.get('merged_trades_file', str(outputs_base / 'data' / config['output']['merged_filename'])),
            'base_data_dir': paths_section.get('base_data_dir', str(outputs_base / 'data' / 'base_data')),
            'outputs_base': paths_section.get('outputs_base', str(outputs_base)),
        }
    else:
        merged_filename = config['output']['merged_filename']
        paths = {
            'merged_trades_file': str(outputs_base / 'data' / merged_filename),
            'base_data_dir': str(outputs_base / 'data' / 'base_data'),
            'outputs_base': str(outputs_base),
            'strategy_trades_dir': str(outputs_base / 'data' / 'strategy_trades'),
            'risk_approved_trades_dir': str(outputs_base / 'data' / 'risk_approved_trades'),
        }

    merged_file = Path(paths['merged_trades_file'])
    if not merged_file.exists():
        print(f"\n⚠️  WARNING: Merged trades file not found!")
        print(f"   Expected: {merged_file}")
        print(f"\n   Run merge script first:")
        print(f"   python ../utils/merge_trades.py --config config.yaml\n")

    return paths

def get_analysis_config(config: Dict[str, Any], module_name: str, category: str = 'generic') -> Dict[str, Any]:
    """
    Get configuration for a specific analysis module.

    Args:
        config: Full configuration dictionary
        module_name: Name of analysis module (e.g., 'cascade_analysis')

    Returns:
        Module-specific configuration, or empty dict if not found
    """
    if 'analysis' not in config or category not in config['analysis']:
        return {}

    modules = config['analysis'][category].get('modules', {})
    module_spec = modules.get(module_name, {})
    if not module_spec:
        return {}

    if not module_spec.get('enabled', True):
        print(f"⚠️  Module '{category}:{module_name}' is disabled in config")
        return {}

    return module_spec.get('config', {})


def get_module_spec(config: Dict[str, Any], module_name: str, category: str = 'generic') -> Dict[str, Any]:
    """Return full module specification from config."""
    if 'analysis' not in config or category not in config['analysis']:
        return {}
    return config['analysis'][category].get('modules', {}).get(module_name, {})


def get_output_dir(config: Dict[str, Any], module_name: str, category: str = 'generic') -> str:
    """
    Get output directory for saving analysis results.

    Args:
        config: Configuration dictionary
        subdir: Optional subdirectory name

    Returns:
        Path to output directory
    """
    schema = config.get('_schema', _detect_schema(config))
    if schema == 'modular':
        defaults = config['analysis'][category].get('defaults', {})
        pattern = get_module_spec(config, module_name, category).get('output', {}).get(
            'dir',
            defaults.get('output_dir', config['output'].get('dir_pattern', "{root}/{strategy}/{run_id}/{category}/{module}"))
        )
        context = _build_context(config, category=category, module=module_name)
        output_dir = Path(_render_pattern(pattern, context))
        return str(_ensure_directory(output_dir))

    # Legacy fallback behaviour
    base_output = config['output'].get('analysis_output_dir', 'generic/output')
    output_dir = Path(base_output) / module_name if module_name else Path(base_output)
    return str(_ensure_directory(output_dir))


def get_report_dir(config: Dict[str, Any], module_name: Optional[str] = None, category: str = 'generic') -> str:
    schema = config.get('_schema', _detect_schema(config))
    if schema == 'modular':
        defaults = config['analysis'][category].get('defaults', {})
        pattern = get_module_spec(config, module_name or '', category).get('output', {}).get(
            'report_dir',
            defaults.get('report_dir', config['output'].get('report_pattern', "{reports_root}/{strategy}/{run_id}/{category}/{module}"))
        )
        context = _build_context(config, category=category, module=module_name or '')
        report_dir = Path(_render_pattern(pattern, context))
        return str(_ensure_directory(report_dir))

    reports_dir = Path(config['output'].get('reports_dir', 'generic/reports'))
    return str(_ensure_directory(reports_dir))


def resolve_artifact_path(
    config: Dict[str, Any],
    module_name: str,
    artifact: str,
    category: str = 'generic',
    artifact_type: str = 'csv'
) -> str:
    """
    Resolve full path for a module artifact based on config patterns.
    """
    schema = config.get('_schema', _detect_schema(config))
    filename_template = config.get('output', {}).get('filename_template', '{module}_{artifact}')
    context = _build_context(config, category=category, module=module_name, artifact=artifact)
    filename = _render_pattern(filename_template, context)

    extension_map = {
        'csv': '.csv',
        'json': '.json',
        'parquet': '.parquet',
        'markdown': '.md',
        'md': '.md',
        'html': '.html',
        'png': '.png',
        'pdf': '.pdf',
        'txt': '.txt',
    }
    ext = extension_map.get(artifact_type.lower(), '')

    if schema == 'modular':
        if artifact_type.lower() in ('markdown', 'md', 'pdf', 'html'):
            base_dir = Path(get_report_dir(config, module_name=module_name, category=category))
        else:
            base_dir = Path(get_output_dir(config, module_name=module_name, category=category))
    else:
        base_dir = Path(get_output_dir(config, module_name))

    full_path = base_dir / f"{filename}{ext}"
    _ensure_directory(full_path.parent)
    return str(full_path)
