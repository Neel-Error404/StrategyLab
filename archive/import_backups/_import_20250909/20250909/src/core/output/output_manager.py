# src/output/output_manager.py
"""
Comprehensive Output Management System for Backtesting Framework.

This module provides unified output structure management to ensure:
1. Single-source input → organized output structure for easy analysis
2. All outputs (trades, analysis, visualizations, costs) consolidated in one coherent place per strategy test
3. Clear naming conventions showing what analysis belongs to which strategy/setup
4. Easy analysis workflow from input to consolidated output
5. Validation that analysis calculations, visualizations, and storage structure are working correctly
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

class OutputManager:
    """
    Centralized output management for backtesting framework.
    Provides unified, organized output structure with clear naming conventions.
    """
    
    def __init__(self, base_output_dir: Union[str, Path] = "Backtester/Strat_out"):
        """
        Initialize the Output Manager.
        
        Args:
            base_output_dir: Base directory for all strategy outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger("OutputManager")
        
        # Ensure base directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define standard naming conventions
        self.naming_conventions = {
            'strategy_run_dir': '{strategy_name}_{date_range}_{run_id}',
            'base_file': '{ticker}_Base_{date_range}.csv',
            'trades_file': '{ticker}_Trades_{date_range}.csv',
            'report_file': '{ticker}_Report_{date_range}.json',
            'summary_file': '{date_range}_Summary.csv',
            'portfolio_analysis': 'Portfolio_Analysis_{date_range}.json',
            'visualization_dir': 'visualizations',
            'analysis_dir': 'analysis',
            'costs_dir': 'transaction_costs',
            'metadata_file': 'run_metadata.json'
        }
        
    def create_strategy_run_structure(self, strategy_name: str, date_range: str, 
                                    run_id: Optional[str] = None) -> Path:
        """
        Create a comprehensive directory structure for a strategy run.
        
        Args:
            strategy_name: Name of the strategy
            date_range: Date range in format YYYY-MM-DD_to_YYYY-MM-DD
            run_id: Optional unique run identifier (defaults to timestamp)
            
        Returns:
            Path to the strategy run directory
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create main strategy run directory
        strategy_run_dir = self.base_output_dir / f"{strategy_name}_{date_range}_{run_id}"
        
        # Create subdirectories
        subdirs = [
            'raw_data',           # Base data files with signals
            'trades',             # Trade execution data
            'reports',            # Individual ticker reports
            'analysis',           # Portfolio and cross-ticker analysis
            'visualizations',     # All charts and plots
            'transaction_costs',  # Cost analysis and breakdown
            'metadata'            # Run configuration and metadata
        ]
        
        for subdir in subdirs:
            (strategy_run_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"Created strategy run structure: {strategy_run_dir}")
        return strategy_run_dir
        
    def save_run_metadata(self, strategy_run_dir: Path, metadata: Dict[str, Any]) -> None:
        """
        Save comprehensive metadata for the strategy run.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            metadata: Dictionary containing run metadata
        """
        metadata_enhanced = {
            'run_timestamp': datetime.now().isoformat(),
            'framework_version': '2.0.0',  # Framework version
            'output_structure_version': '1.0.0',
            **metadata
        }
        
        metadata_file = strategy_run_dir / 'metadata' / 'run_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata_enhanced, f, indent=2, default=str)
            
        self.logger.info(f"Saved run metadata to {metadata_file}")
        
    def save_ticker_data(self, strategy_run_dir: Path, ticker: str, date_range: str,
                        base_data: pd.DataFrame, trades: List[Dict], 
                        metrics: Dict[str, Any], bias_report: Optional[Dict] = None) -> Dict[str, Path]:
        """
        Save all ticker-related data with consistent naming.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            ticker: Ticker symbol
            date_range: Date range string
            base_data: DataFrame with base data and signals
            trades: List of trade dictionaries
            metrics: Dictionary of calculated metrics
            bias_report: Optional bias detection report
            
        Returns:
            Dictionary mapping file types to saved file paths
        """
        saved_files = {}
        
        # Save base data (with signals)
        base_file = strategy_run_dir / 'raw_data' / f"{ticker}_Base_{date_range}.csv"
        base_data.to_csv(base_file, index=False)
        saved_files['base_data'] = base_file
        
        # Save trades
        if trades:
            trades_file = strategy_run_dir / 'trades' / f"{ticker}_Trades_{date_range}.csv"
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(trades_file, index=False)
            saved_files['trades'] = trades_file
        
        # Create detailed report
        detailed_report = {
            'strategy': strategy_run_dir.name.split('_')[0],  # Extract strategy name
            'ticker': ticker,
            'date_range': date_range,
            'metrics': metrics,
            'bias_report': bias_report or {},
            'data_summary': {
                'total_data_points': len(base_data),
                'date_range_actual': {
                    'start': str(base_data['timestamp'].min()) if 'timestamp' in base_data.columns else 'N/A',
                    'end': str(base_data['timestamp'].max()) if 'timestamp' in base_data.columns else 'N/A'
                },
                'signals_generated': {
                    'long_entry': base_data['long_entry'].sum() if 'long_entry' in base_data.columns else 0,
                    'long_exit': base_data['long_exit'].sum() if 'long_exit' in base_data.columns else 0,
                    'short_entry': base_data['short_entry'].sum() if 'short_entry' in base_data.columns else 0,
                    'short_exit': base_data['short_exit'].sum() if 'short_exit' in base_data.columns else 0,
                }
            },
            'file_locations': {str(k): str(v) for k, v in saved_files.items()},
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save detailed report
        report_file = strategy_run_dir / 'reports' / f"{ticker}_Report_{date_range}.json"
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        saved_files['report'] = report_file
        
        self.logger.info(f"Saved ticker data for {ticker}: {len(saved_files)} files")
        return saved_files
        
    def save_portfolio_analysis(self, strategy_run_dir: Path, date_range: str,
                              portfolio_analysis: Dict[str, Any]) -> Path:
        """
        Save comprehensive portfolio analysis.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            portfolio_analysis: Portfolio analysis results
            
        Returns:
            Path to saved portfolio analysis file
        """
        # Enhanced portfolio analysis with metadata
        enhanced_analysis = {
            'analysis_type': 'portfolio_analysis',
            'date_range': date_range,
            'strategy_name': strategy_run_dir.name.split('_')[0],
            'analysis_timestamp': datetime.now().isoformat(),
            'results': portfolio_analysis
        }
        
        analysis_file = strategy_run_dir / 'analysis' / f"Portfolio_Analysis_{date_range}.json"
        with open(analysis_file, 'w') as f:
            json.dump(enhanced_analysis, f, indent=2, default=str)
            
        self.logger.info(f"Saved portfolio analysis to {analysis_file}")
        return analysis_file
        
    def save_visualization_outputs(self, strategy_run_dir: Path, date_range: str,
                                 visualization_files: Dict[str, Path]) -> Dict[str, Path]:
        """
        Organize and save visualization outputs with clear naming.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            visualization_files: Dictionary mapping chart types to file paths
            
        Returns:
            Dictionary mapping chart types to organized file paths
        """
        viz_dir = strategy_run_dir / 'visualizations'
        organized_files = {}
        
        # Define standard visualization categories
        viz_categories = {
            'equity_curve': 'performance',
            'trade_distribution': 'trade_analysis',
            'correlation_heatmap': 'correlations',
            'performance_metrics': 'performance',
            'drawdown_chart': 'risk',
            'returns_distribution': 'returns',
            'monthly_returns': 'returns',
            'rolling_metrics': 'performance',
            'risk_dashboard': 'risk'
        }
        
        for viz_type, source_path in visualization_files.items():
            # Determine category
            category = viz_categories.get(viz_type, 'other')
            category_dir = viz_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Create standardized filename
            file_extension = Path(source_path).suffix
            target_filename = f"{viz_type}_{date_range}{file_extension}"
            target_path = category_dir / target_filename
            
            # Copy/move file to organized location
            if Path(source_path).exists():
                shutil.copy2(source_path, target_path)
                organized_files[viz_type] = target_path
                self.logger.info(f"Organized visualization: {viz_type} -> {target_path}")
            
        return organized_files
        
    def save_transaction_cost_analysis(self, strategy_run_dir: Path, date_range: str,
                                     cost_analysis: Dict[str, Any]) -> Path:
        """
        Save transaction cost analysis with breakdown.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            cost_analysis: Transaction cost analysis results
            
        Returns:
            Path to saved cost analysis file
        """
        # Enhanced cost analysis with metadata
        enhanced_cost_analysis = {
            'analysis_type': 'transaction_cost_analysis',
            'date_range': date_range,
            'strategy_name': strategy_run_dir.name.split('_')[0],
            'analysis_timestamp': datetime.now().isoformat(),
            'cost_breakdown': cost_analysis
        }
        
        cost_file = strategy_run_dir / 'transaction_costs' / f"Cost_Analysis_{date_range}.json"
        with open(cost_file, 'w') as f:
            json.dump(enhanced_cost_analysis, f, indent=2, default=str)
            
        self.logger.info(f"Saved transaction cost analysis to {cost_file}")
        return cost_file
        
    def create_run_summary(self, strategy_run_dir: Path, date_range: str,
                          ticker_summaries: List[Dict[str, Any]]) -> Path:
        """
        Create a comprehensive run summary combining all ticker results.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            ticker_summaries: List of individual ticker summary dictionaries
            
        Returns:
            Path to saved run summary file
        """
        # Create DataFrame from ticker summaries
        summary_df = pd.DataFrame(ticker_summaries)
        
        # Save CSV summary
        summary_csv = strategy_run_dir / f"Run_Summary_{date_range}.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        # Create enhanced JSON summary
        run_summary = {
            'summary_type': 'strategy_run_summary',
            'strategy_name': strategy_run_dir.name.split('_')[0],
            'date_range': date_range,
            'run_id': strategy_run_dir.name.split('_')[-1],
            'summary_timestamp': datetime.now().isoformat(),            'aggregate_metrics': {
                'total_tickers': len(ticker_summaries),
                'total_trades': summary_df['Total Trades'].sum() if 'Total Trades' in summary_df.columns else 0,
                'aggregate_win_rate': summary_df['Accuracy (%)'].mean() if 'Accuracy (%)' in summary_df.columns else 0,
                'aggregate_profit': summary_df['Average Profit (Currency)'].sum() if 'Average Profit (Currency)' in summary_df.columns else 0,
                'best_performing_ticker': summary_df.loc[summary_df['Average Profit (Currency)'].idxmax(), 'Ticker'] if not summary_df.empty and 'Average Profit (Currency)' in summary_df.columns and 'Ticker' in summary_df.columns else 'N/A',
                'worst_performing_ticker': summary_df.loc[summary_df['Average Profit (Currency)'].idxmin(), 'Ticker'] if not summary_df.empty and 'Average Profit (Currency)' in summary_df.columns and 'Ticker' in summary_df.columns else 'N/A'
            },
            'file_structure': self._generate_file_structure_map(strategy_run_dir),
            'ticker_summaries': ticker_summaries
        }
        
        summary_json = strategy_run_dir / f"Run_Summary_{date_range}.json"
        with open(summary_json, 'w') as f:
            json.dump(run_summary, f, indent=2, default=str)
            
        self.logger.info(f"Created run summary: CSV at {summary_csv}, JSON at {summary_json}")
        return summary_json
        
    def _generate_file_structure_map(self, strategy_run_dir: Path) -> Dict[str, Any]:
        """
        Generate a map of the file structure for easy navigation.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            
        Returns:
            Dictionary representing the file structure
        """
        file_structure = {}
        
        for item in strategy_run_dir.rglob('*'):
            if item.is_file():
                # Get relative path from strategy_run_dir
                rel_path = item.relative_to(strategy_run_dir)
                
                # Build nested structure
                current_level = file_structure
                parts = rel_path.parts
                
                for part in parts[:-1]:  # All parts except the last (filename)
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                
                # Add the file
                filename = parts[-1]
                current_level[filename] = {
                    'path': str(rel_path),
                    'size_bytes': item.stat().st_size,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
                
        return file_structure
        
    def validate_output_structure(self, strategy_run_dir: Path) -> Dict[str, Any]:
        """
        Validate that the output structure is complete and consistent.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            
        Returns:
            Validation report
        """
        validation_report = {
            'structure_valid': True,
            'validation_timestamp': datetime.now().isoformat(),
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Check required directories
        required_dirs = ['raw_data', 'trades', 'reports', 'analysis', 'visualizations', 'transaction_costs', 'metadata']
        for req_dir in required_dirs:
            dir_path = strategy_run_dir / req_dir
            if dir_path.exists():
                validation_report['checks'][f'{req_dir}_directory'] = 'PASS'
            else:
                validation_report['checks'][f'{req_dir}_directory'] = 'FAIL'
                validation_report['errors'].append(f"Missing required directory: {req_dir}")
                validation_report['structure_valid'] = False
                
        # Check for metadata file
        metadata_file = strategy_run_dir / 'metadata' / 'run_metadata.json'
        if metadata_file.exists():
            validation_report['checks']['metadata_file'] = 'PASS'
        else:
            validation_report['checks']['metadata_file'] = 'FAIL'
            validation_report['errors'].append("Missing run metadata file")
            
        # Check for run summary
        summary_files = list(strategy_run_dir.glob('Run_Summary_*.json'))
        if summary_files:
            validation_report['checks']['run_summary'] = 'PASS'
        else:
            validation_report['checks']['run_summary'] = 'FAIL'
            validation_report['warnings'].append("Missing run summary file")
            
        # Count files in each category
        file_counts = {}
        for subdir in required_dirs:
            subdir_path = strategy_run_dir / subdir
            if subdir_path.exists():
                file_count = len([f for f in subdir_path.rglob('*') if f.is_file()])
                file_counts[subdir] = file_count
                
        validation_report['file_counts'] = file_counts
        
        # Overall validation
        if validation_report['structure_valid']:
            self.logger.info(f"Output structure validation PASSED for {strategy_run_dir}")
        else:
            self.logger.error(f"Output structure validation FAILED for {strategy_run_dir}")
            
        return validation_report
        
    def create_analysis_index(self, strategy_run_dir: Path) -> Path:
        """
        Create an analysis index file for easy navigation and discovery.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            
        Returns:
            Path to analysis index file
        """
        # Scan for all analysis files
        analysis_files = {
            'portfolio_analysis': list(strategy_run_dir.glob('analysis/Portfolio_Analysis_*.json')),
            'cost_analysis': list(strategy_run_dir.glob('transaction_costs/Cost_Analysis_*.json')),
            'ticker_reports': list(strategy_run_dir.glob('reports/*_Report_*.json')),
            'visualizations': list(strategy_run_dir.glob('visualizations/**/*.png')),
            'run_summary': list(strategy_run_dir.glob('Run_Summary_*.json'))
        }
        
        # Create index structure
        analysis_index = {
            'index_type': 'analysis_index',
            'strategy_run_dir': str(strategy_run_dir),
            'created_timestamp': datetime.now().isoformat(),
            'quick_access': {
                'run_summary': str(analysis_files['run_summary'][0]) if analysis_files['run_summary'] else None,
                'portfolio_analysis': str(analysis_files['portfolio_analysis'][0]) if analysis_files['portfolio_analysis'] else None,
                'cost_analysis': str(analysis_files['cost_analysis'][0]) if analysis_files['cost_analysis'] else None
            },
            'file_catalog': {
                category: [str(f.relative_to(strategy_run_dir)) for f in files]
                for category, files in analysis_files.items()
            },
            'navigation_guide': {
                'start_here': 'Run_Summary_*.json - Overall performance summary',
                'detailed_analysis': 'analysis/ - Portfolio and cross-ticker analysis',
                'individual_tickers': 'reports/ - Individual ticker performance',
                'visualizations': 'visualizations/ - Charts and plots organized by category',
                'costs': 'transaction_costs/ - Transaction cost breakdown',
                'raw_data': 'raw_data/ - Base data with signals for debugging'
            }
        }
        
        index_file = strategy_run_dir / 'ANALYSIS_INDEX.json'
        with open(index_file, 'w') as f:
            json.dump(analysis_index, f, indent=2, default=str)
            
        self.logger.info(f"Created analysis index at {index_file}")
        return index_file
        
    def cleanup_legacy_structure(self, legacy_output_dir: Path) -> None:
        """
        Clean up legacy output structure and provide migration guidance.
        
        Args:
            legacy_output_dir: Path to legacy output directory
        """
        self.logger.info(f"Legacy structure cleanup for {legacy_output_dir}")
        
        # Create migration report
        migration_report = {
            'migration_timestamp': datetime.now().isoformat(),
            'legacy_structure_found': legacy_output_dir.exists(),
            'files_found': [],
            'migration_suggestions': []
        }
        
        if legacy_output_dir.exists():
            # Scan legacy files
            for file_path in legacy_output_dir.rglob('*'):
                if file_path.is_file():
                    migration_report['files_found'].append(str(file_path))
                    
            migration_report['migration_suggestions'] = [
                "Consider re-running analysis with new output structure",
                "Legacy files can be manually organized using OutputManager.organize_legacy_files()",
                "New structure provides better organization and analysis capabilities"
            ]
            
        # Save migration report
        migration_file = self.base_output_dir / 'MIGRATION_REPORT.json'
        with open(migration_file, 'w') as f:
            json.dump(migration_report, f, indent=2, default=str)
            
        self.logger.info(f"Migration report saved to {migration_file}")


def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Create output manager
    output_mgr = OutputManager()
    
    # Example: Create structure for a strategy run
    strategy_run_dir = output_mgr.create_strategy_run_structure(
        strategy_name="mse",
        date_range="2025-06-07_to_2025-06-14"
    )
    
    # Example metadata
    metadata = {
        'strategy_name': 'mse',
        'date_range': '2025-06-07_to_2025-06-14',
        'tickers': ['INFY', 'TCS', 'RELIANCE'],
        'parameters': {'lookback_period': 20, 'threshold': 0.02},
        'configuration_file': 'config/mse_config.yaml'
    }
    
    output_mgr.save_run_metadata(strategy_run_dir, metadata)
    
    # Validate structure
    validation_report = output_mgr.validate_output_structure(strategy_run_dir)
    print(f"Validation result: {validation_report['structure_valid']}")
    
    # Create analysis index
    index_file = output_mgr.create_analysis_index(strategy_run_dir)
    print(f"Analysis index created at: {index_file}")
