#!/usr/bin/env python3
"""
Execution Logger Utility
Provides comprehensive logging capabilities for backtesting operations
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class ExecutionLogger:
    """
    Comprehensive logging system for tracking backtesting execution.

    Features:
    - File creation/failure tracking
    - Performance metrics
    - Task generation logging
    - Error and warning capture
    - Execution phase tracking
    - JSON-based log export
    """

    def __init__(self, session_id: str = None):
        """
        Initialize execution logger.

        Args:
            session_id: Unique session identifier (auto-generated if None)
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)

        # Initialize execution log structure
        self.execution_log = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'config_snapshot': {},
            'tasks_generated': [],
            'files_created': [],
            'files_failed': [],
            'execution_phases': [],
            'errors': [],
            'warnings': [],
            'performance_metrics': {},
            'ticker_processing': {},
            'strategy_execution': {}
        }

    def log_execution_phase(self, phase: str, message: str, details: Dict[str, Any] = None):
        """Log execution phase with timestamp and details."""
        phase_entry = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'details': details or {}
        }
        self.execution_log['execution_phases'].append(phase_entry)
        self.logger.info(f"[{phase.upper()}] {message}")
        if details:
            self.logger.debug(f"[{phase.upper()}] Details: {details}")

    def log_file_creation(self, file_path: str, file_type: str, status: str = "success",
                         error_msg: str = None, metadata: Dict[str, Any] = None):
        """Log file creation attempts with detailed information."""
        file_entry = {
            'file_path': str(file_path),
            'file_type': file_type,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'error_message': error_msg,
            'metadata': metadata or {}
        }

        # Add file size if successful
        if status == "success" and Path(file_path).exists():
            file_entry['metadata']['file_size_bytes'] = Path(file_path).stat().st_size

        if status == "success":
            self.execution_log['files_created'].append(file_entry)
            self.logger.info(f"✅ File created: {file_path} ({file_type})")
        else:
            self.execution_log['files_failed'].append(file_entry)
            self.logger.error(f"❌ File creation failed: {file_path} ({file_type}) - {error_msg}")

    def log_task_generation(self, tasks: List, discovery_info: Dict[str, Any] = None):
        """Log task generation with comprehensive details."""
        task_info = {
            'total_tasks': len(tasks),
            'timestamp': datetime.now().isoformat(),
            'discovery_info': discovery_info or {},
            'task_breakdown': {}
        }

        # Analyze task composition
        strategies = set()
        tickers = set()
        date_ranges = set()

        for task in tasks:
            if len(task) >= 3:
                ticker, date_range, strategy = task[:3]
                tickers.add(ticker)
                date_ranges.add(date_range)
                strategies.add(strategy)

        task_info['task_breakdown'] = {
            'unique_strategies': list(strategies),
            'unique_tickers': list(tickers),
            'unique_date_ranges': list(date_ranges),
            'combinations': len(tasks)
        }

        self.execution_log['tasks_generated'] = task_info

        self.logger.info(f"📋 Generated {len(tasks)} tasks:")
        self.logger.info(f"   - Strategies: {len(strategies)} ({list(strategies)})")
        self.logger.info(f"   - Tickers: {len(tickers)} ({list(tickers)})")
        self.logger.info(f"   - Date Ranges: {len(date_ranges)} ({list(date_ranges)})")

        if discovery_info:
            self.logger.info(f"   - Discovery: {discovery_info}")

    def log_ticker_processing(self, ticker: str, status: str, details: Dict[str, Any] = None):
        """Log individual ticker processing status."""
        if ticker not in self.execution_log['ticker_processing']:
            self.execution_log['ticker_processing'][ticker] = []

        processing_entry = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }

        self.execution_log['ticker_processing'][ticker].append(processing_entry)

        status_emoji = "✅" if status == "success" else "❌" if status == "failed" else "⏳"
        self.logger.info(f"{status_emoji} Ticker {ticker}: {status}")
        if details:
            self.logger.debug(f"   Details: {details}")

    def log_strategy_execution(self, strategy: str, ticker: str, date_range: str,
                              status: str, metrics: Dict[str, Any] = None):
        """Log strategy execution details."""
        strategy_key = f"{strategy}_{ticker}_{date_range}"

        execution_entry = {
            'strategy': strategy,
            'ticker': ticker,
            'date_range': date_range,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {}
        }

        self.execution_log['strategy_execution'][strategy_key] = execution_entry

        status_emoji = "✅" if status == "success" else "❌" if status == "failed" else "⏳"
        self.logger.info(f"{status_emoji} Strategy {strategy} on {ticker} ({date_range}): {status}")

        if metrics:
            self.logger.debug(f"   Metrics: {metrics}")

    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics throughout execution."""
        timestamp = datetime.now().isoformat()
        self.execution_log['performance_metrics'][timestamp] = metrics
        self.logger.info(f"⏱️ Performance: {metrics}")

    def log_error(self, error_msg: str, context: str = None, traceback_str: str = None):
        """Log error with context."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_message': error_msg,
            'context': context,
            'traceback': traceback_str
        }
        self.execution_log['errors'].append(error_entry)
        self.logger.error(f"❌ Error: {error_msg}")
        if context:
            self.logger.error(f"   Context: {context}")

    def log_warning(self, warning_msg: str, context: str = None):
        """Log warning with context."""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'warning_message': warning_msg,
            'context': context
        }
        self.execution_log['warnings'].append(warning_entry)
        self.logger.warning(f"⚠️ Warning: {warning_msg}")
        if context:
            self.logger.warning(f"   Context: {context}")

    def save_execution_log(self, output_dir: str = None) -> Optional[str]:
        """Save comprehensive execution log to file."""
        try:
            # Finalize execution log
            self.execution_log['end_time'] = datetime.now().isoformat()
            self.execution_log['total_execution_time'] = (
                datetime.now() - self.start_time
            ).total_seconds()

            # Determine output directory
            if output_dir:
                log_dir = Path(output_dir) / "logs"
            else:
                log_dir = Path("logs")

            log_dir.mkdir(parents=True, exist_ok=True)

            # Create detailed log file
            log_file = log_dir / f"execution_log_{self.session_id}.json"

            with open(log_file, 'w') as f:
                json.dump(self.execution_log, f, indent=2, default=str)

            # Create summary log file
            summary = self._create_execution_summary()
            summary_file = log_dir / f"execution_summary_{self.session_id}.json"

            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            # Create human-readable report
            report_file = log_dir / f"execution_report_{self.session_id}.txt"
            self._create_human_readable_report(report_file)

            self.logger.info(f"📊 Execution logs saved:")
            self.logger.info(f"   - Detailed log: {log_file}")
            self.logger.info(f"   - Summary: {summary_file}")
            self.logger.info(f"   - Report: {report_file}")

            return str(log_file)

        except Exception as e:
            self.logger.error(f"Failed to save execution log: {e}")
            return None

    def _create_execution_summary(self) -> Dict[str, Any]:
        """Create execution summary."""
        task_info = self.execution_log.get('tasks_generated', {})

        return {
            'session_id': self.session_id,
            'execution_time_seconds': self.execution_log.get('total_execution_time', 0),
            'total_tasks': task_info.get('total_tasks', 0),
            'files_created_count': len(self.execution_log['files_created']),
            'files_failed_count': len(self.execution_log['files_failed']),
            'errors_count': len(self.execution_log['errors']),
            'warnings_count': len(self.execution_log['warnings']),
            'tickers_processed': len(self.execution_log['ticker_processing']),
            'strategies_executed': len(self.execution_log['strategy_execution']),
            'status': self._determine_overall_status()
        }

    def _determine_overall_status(self) -> str:
        """Determine overall execution status."""
        if len(self.execution_log['errors']) > 0:
            return 'completed_with_errors'
        elif len(self.execution_log['warnings']) > 0:
            return 'completed_with_warnings'
        else:
            return 'success'

    def _create_human_readable_report(self, report_file: Path):
        """Create human-readable execution report."""
        try:
            with open(report_file, 'w') as f:
                f.write(f"BACKTESTING EXECUTION REPORT\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Start Time: {self.execution_log['start_time']}\n")
                f.write(f"End Time: {self.execution_log.get('end_time', 'In Progress')}\n")
                f.write(f"Duration: {self.execution_log.get('total_execution_time', 0):.2f} seconds\n")
                f.write(f"Status: {self._determine_overall_status()}\n\n")

                # Task summary
                task_info = self.execution_log.get('tasks_generated', {})
                if task_info:
                    f.write(f"TASK SUMMARY\n")
                    f.write(f"-" * 20 + "\n")
                    f.write(f"Total Tasks: {task_info.get('total_tasks', 0)}\n")

                    breakdown = task_info.get('task_breakdown', {})
                    f.write(f"Strategies: {len(breakdown.get('unique_strategies', []))}\n")
                    f.write(f"Tickers: {len(breakdown.get('unique_tickers', []))}\n")
                    f.write(f"Date Ranges: {len(breakdown.get('unique_date_ranges', []))}\n\n")

                # File summary
                f.write(f"FILE OPERATIONS\n")
                f.write(f"-" * 20 + "\n")
                f.write(f"Files Created: {len(self.execution_log['files_created'])}\n")
                f.write(f"Files Failed: {len(self.execution_log['files_failed'])}\n\n")

                # Error summary
                if self.execution_log['errors']:
                    f.write(f"ERRORS ({len(self.execution_log['errors'])})\n")
                    f.write(f"-" * 20 + "\n")
                    for error in self.execution_log['errors']:
                        f.write(f"• {error['timestamp']}: {error['error_message']}\n")
                    f.write("\n")

                # Warning summary
                if self.execution_log['warnings']:
                    f.write(f"WARNINGS ({len(self.execution_log['warnings'])})\n")
                    f.write(f"-" * 20 + "\n")
                    for warning in self.execution_log['warnings']:
                        f.write(f"• {warning['timestamp']}: {warning['warning_message']}\n")
                    f.write("\n")

                # Performance metrics
                if self.execution_log['performance_metrics']:
                    f.write(f"PERFORMANCE METRICS\n")
                    f.write(f"-" * 20 + "\n")
                    for timestamp, metrics in self.execution_log['performance_metrics'].items():
                        f.write(f"{timestamp}: {metrics}\n")

        except Exception as e:
            self.logger.error(f"Failed to create human-readable report: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get current execution summary."""
        return self._create_execution_summary()