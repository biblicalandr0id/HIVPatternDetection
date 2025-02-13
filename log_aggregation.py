"""
Logging Aggregation System
Created: 2025-02-13 05:06:51
Author: biblicalandr0id
Version: 1.0.0

Centralized logging system that aggregates, analyzes, and manages logs
from all HIV simulation system components.
"""

import logging
from logging.handlers import RotatingFileHandler
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import queue
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re

@dataclass
class LogState:
    """Logging system state"""
    timestamp: datetime
    active_loggers: Dict[str, logging.Logger]
    log_levels: Dict[str, int]
    log_paths: Dict[str, Path]
    aggregation_active: bool
    analysis_enabled: bool
    alert_threshold: int
    rotation_size: int
    backup_count: int
    pattern_filters: List[str]

class LogAggregator:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.timestamp = datetime.strptime("2025-02-13 05:06:51", "%Y-%m-%d %H:%M:%S")
            self.user = "biblicalandr0id"
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)
            self.state = self._initialize_state()
            self.log_queue = queue.Queue()
            self.analysis_thread = None
            self.executor = ThreadPoolExecutor(max_workers=4)
            self._setup_logging()
            self.initialized = True
    
    def _initialize_state(self) -> LogState:
        """Initialize logging system state"""
        return LogState(
            timestamp=self.timestamp,
            active_loggers={},
            log_levels={
                'core_simulation': logging.INFO,
                'pattern_recognition': logging.INFO,
                'resource_dynamics': logging.INFO,
                'enhanced_processing': logging.INFO,
                'data_management': logging.INFO,
                'emergency_protocols': logging.WARNING,
                'integrated_visualization': logging.INFO,
                'validation': logging.INFO,
                'config_management': logging.INFO
            },
            log_paths={},
            aggregation_active=True,
            analysis_enabled=True,
            alert_threshold=100,
            rotation_size=1024 * 1024,  # 1MB
            backup_count=5,
            pattern_filters=[
                r'ERROR',
                r'CRITICAL',
                r'WARNING',
                r'EMERGENCY'
            ]
        )

    def _setup_logging(self) -> None:
        """Setup logging system"""
        for system_name, level in self.state.log_levels.items():
            logger = logging.getLogger(system_name)
            logger.setLevel(level)
            
            # Create log file handler
            log_path = self.log_dir / f"{system_name}.log"
            handler = RotatingFileHandler(
                log_path,
                maxBytes=self.state.rotation_size,
                backupCount=self.state.backup_count
            )
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(handler)
            
            # Store in state
            self.state.active_loggers[system_name] = logger
            self.state.log_paths[system_name] = log_path

    def get_logger(self, system_name: str) -> logging.Logger:
        """Get logger for specific system"""
        if system_name not in self.state.active_loggers:
            self._setup_system_logger(system_name)
        return self.state.active_loggers[system_name]

    def aggregate_logs(self) -> pd.DataFrame:
        """Aggregate logs from all systems"""
        try:
            all_logs = []
            
            for system_name, log_path in self.state.log_paths.items():
                system_logs = self._read_system_logs(log_path)
                if system_logs:
                    all_logs.extend(system_logs)
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(all_logs)
            
            # Sort by timestamp
            if not df.empty and 'timestamp' in df.columns:
                df.sort_values('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self._handle_aggregation_error(e)
            return pd.DataFrame()

    def analyze_logs(self, timeframe: str = '1h') -> Dict[str, Any]:
        """Analyze aggregated logs"""
        try:
            logs_df = self.aggregate_logs()
            if logs_df.empty:
                return {}
            
            analysis = {
                'timestamp': self.timestamp,
                'total_logs': len(logs_df),
                'error_count': len(logs_df[logs_df['level'] == 'ERROR']),
                'warning_count': len(logs_df[logs_df['level'] == 'WARNING']),
                'critical_count': len(logs_df[logs_df['level'] == 'CRITICAL']),
                'system_statistics': self._calculate_system_statistics(logs_df),
                'pattern_matches': self._find_patterns(logs_df),
                'trend_analysis': self._analyze_trends(logs_df, timeframe)
            }
            
            # Check alert conditions
            self._check_alert_conditions(analysis)
            
            return analysis
            
        except Exception as e:
            self._handle_analysis_error(e)
            return {}

    def generate_report(self, report_type: str = 'summary') -> Dict[str, Any]:
        """Generate log analysis report"""
        try:
            analysis = self.analyze_logs()
            
            if report_type == 'summary':
                return self._generate_summary_report(analysis)
            elif report_type == 'detailed':
                return self._generate_detailed_report(analysis)
            elif report_type == 'alert':
                return self._generate_alert_report(analysis)
            else:
                return self._generate_custom_report(analysis, report_type)
                
        except Exception as e:
            self._handle_report_error(e)
            return {}

    def _read_system_logs(self, log_path: Path) -> List[Dict[str, Any]]:
        """Read logs from a specific system"""
        try:
            logs = []
            if log_path.exists():
                with open(log_path, 'r') as f:
                    for line in f:
                        log_entry = self._parse_log_entry(line)
                        if log_entry:
                            logs.append(log_entry)
            return logs
            
        except Exception as e:
            self._handle_read_error(e, log_path)
            return []

    def _parse_log_entry(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a log entry line"""
        try:
            # Example format: 2025-02-13 05:06:51,123 - system_name - LEVEL - message
            pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)'
            match = re.match(pattern, line)
            
            if match:
                timestamp_str, system, level, message = match.groups()
                return {
                    'timestamp': datetime.strptime(timestamp_str, 
                                                 '%Y-%m-%d %H:%M:%S,%f'),
                    'system': system,
                    'level': level,
                    'message': message.strip()
                }
            return None
            
        except Exception:
            return None

    def _calculate_system_statistics(self, 
                                   logs_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for each system"""
        stats = {}
        
        for system in logs_df['system'].unique():
            system_logs = logs_df[logs_df['system'] == system]
            stats[system] = {
                'total_logs': len(system_logs),
                'error_rate': len(system_logs[system_logs['level'] == 'ERROR']) / len(system_logs),
                'warning_rate': len(system_logs[system_logs['level'] == 'WARNING']) / len(system_logs),
                'critical_rate': len(system_logs[system_logs['level'] == 'CRITICAL']) / len(system_logs)
            }
            
        return stats

    def _find_patterns(self, logs_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find patterns in log entries"""
        patterns = []
        
        for pattern in self.state.pattern_filters:
            matches = logs_df[logs_df['message'].str.contains(pattern, 
                                                            na=False)]
            if not matches.empty:
                patterns.append({
                    'pattern': pattern,
                    'count': len(matches),
                    'systems_affected': matches['system'].unique().tolist()
                })
                
        return patterns

    def _analyze_trends(self, logs_df: pd.DataFrame, 
                       timeframe: str) -> Dict[str, Any]:
        """Analyze trends in log data"""
        try:
            # Convert timeframe to timedelta
            if timeframe.endswith('h'):
                hours = int(timeframe[:-1])
                start_time = self.timestamp - pd.Timedelta(hours=hours)
            else:
                start_time = self.timestamp - pd.Timedelta(hours=1)
            
            # Filter logs by timeframe
            recent_logs = logs_df[logs_df['timestamp'] >= start_time]
            
            return {
                'total_trend': self._calculate_trend(recent_logs['timestamp']),
                'error_trend': self._calculate_trend(
                    recent_logs[recent_logs['level'] == 'ERROR']['timestamp']
                ),
                'system_trends': self._calculate_system_trends(recent_logs)
            }
            
        except Exception as e:
            self._handle_analysis_error(e)
            return {}

    def _calculate_trend(self, timestamps: pd.Series) -> str:
        """Calculate trend direction from timestamp series"""
        if len(timestamps) < 2:
            return 'stable'
            
        counts = timestamps.value_counts(bins=10)
        slope = np.polyfit(range(len(counts)), counts, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def _check_alert_conditions(self, analysis: Dict[str, Any]) -> None:
        """Check if any alert conditions are met"""
        if analysis['error_count'] > self.state.alert_threshold:
            self._trigger_alert('High error count detected', analysis)
            
        for system, stats in analysis['system_statistics'].items():
            if stats['error_rate'] > 0.2:  # 20% error rate threshold
                self._trigger_alert(f'High error rate in {system}', stats)

    def _trigger_alert(self, message: str, data: Dict[str, Any]) -> None:
        """Trigger alert for serious conditions"""
        alert_logger = self.get_logger('alerts')
        alert_logger.critical(f"ALERT: {message}\nData: {json.dumps(data)}")

    def _handle_aggregation_error(self, error: Exception) -> None:
        """Handle log aggregation errors"""
        error_logger = self.get_logger('system')
        error_logger.error(f"Log aggregation error: {str(error)}")

    def _handle_analysis_error(self, error: Exception) -> None:
        """Handle log analysis errors"""
        error_logger = self.get_logger('system')
        error_logger.error(f"Log analysis error: {str(error)}")

    def _handle_report_error(self, error: Exception) -> None:
        """Handle report generation errors"""
        error_logger = self.get_logger('system')
        error_logger.error(f"Report generation error: {str(error)}")

    def __del__(self):
        """Cleanup logging system"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)