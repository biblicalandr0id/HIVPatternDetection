"""
Configuration Management System
Created: 2025-02-13 05:03:43
Author: biblicalandr0id
Version: 1.0.0

Centralized configuration management for all HIV simulation systems.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import logging

@dataclass
class SystemConfig:
    """System-wide configuration state"""
    timestamp: datetime
    core_simulation: Dict[str, Any]
    pattern_recognition: Dict[str, Any]
    resource_dynamics: Dict[str, Any]
    enhanced_processing: Dict[str, Any]
    data_management: Dict[str, Any]
    emergency_protocols: Dict[str, Any]
    integrated_visualization: Dict[str, Any]
    validation: Dict[str, Any]
    logging: Dict[str, Any]

class ConfigurationManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.timestamp = datetime.strptime("2025-02-13 05:03:43", "%Y-%m-%d %H:%M:%S")
            self.user = "biblicalandr0id"
            self.config_path = Path("config")
            self.config_path.mkdir(exist_ok=True)
            self.state = self._initialize_state()
            self.initialized = True
            
    def _initialize_state(self) -> SystemConfig:
        """Initialize configuration state"""
        return SystemConfig(
            timestamp=self.timestamp,
            core_simulation=self._load_default_core_config(),
            pattern_recognition=self._load_default_pattern_config(),
            resource_dynamics=self._load_default_resource_config(),
            enhanced_processing=self._load_default_processing_config(),
            data_management=self._load_default_data_config(),
            emergency_protocols=self._load_default_emergency_config(),
            integrated_visualization=self._load_default_visualization_config(),
            validation=self._load_default_validation_config(),
            logging=self._load_default_logging_config()
        )
    
    def get_config(self, system_name: str) -> Dict[str, Any]:
        """Get configuration for specific system"""
        return getattr(self.state, system_name, {})
    
    def update_config(self, system_name: str, 
                     config_updates: Dict[str, Any]) -> bool:
        """Update configuration for specific system"""
        try:
            current_config = getattr(self.state, system_name, {})
            updated_config = {**current_config, **config_updates}
            
            # Validate before applying
            if self._validate_config(system_name, updated_config):
                setattr(self.state, system_name, updated_config)
                self._save_config(system_name, updated_config)
                return True
            return False
            
        except Exception as e:
            logging.error(f"Config update failed: {str(e)}")
            return False
    
    def _load_default_core_config(self) -> Dict[str, Any]:
        """Load default core simulation configuration"""
        return {
            'time_step': 0.01,
            'max_duration': 1000,
            'integration_method': 'RK4',
            'tolerance': 1e-6,
            'validate_parameters': True
        }
    
    def _load_default_pattern_config(self) -> Dict[str, Any]:
        """Load default pattern recognition configuration"""
        return {
            'confidence_threshold': 0.95,
            'pattern_database_path': 'patterns.db',
            'max_patterns': 1000,
            'analysis_window': 100
        }
    
    def _load_default_resource_config(self) -> Dict[str, Any]:
        """Load default resource dynamics configuration"""
        return {
            'optimization_threshold': 0.8,
            'emergency_threshold': 0.2,
            'neural_adaptation_rate': 0.1,
            'resource_check_interval': 0.1
        }
    
    def _load_default_processing_config(self) -> Dict[str, Any]:
        """Load default enhanced processing configuration"""
        return {
            'use_gpu': True,
            'batch_size': 64,
            'num_workers': 4,
            'optimization_level': 'maximum'
        }
    
    def _load_default_data_config(self) -> Dict[str, Any]:
        """Load default data management configuration"""
        return {
            'storage_path': 'data/',
            'backup_interval': 3600,
            'compression_level': 9,
            'encryption_enabled': True
        }
    
    def _load_default_emergency_config(self) -> Dict[str, Any]:
        """Load default emergency protocols configuration"""
        return {
            'auto_response_enabled': True,
            'notification_endpoints': ['admin@system'],
            'recovery_attempts': 3,
            'stability_threshold': 0.5
        }
    
    def _load_default_visualization_config(self) -> Dict[str, Any]:
        """Load default visualization configuration"""
        return {
            'dashboard_port': 8050,
            'update_interval': 1.0,
            'max_data_points': 1000,
            'plot_theme': 'dark'
        }
    
    def _load_default_validation_config(self) -> Dict[str, Any]:
        """Load default validation configuration"""
        return {
            'validate_output': True,
            'constraint_tolerance': 1e-6,
            'biological_constraints_enabled': True,
            'validation_frequency': 10
        }
    
    def _load_default_logging_config(self) -> Dict[str, Any]:
        """Load default logging configuration"""
        return {
            'log_level': 'INFO',
            'log_path': 'logs/',
            'rotate_logs': True,
            'max_log_size': 1024 * 1024  # 1MB
        }
    
    def _validate_config(self, system_name: str, 
                        config: Dict[str, Any]) -> bool:
        """Validate configuration for specific system"""
        validators = {
            'core_simulation': self._validate_core_config,
            'pattern_recognition': self._validate_pattern_config,
            'resource_dynamics': self._validate_resource_config,
            'enhanced_processing': self._validate_processing_config,
            'data_management': self._validate_data_config,
            'emergency_protocols': self._validate_emergency_config,
            'integrated_visualization': self._validate_visualization_config,
            'validation': self._validate_validation_config,
            'logging': self._validate_logging_config
        }
        
        validator = validators.get(system_name)
        if validator:
            return validator(config)
        return False
    
    def _save_config(self, system_name: str, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        config_file = self.config_path / f"{system_name}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
    
    def load_all_configs(self) -> None:
        """Load all configuration files"""
        for system_name in vars(self.state):
            if system_name != 'timestamp':
                config_file = self.config_path / f"{system_name}.yaml"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        if self._validate_config(system_name, config):
                            setattr(self.state, system_name, config)
