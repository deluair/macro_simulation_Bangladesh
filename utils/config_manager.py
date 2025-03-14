#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration manager for the Bangladesh simulation model.
This module handles loading, validating, and providing access to configuration settings.
"""

import os
import yaml
import json
from pathlib import Path


class ConfigManager:
    """
    Utility for loading and managing configuration settings for the simulation model.
    Handles loading the main configuration file and any component-specific configuration files.
    """
    
    def __init__(self, config_path):
        """
        Initialize the configuration manager with path to main config file.
        
        Args:
            config_path (str): Path to the main configuration file
        """
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            os.makedirs(self.config_dir, exist_ok=True)
            print(f"Created config directory: {self.config_dir}")
        
        # Try to load config file, create default if not found
        if not self.config_path.exists():
            self._create_default_config()
            print(f"Created default configuration file: {self.config_path}")
        
        # Load the configuration
        self.config = self._load_config()
        print(f"Configuration loaded from: {config_path}")
        
        # Validate configuration
        self._validate_config()
    
    def _create_default_config(self):
        """Create a default configuration file if none exists."""
        default_config = {
            'simulation': {
                'name': 'Bangladesh Integrated System Model',
                'description': 'Simulation of Bangladesh socioeconomic and environmental systems',
                'start_year': 2023,
                'end_year': 2050,
                'time_step': 1,  # Years
                'random_seed': 42,
                'monte_carlo': {
                    'enabled': False,
                    'n_runs': 100,
                    'parameters': [
                        {'name': 'economic.growth_volatility', 'min': 0.005, 'max': 0.02},
                        {'name': 'environmental.disaster_frequency', 'min': 0.8, 'max': 1.5}
                    ]
                }
            },
            'data': {
                'data_dir': 'data',
                'output_dir': 'output',
                'gis_enabled': True
            },
            'output': {
                'output_dir': 'output',
                'save_frequency': 1,  # Save every n time steps
                'formats': ['csv', 'json'],
                'visualization': {
                    'enabled': True,
                    'dashboard': True,
                    'maps': True,
                    'plots': True
                }
            },
            'economic': {
                'growth_volatility': 0.01,
                'sector_linkage_strength': 0.7,
                'international_trade_dependence': 0.6,
                'remittance_dependence': 0.4,
                'sectors': {
                    'agriculture': {'initial_share': 0.15, 'growth_potential': 0.04},
                    'manufacturing': {'initial_share': 0.30, 'growth_potential': 0.08},
                    'services': {'initial_share': 0.50, 'growth_potential': 0.06},
                    'informal': {'initial_share': 0.05, 'growth_potential': 0.03}
                },
                'financial_system': {
                    'depth': 0.6,
                    'stability': 0.7,
                    'inclusion': 0.5
                }
            },
            'environmental': {
                'climate_change_scenario': 'ssp245',  # SSP2-4.5 moderate scenario
                'temperature_trend': 0.04,  # Annual increase °C
                'precipitation_volatility': 0.1,
                'sea_level_rise_rate': 0.01,  # Meters per year
                'disaster_parameters': {
                    'base_frequency': 1.0,
                    'intensity_trend': 0.02  # Annual increase in intensity
                },
                'land_degradation_rate': 0.005,  # Annual rate
                'forest_loss_rate': 0.01,  # Annual rate
                'pollution': {
                    'air_quality_trend': -0.02,  # Declining
                    'water_quality_trend': -0.015  # Declining
                }
            },
            'demographic': {
                'fertility_trend': -0.02,  # Annual rate of change
                'mortality_trend': -0.01,  # Annual rate of change
                'migration': {
                    'rural_urban_rate': 0.02,  # Annual rate
                    'international_rate': 0.005  # Annual rate
                },
                'education': {
                    'enrollment_trend': 0.01,  # Annual increase
                    'quality_trend': 0.005  # Annual increase
                },
                'urbanization_rate': 0.015  # Annual rate
            },
            'infrastructure': {
                'investment_rate': 0.08,  # Share of GDP
                'maintenance_factor': 0.6,  # Proportion of optimal maintenance
                'technology_adoption_rate': 0.05,  # Annual rate
                'climate_vulnerability': 0.7,  # Index 0-1
                'transport': {
                    'road_expansion_rate': 0.03,
                    'rail_expansion_rate': 0.02,
                    'port_expansion_rate': 0.04
                },
                'energy': {
                    'generation_growth_rate': 0.06,
                    'renewable_transition_rate': 0.03,
                    'efficiency_improvement_rate': 0.02
                },
                'water': {
                    'supply_expansion_rate': 0.04,
                    'sanitation_expansion_rate': 0.05
                },
                'telecom': {
                    'mobile_expansion_rate': 0.03,
                    'internet_expansion_rate': 0.08,
                    'broadband_expansion_rate': 0.10
                },
                'urban': {
                    'planning_quality': 0.5,
                    'housing_improvement_rate': 0.03
                }
            },
            'governance': {
                'institutional_improvement_rate': 0.01,
                'corruption_reduction_rate': 0.02,
                'policy_effectiveness': 0.6,
                'governance_shocks': {
                    'frequency': 0.3,  # Probability per year
                    'magnitude': 0.2   # Impact magnitude
                },
                'policy_coherence': 0.5,
                'regional_cooperation': 0.4
            }
        }
        
        # Save the default configuration to file
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    def _load_config(self):
        """
        Load the main configuration file.
        
        Returns:
            dict: Loaded configuration
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Load component specific configs if referenced
            if 'component_configs' in config:
                for component, comp_path in config['component_configs'].items():
                    comp_config_path = self.config_dir / comp_path
                    if comp_config_path.exists():
                        with open(comp_config_path, 'r') as f:
                            comp_config = yaml.safe_load(f)
                            config[component] = comp_config
            
            return config
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration instead.")
            self._create_default_config()
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
    
    def _validate_config(self):
        """Validate the configuration for required elements and value ranges."""
        
        # Check essential configuration sections
        required_sections = ['simulation', 'data', 'output', 'economic', 
                            'environmental', 'demographic', 'infrastructure', 'governance']
        
        missing_sections = [section for section in required_sections if section not in self.config]
        if missing_sections:
            print(f"Warning: Missing configuration sections: {missing_sections}")
            for section in missing_sections:
                self.config[section] = {}
        
        # Validate simulation timeframe
        if 'simulation' in self.config:
            if self.config['simulation'].get('start_year', 0) >= self.config['simulation'].get('end_year', 0):
                print("Warning: Invalid simulation timeframe. Setting default 2023-2050.")
                self.config['simulation']['start_year'] = 2023
                self.config['simulation']['end_year'] = 2050
        
        # Ensure data and output directories exist
        if 'data' in self.config and 'data_dir' in self.config['data']:
            data_dir = Path(self.config['data']['data_dir'])
            if not data_dir.exists():
                os.makedirs(data_dir, exist_ok=True)
                print(f"Created data directory: {data_dir}")
        
        if 'output' in self.config and 'output_dir' in self.config['output']:
            output_dir = Path(self.config['output']['output_dir'])
            if not output_dir.exists():
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
    
    def save_config(self, config=None):
        """
        Save the current configuration to file.
        
        Args:
            config (dict, optional): Configuration to save. If None, save the current config.
        """
        if config is None:
            config = self.config
            
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
            print(f"Configuration saved to: {self.config_path}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_component_config(self, component_name):
        """
        Get configuration for a specific component.
        
        Args:
            component_name (str): Name of the component
            
        Returns:
            dict: Component configuration or empty dict if not found
        """
        return self.config.get(component_name, {})
    
    def update_component_config(self, component_name, new_config):
        """
        Update configuration for a specific component.
        
        Args:
            component_name (str): Name of the component
            new_config (dict): New configuration for the component
        """
        # Update the component config
        if component_name in self.config:
            self.config[component_name].update(new_config)
        else:
            self.config[component_name] = new_config
            
        # Save the updated configuration
        self.save_config()
