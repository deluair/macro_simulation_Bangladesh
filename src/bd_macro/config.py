"""
Configuration module for the Bangladesh Development Simulation Model.

This module handles loading, validation, and management of simulation parameters.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for the simulation model."""
    
    # Simulation parameters
    start_year: int
    end_year: int
    time_step: int
    
    # Model parameters
    population_growth_rate: float
    urbanization_rate: float
    gdp_growth_rate: float
    investment_rate: float
    savings_rate: float
    
    # Sector-specific parameters
    agriculture_growth: float
    industry_growth: float
    services_growth: float
    
    # Environmental parameters
    carbon_emissions_factor: float
    deforestation_rate: float
    water_consumption_rate: float
    
    # Infrastructure parameters
    infrastructure_investment: float
    maintenance_rate: float
    efficiency_factor: float
    
    # Social parameters
    education_investment: float
    healthcare_investment: float
    poverty_reduction_rate: float
    
    # Governance parameters
    corruption_index: float
    policy_effectiveness: float
    institutional_strength: float
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """Create a SimulationConfig instance from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'start_year': self.start_year,
            'end_year': self.end_year,
            'time_step': self.time_step,
            'population_growth_rate': self.population_growth_rate,
            'urbanization_rate': self.urbanization_rate,
            'gdp_growth_rate': self.gdp_growth_rate,
            'investment_rate': self.investment_rate,
            'savings_rate': self.savings_rate,
            'agriculture_growth': self.agriculture_growth,
            'industry_growth': self.industry_growth,
            'services_growth': self.services_growth,
            'carbon_emissions_factor': self.carbon_emissions_factor,
            'deforestation_rate': self.deforestation_rate,
            'water_consumption_rate': self.water_consumption_rate,
            'infrastructure_investment': self.infrastructure_investment,
            'maintenance_rate': self.maintenance_rate,
            'efficiency_factor': self.efficiency_factor,
            'education_investment': self.education_investment,
            'healthcare_investment': self.healthcare_investment,
            'poverty_reduction_rate': self.poverty_reduction_rate,
            'corruption_index': self.corruption_index,
            'policy_effectiveness': self.policy_effectiveness,
            'institutional_strength': self.institutional_strength
        }

class ConfigManager:
    """Manages simulation configuration loading and validation."""
    
    def __init__(self, config_dir: str = 'config'):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.current_config: Optional[SimulationConfig] = None
    
    def load_config(self, config_file: str) -> SimulationConfig:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Name of the configuration file
            
        Returns:
            SimulationConfig instance
        """
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            self.current_config = SimulationConfig.from_dict(config_dict)
            logger.info(f"Loaded configuration from {config_path}")
            return self.current_config
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_config(self, config: SimulationConfig, config_file: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config: Configuration to save
            config_file: Name of the configuration file
        """
        config_path = self.config_dir / config_file
        try:
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=4)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate_config(self, config: SimulationConfig) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check time parameters
            assert config.start_year >= 2000
            assert config.end_year > config.start_year
            assert config.time_step > 0
            
            # Check growth rates
            assert -1 <= config.population_growth_rate <= 1
            assert -1 <= config.urbanization_rate <= 1
            assert -1 <= config.gdp_growth_rate <= 1
            
            # Check investment and savings rates
            assert 0 <= config.investment_rate <= 1
            assert 0 <= config.savings_rate <= 1
            
            # Check sector growth rates
            assert -1 <= config.agriculture_growth <= 1
            assert -1 <= config.industry_growth <= 1
            assert -1 <= config.services_growth <= 1
            
            # Check environmental parameters
            assert 0 <= config.carbon_emissions_factor <= 1
            assert 0 <= config.deforestation_rate <= 1
            assert 0 <= config.water_consumption_rate <= 1
            
            # Check infrastructure parameters
            assert 0 <= config.infrastructure_investment <= 1
            assert 0 <= config.maintenance_rate <= 1
            assert 0 <= config.efficiency_factor <= 1
            
            # Check social parameters
            assert 0 <= config.education_investment <= 1
            assert 0 <= config.healthcare_investment <= 1
            assert 0 <= config.poverty_reduction_rate <= 1
            
            # Check governance parameters
            assert 0 <= config.corruption_index <= 1
            assert 0 <= config.policy_effectiveness <= 1
            assert 0 <= config.institutional_strength <= 1
            
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def create_default_config(self) -> SimulationConfig:
        """
        Create a default configuration with reasonable values.
        
        Returns:
            SimulationConfig instance with default values
        """
        config = SimulationConfig(
            start_year=2020,
            end_year=2050,
            time_step=1,
            population_growth_rate=0.015,
            urbanization_rate=0.02,
            gdp_growth_rate=0.06,
            investment_rate=0.3,
            savings_rate=0.25,
            agriculture_growth=0.04,
            industry_growth=0.08,
            services_growth=0.07,
            carbon_emissions_factor=0.5,
            deforestation_rate=0.01,
            water_consumption_rate=0.02,
            infrastructure_investment=0.2,
            maintenance_rate=0.05,
            efficiency_factor=0.7,
            education_investment=0.15,
            healthcare_investment=0.1,
            poverty_reduction_rate=0.03,
            corruption_index=0.4,
            policy_effectiveness=0.6,
            institutional_strength=0.5
        )
        
        if self.validate_config(config):
            return config
        else:
            raise ValueError("Default configuration validation failed") 