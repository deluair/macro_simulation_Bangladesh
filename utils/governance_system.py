#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Governance System for the Bangladesh simulation model.
This module simulates governance aspects including institutional effectiveness,
policy-making, regulatory quality, and corruption.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GovernanceSystem:
    """
    Simulates the governance system of Bangladesh, including institutional
    effectiveness, policy-making capacity, regulatory quality, and anti-corruption efforts.
    """
    
    def __init__(self, config, initial_conditions=None):
        """
        Initialize the governance system with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary
            initial_conditions (dict, optional): Initial state of governance indicators
        """
        self.config = config
        
        # Initialize governance indicators (0-1 scale, higher is better)
        self.institutional_effectiveness = 0.4  # Base institutional capability
        self.corruption_index = 0.6  # Higher values mean more corruption
        self.policy_effectiveness = 0.5  # Effectiveness of policy implementation
        self.regulatory_quality = 0.45  # Quality of regulatory framework
        self.political_stability = 0.55  # Political stability indicator
        self.public_service_delivery = 0.5  # Public service delivery quality
        
        # Set up historical data tracking
        self.history = {
            'institutional_effectiveness': [],
            'corruption_index': [],
            'policy_effectiveness': [],
            'regulatory_quality': [],
            'political_stability': [],
            'public_service_delivery': []
        }
        
        # Override defaults with initial conditions if provided
        if initial_conditions:
            self._apply_initial_conditions(initial_conditions)
        
        # Record initial state
        self._record_history()
        
        logger.info("GovernanceSystem initialized successfully")
    
    def _apply_initial_conditions(self, initial_conditions):
        """Apply initial conditions to governance indicators."""
        if 'institutional_effectiveness' in initial_conditions:
            self.institutional_effectiveness = initial_conditions['institutional_effectiveness']
        if 'corruption_index' in initial_conditions:
            self.corruption_index = initial_conditions['corruption_index']
        if 'policy_effectiveness' in initial_conditions:
            self.policy_effectiveness = initial_conditions['policy_effectiveness']
        if 'regulatory_quality' in initial_conditions:
            self.regulatory_quality = initial_conditions['regulatory_quality']
        if 'political_stability' in initial_conditions:
            self.political_stability = initial_conditions['political_stability']
        if 'public_service_delivery' in initial_conditions:
            self.public_service_delivery = initial_conditions['public_service_delivery']
    
    def _record_history(self):
        """Record current state to history."""
        self.history['institutional_effectiveness'].append(self.institutional_effectiveness)
        self.history['corruption_index'].append(self.corruption_index)
        self.history['policy_effectiveness'].append(self.policy_effectiveness)
        self.history['regulatory_quality'].append(self.regulatory_quality)
        self.history['political_stability'].append(self.political_stability)
        self.history['public_service_delivery'].append(self.public_service_delivery)
    
    def update(self, year, economic_growth=None, education_level=None):
        """
        Update governance indicators for the current simulation step.
        
        Args:
            year (int): Current simulation year
            economic_growth (float, optional): Economic growth rate
            education_level (float, optional): Education level indicator
        
        Returns:
            dict: Updated governance indicators
        """
        # Basic autonomous changes (small random fluctuations)
        random_factor = 0.02  # Maximum random change
        
        # Apply random changes with trend
        self.institutional_effectiveness += np.random.uniform(-random_factor, random_factor * 1.5)
        self.corruption_index += np.random.uniform(-random_factor * 1.2, random_factor)
        self.policy_effectiveness += np.random.uniform(-random_factor, random_factor * 1.3)
        self.regulatory_quality += np.random.uniform(-random_factor, random_factor * 1.4)
        self.political_stability += np.random.uniform(-random_factor * 1.5, random_factor * 1.5)
        
        # Apply economic growth effects if provided
        if economic_growth is not None:
            # Economic growth tends to improve governance over time
            eco_impact = economic_growth * 0.1
            self.institutional_effectiveness += eco_impact * 0.3
            self.corruption_index -= eco_impact * 0.2  # Less corruption
            self.policy_effectiveness += eco_impact * 0.25
            self.regulatory_quality += eco_impact * 0.2
        
        # Apply education effects if provided
        if education_level is not None:
            # Higher education improves governance
            edu_impact = education_level * 0.05
            self.institutional_effectiveness += edu_impact
            self.corruption_index -= edu_impact * 1.2  # Stronger effect on reducing corruption
            self.policy_effectiveness += edu_impact * 0.8
        
        # Calculate public service delivery based on other indicators
        self.public_service_delivery = (
            self.institutional_effectiveness * 0.4 +
            (1 - self.corruption_index) * 0.3 +  # Invert corruption impact
            self.policy_effectiveness * 0.3
        )
        
        # Ensure all values stay within bounds (0-1)
        self._normalize_indicators()
        
        # Record current state
        self._record_history()
        
        return self.get_indicators()
    
    def _normalize_indicators(self):
        """Ensure all governance indicators stay within bounds (0-1)."""
        self.institutional_effectiveness = max(0, min(1, self.institutional_effectiveness))
        self.corruption_index = max(0, min(1, self.corruption_index))
        self.policy_effectiveness = max(0, min(1, self.policy_effectiveness))
        self.regulatory_quality = max(0, min(1, self.regulatory_quality))
        self.political_stability = max(0, min(1, self.political_stability))
        self.public_service_delivery = max(0, min(1, self.public_service_delivery))
    
    def get_indicators(self):
        """
        Get the current governance indicators.
        
        Returns:
            dict: Current governance indicators
        """
        return {
            'institutional_effectiveness': self.institutional_effectiveness,
            'corruption_index': self.corruption_index,
            'policy_effectiveness': self.policy_effectiveness,
            'regulatory_quality': self.regulatory_quality,
            'political_stability': self.political_stability,
            'public_service_delivery': self.public_service_delivery
        }
    
    def get_overall_governance_index(self):
        """
        Calculate an overall governance index based on all indicators.
        
        Returns:
            float: Overall governance index (0-1 scale)
        """
        return (
            self.institutional_effectiveness * 0.25 +
            (1 - self.corruption_index) * 0.25 +  # Invert corruption impact
            self.policy_effectiveness * 0.2 +
            self.regulatory_quality * 0.15 +
            self.political_stability * 0.15
        )
    
    def get_historical_data(self):
        """
        Get historical data for all governance indicators.
        
        Returns:
            dict: Historical data for all indicators
        """
        return self.history
    
    # Methods required for system integration
    def adjust_institutional_effectiveness(self, impact_value):
        """
        Adjust institutional effectiveness based on external factors.
        
        Args:
            impact_value (float): Impact value to apply
        """
        self.institutional_effectiveness += impact_value
        self._normalize_indicators()
        logger.debug(f"Adjusted institutional effectiveness by {impact_value:.4f}")
    
    def adjust_policy_effectiveness(self, impact_value):
        """
        Adjust policy effectiveness based on external factors.
        
        Args:
            impact_value (float): Impact value to apply
        """
        self.policy_effectiveness += impact_value
        self._normalize_indicators()
        logger.debug(f"Adjusted policy effectiveness by {impact_value:.4f}")
    
    def adjust_public_service_delivery(self, impact_value):
        """
        Adjust public service delivery based on external factors.
        
        Args:
            impact_value (float): Impact value to apply
        """
        self.public_service_delivery += impact_value
        self._normalize_indicators()
        logger.debug(f"Adjusted public service delivery by {impact_value:.4f}")

    def generate_simulated_data(self, start_year, end_year):
        """
        Generate simulated historical data for governance indicators.
        
        Args:
            start_year (int): Start year for simulation
            end_year (int): End year for simulation
            
        Returns:
            pd.DataFrame: Simulated governance data
        """
        years = range(start_year, end_year + 1)
        data = {
            'Year': list(years),
            'institutional_effectiveness': [],
            'corruption_index': [],
            'policy_effectiveness': [],
            'regulatory_quality': [],
            'political_stability': [],
            'public_service_delivery': []
        }
        
        # Start with baseline values
        inst_eff = 0.35
        corruption = 0.65
        policy_eff = 0.4
        reg_quality = 0.38
        pol_stability = 0.5
        pub_service = 0.4
        
        # Generate data with trends and some randomness
        for year in years:
            # Add trend component (gradual improvement over time)
            trend_factor = (year - start_year) / (end_year - start_year) * 0.15
            
            # Add random component
            random_factor = np.random.normal(0, 0.03)
            
            # Calculate values for this year
            inst_eff = min(1.0, max(0.0, inst_eff + trend_factor * 0.02 + random_factor))
            corruption = min(1.0, max(0.0, corruption - trend_factor * 0.025 + random_factor))
            policy_eff = min(1.0, max(0.0, policy_eff + trend_factor * 0.02 + random_factor))
            reg_quality = min(1.0, max(0.0, reg_quality + trend_factor * 0.015 + random_factor))
            pol_stability = min(1.0, max(0.0, pol_stability + trend_factor * 0.01 + random_factor * 1.5))
            
            # Public service is derived from other indicators
            pub_service = min(1.0, max(0.0, 
                inst_eff * 0.4 + (1 - corruption) * 0.3 + policy_eff * 0.3
            ))
            
            # Add values to data dictionary
            data['institutional_effectiveness'].append(inst_eff)
            data['corruption_index'].append(corruption)
            data['policy_effectiveness'].append(policy_eff)
            data['regulatory_quality'].append(reg_quality)
            data['political_stability'].append(pol_stability)
            data['public_service_delivery'].append(pub_service)
        
        return pd.DataFrame(data).set_index('Year')
