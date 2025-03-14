#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Infrastructure System for the Bangladesh simulation model.
This module simulates key infrastructure components including transportation,
energy, water, telecommunications, and urban development.
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


class InfrastructureSystem:
    """
    Simulates the infrastructure system of Bangladesh, including
    transportation, energy, water, telecommunications, and urban development.
    """
    
    def __init__(self, config, initial_conditions=None):
        """
        Initialize the infrastructure system with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary
            initial_conditions (dict, optional): Initial state of infrastructure components
        """
        self.config = config
        
        # Transportation infrastructure (0-1 scale)
        self.road_network_coverage = 0.45  # Road network coverage
        self.road_quality = 0.40  # Quality of roads
        self.rail_network_coverage = 0.35  # Railway network coverage
        self.port_capacity = 0.50  # Port infrastructure capacity
        self.airport_capacity = 0.55  # Airport infrastructure capacity
        
        # Energy infrastructure (0-1 scale)
        self.electricity_generation = 0.60  # Electricity generation capacity
        self.electricity_distribution = 0.50  # Electricity distribution network
        self.electricity_access = 0.70  # Population with electricity access
        self.renewable_energy_share = 0.25  # Share of renewable energy
        self.energy_reliability = 0.55  # Reliability of energy supply

        # Water infrastructure (0-1 scale)
        self.water_supply_coverage = 0.65  # Clean water supply coverage
        self.sanitation_coverage = 0.50  # Sanitation infrastructure coverage
        self.water_treatment_capacity = 0.40  # Water treatment capacity
        self.irrigation_coverage = 0.55  # Irrigation infrastructure coverage
        self.water_sanitation = 0.45  # Combined water and sanitation indicator
        
        # Telecommunications infrastructure (0-1 scale)
        self.mobile_network_coverage = 0.80  # Mobile network coverage
        self.internet_coverage = 0.55  # Internet coverage
        self.broadband_access = 0.35  # Broadband access
        self.digital_service_quality = 0.45  # Quality of digital services
        self.telecom_coverage = 0.60  # Overall telecom coverage
        
        # Urban infrastructure (0-1 scale)
        self.urbanization_level = 0.50  # Level of urbanization
        self.urban_planning_quality = 0.40  # Quality of urban planning
        self.housing_quality = 0.45  # Housing infrastructure quality
        self.waste_management = 0.35  # Waste management infrastructure
        self.urban_planning = 0.40  # Overall urban planning indicator
        
        # Set up historical data tracking
        self.history = {
            # Transportation
            'transport_quality': [],
            # Energy
            'energy_reliability': [],
            # Water
            'water_sanitation': [],
            # Telecommunications
            'telecom_coverage': [],
            # Urban development
            'urban_planning': [],
            # Overall index
            'overall_infrastructure_quality': []
        }
        
        # Override defaults with initial conditions if provided
        if initial_conditions:
            self._apply_initial_conditions(initial_conditions)
        
        # Calculate composite indices
        self._calculate_composite_indices()
        
        # Record initial state
        self._record_history()
        
        logger.info("InfrastructureSystem initialized successfully")
    
    def _apply_initial_conditions(self, initial_conditions):
        """Apply initial conditions to infrastructure components."""
        # Apply each initial condition if provided
        for attr, value in initial_conditions.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
    
    def _calculate_composite_indices(self):
        """Calculate composite indices for each infrastructure sector."""
        # Transportation index
        self.transport_quality = (
            self.road_network_coverage * 0.30 +
            self.road_quality * 0.25 +
            self.rail_network_coverage * 0.15 +
            self.port_capacity * 0.15 +
            self.airport_capacity * 0.15
        )
        
        # Energy index
        self.energy_reliability = (
            self.electricity_generation * 0.25 +
            self.electricity_distribution * 0.25 +
            self.electricity_access * 0.25 +
            self.renewable_energy_share * 0.10 +
            self.energy_reliability * 0.15
        )
        
        # Water and sanitation index
        self.water_sanitation = (
            self.water_supply_coverage * 0.30 +
            self.sanitation_coverage * 0.25 +
            self.water_treatment_capacity * 0.20 +
            self.irrigation_coverage * 0.25
        )
        
        # Telecommunications index
        self.telecom_coverage = (
            self.mobile_network_coverage * 0.35 +
            self.internet_coverage * 0.30 +
            self.broadband_access * 0.20 +
            self.digital_service_quality * 0.15
        )
        
        # Urban development index
        self.urban_planning = (
            self.urbanization_level * 0.20 +
            self.urban_planning_quality * 0.30 +
            self.housing_quality * 0.30 +
            self.waste_management * 0.20
        )
        
        # Overall infrastructure quality index
        self.overall_infrastructure_quality = (
            self.transport_quality * 0.25 +
            self.energy_reliability * 0.25 +
            self.water_sanitation * 0.20 +
            self.telecom_coverage * 0.15 +
            self.urban_planning * 0.15
        )
    
    def _record_history(self):
        """Record current state to history."""
        self.history['transport_quality'].append(self.transport_quality)
        self.history['energy_reliability'].append(self.energy_reliability)
        self.history['water_sanitation'].append(self.water_sanitation)
        self.history['telecom_coverage'].append(self.telecom_coverage)
        self.history['urban_planning'].append(self.urban_planning)
        self.history['overall_infrastructure_quality'].append(self.overall_infrastructure_quality)
    
    def update(self, year, economic_growth=None, population_growth=None, investment_rate=None, governance_quality=None):
        """
        Update infrastructure components for the current simulation step.
        
        Args:
            year (int): Current simulation year
            economic_growth (float, optional): Economic growth rate
            population_growth (float, optional): Population growth rate
            investment_rate (float, optional): Infrastructure investment rate
            governance_quality (float, optional): Governance quality index
        
        Returns:
            dict: Updated infrastructure indicators
        """
        # Basic autonomous changes (small improvements over time)
        base_improvement = 0.005  # Base annual improvement
        random_factor = 0.01  # Maximum random fluctuation
        
        # Apply random changes to each component
        for attr in dir(self):
            if attr.endswith('_coverage') or attr.endswith('_quality') or attr.endswith('_capacity'):
                current_value = getattr(self, attr)
                # Small improvement plus random factor
                change = base_improvement + np.random.uniform(-random_factor, random_factor)
                setattr(self, attr, current_value + change)
        
        # Apply economic growth effects if provided
        if economic_growth is not None:
            eco_impact = economic_growth * 0.1
            self.road_network_coverage += eco_impact * 0.2
            self.electricity_generation += eco_impact * 0.25
            self.electricity_distribution += eco_impact * 0.15
            self.internet_coverage += eco_impact * 0.3
        
        # Apply population growth effects if provided
        if population_growth is not None:
            # Population growth can strain infrastructure if too rapid
            pop_impact = -population_growth * 0.15
            self.road_quality += pop_impact
            self.water_supply_coverage += pop_impact
            self.sanitation_coverage += pop_impact
            self.urbanization_level += population_growth * 0.2  # Urbanization increases with population
        
        # Apply investment effects if provided
        if investment_rate is not None:
            inv_impact = investment_rate * 0.2
            self.road_quality += inv_impact * 0.2
            self.electricity_distribution += inv_impact * 0.25
            self.water_treatment_capacity += inv_impact * 0.2
            self.broadband_access += inv_impact * 0.3
            self.urban_planning_quality += inv_impact * 0.15
        
        # Apply governance quality effects if provided
        if governance_quality is not None:
            gov_impact = governance_quality * 0.1
            self.road_quality += gov_impact * 0.15
            self.rail_network_coverage += gov_impact * 0.1
            self.renewable_energy_share += gov_impact * 0.2
            self.water_treatment_capacity += gov_impact * 0.15
            self.waste_management += gov_impact * 0.25
        
        # Ensure all values stay within bounds (0-1)
        self._normalize_components()
        
        # Recalculate composite indices
        self._calculate_composite_indices()
        
        # Record current state
        self._record_history()
        
        return self.get_indicators()
    
    def _normalize_components(self):
        """Ensure all infrastructure components stay within bounds (0-1)."""
        for attr in dir(self):
            if attr.endswith('_coverage') or attr.endswith('_quality') or attr.endswith('_capacity') or \
               attr.endswith('_share') or attr.endswith('_access') or attr.endswith('_level'):
                current_value = getattr(self, attr)
                normalized_value = max(0, min(1, current_value))
                setattr(self, attr, normalized_value)
    
    def get_indicators(self):
        """
        Get the current infrastructure indicators.
        
        Returns:
            dict: Current infrastructure indicators
        """
        return {
            'transport_quality': self.transport_quality,
            'energy_reliability': self.energy_reliability,
            'water_sanitation': self.water_sanitation,
            'telecom_coverage': self.telecom_coverage,
            'urban_planning': self.urban_planning,
            'overall_infrastructure_quality': self.overall_infrastructure_quality
        }
    
    def get_overall_quality(self):
        """
        Get the overall infrastructure quality index.
        
        Returns:
            float: Overall infrastructure quality index (0-1 scale)
        """
        return self.overall_infrastructure_quality
    
    def get_historical_data(self):
        """
        Get historical data for all infrastructure indicators.
        
        Returns:
            dict: Historical data for all indicators
        """
        return self.history
    
    # Methods required for system integration
    def transport_quality_adjustment(self, impact_value):
        """
        Adjust transport quality based on external factors.
        
        Args:
            impact_value (float): Impact value to apply
        """
        self.road_quality += impact_value * 0.5
        self.rail_network_coverage += impact_value * 0.3
        self.port_capacity += impact_value * 0.2
        self._normalize_components()
        self._calculate_composite_indices()
        logger.debug(f"Adjusted transport quality by {impact_value:.4f}")
    
    def energy_reliability_adjustment(self, impact_value):
        """
        Adjust energy reliability based on external factors.
        
        Args:
            impact_value (float): Impact value to apply
        """
        self.electricity_generation += impact_value * 0.3
        self.electricity_distribution += impact_value * 0.4
        self.energy_reliability += impact_value * 0.3
        self._normalize_components()
        self._calculate_composite_indices()
        logger.debug(f"Adjusted energy reliability by {impact_value:.4f}")
    
    def water_sanitation_adjustment(self, impact_value):
        """
        Adjust water and sanitation infrastructure based on external factors.
        
        Args:
            impact_value (float): Impact value to apply
        """
        self.water_supply_coverage += impact_value * 0.4
        self.sanitation_coverage += impact_value * 0.4
        self.water_treatment_capacity += impact_value * 0.2
        self._normalize_components()
        self._calculate_composite_indices()
        logger.debug(f"Adjusted water sanitation by {impact_value:.4f}")
    
    def telecom_coverage_adjustment(self, impact_value):
        """
        Adjust telecommunications coverage based on external factors.
        
        Args:
            impact_value (float): Impact value to apply
        """
        self.mobile_network_coverage += impact_value * 0.3
        self.internet_coverage += impact_value * 0.4
        self.broadband_access += impact_value * 0.3
        self._normalize_components()
        self._calculate_composite_indices()
        logger.debug(f"Adjusted telecom coverage by {impact_value:.4f}")
    
    def urban_planning_adjustment(self, impact_value):
        """
        Adjust urban planning quality based on external factors.
        
        Args:
            impact_value (float): Impact value to apply
        """
        self.urban_planning_quality += impact_value * 0.6
        self.waste_management += impact_value * 0.4
        self._normalize_components()
        self._calculate_composite_indices()
        logger.debug(f"Adjusted urban planning by {impact_value:.4f}")

    def generate_simulated_data(self, start_year, end_year):
        """
        Generate simulated historical data for infrastructure indicators.
        
        Args:
            start_year (int): Start year for simulation
            end_year (int): End year for simulation
            
        Returns:
            pd.DataFrame: Simulated infrastructure data
        """
        years = range(start_year, end_year + 1)
        data = {
            'Year': list(years),
            'transport_quality': [],
            'energy_reliability': [],
            'water_sanitation': [],
            'telecom_coverage': [],
            'urban_planning': [],
            'overall_infrastructure_quality': []
        }
        
        # Start with baseline values
        transport = 0.40
        energy = 0.45
        water = 0.50
        telecom = 0.45
        urban = 0.35
        overall = 0.42
        
        # Generate data with trends and some randomness
        for year in years:
            # Add trend component (gradual improvement over time)
            trend_factor = (year - start_year) / (end_year - start_year) * 0.2
            
            # Add random component
            random_factor = np.random.normal(0, 0.02)
            
            # Calculate values for this year
            transport = min(1.0, max(0.0, transport + trend_factor * 0.02 + random_factor))
            energy = min(1.0, max(0.0, energy + trend_factor * 0.025 + random_factor))
            water = min(1.0, max(0.0, water + trend_factor * 0.015 + random_factor))
            telecom = min(1.0, max(0.0, telecom + trend_factor * 0.03 + random_factor))  # Telecom grows faster
            urban = min(1.0, max(0.0, urban + trend_factor * 0.01 + random_factor))
            
            # Overall index is weighted average
            overall = (
                transport * 0.25 +
                energy * 0.25 +
                water * 0.20 +
                telecom * 0.15 +
                urban * 0.15
            )
            
            # Add values to data dictionary
            data['transport_quality'].append(transport)
            data['energy_reliability'].append(energy)
            data['water_sanitation'].append(water)
            data['telecom_coverage'].append(telecom)
            data['urban_planning'].append(urban)
            data['overall_infrastructure_quality'].append(overall)
        
        return pd.DataFrame(data).set_index('Year')
