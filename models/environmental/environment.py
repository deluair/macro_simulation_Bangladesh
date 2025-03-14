#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Environmental system model for Bangladesh simulation.
This module implements the riverine delta system, climate change impacts,
extreme weather events, and environmental feedback mechanisms.
"""

import numpy as np
import pandas as pd
from scipy.stats import gamma, weibull_min
import geopandas as gpd

from models.environmental.climate import ClimateSystem
from models.environmental.water import WaterSystem
from models.environmental.disasters import DisasterSystem
from models.environmental.land import LandSystem


class EnvironmentalSystem:
    """
    Environmental system model representing Bangladesh's complex delta system,
    climate patterns, and environmental dynamics.
    """
    
    def __init__(self, config, data_loader):
        """
        Initialize the environmental system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the environmental system
            data_loader (DataLoader): Data loading utility for environmental data
        """
        self.config = config
        self.data_loader = data_loader
        
        # Load initial environmental data
        self.environmental_data = data_loader.load_environmental_data()
        
        # Set up time-related variables
        self.current_year = config.get('start_year', 2023)
        self.time_step = config.get('time_step', 1.0)
        self.base_year = config.get('base_year', 2000)
        
        # Initialize climate parameters
        self.temperature_anomaly = self.environmental_data.get('temperature_anomaly', 0.8)  # Degrees C above pre-industrial
        self.sea_level_rise = self.environmental_data.get('sea_level_rise', 0.3)  # Meters since 1900
        self.co2_concentration = self.environmental_data.get('co2_concentration', 410.0)  # ppm
        
        # Regional climate parameters
        self.regional_rainfall = self.environmental_data.get('annual_rainfall', 2500.0)  # mm/year
        self.monsoon_intensity = self.environmental_data.get('monsoon_intensity', 1.0)  # normalized
        self.monsoon_timing_shift = self.environmental_data.get('monsoon_timing_shift', 0.0)  # days
        
        # Initialize environmental subsystems
        self._init_subsystems()
        
        # Initialize extreme event tracking
        self.extreme_events = []
        self.current_extreme_event = None
        self.extreme_event_impact = 0.0
        
        # Environmental quality indicators
        self.air_quality_index = self.environmental_data.get('air_quality_index', 150.0)  # 0-500 scale (higher is worse)
        self.water_quality_index = self.environmental_data.get('water_quality_index', 65.0)  # 0-100 scale (higher is better)
        self.forest_coverage = self.environmental_data.get('forest_coverage', 0.11)  # 11% of land area
        self.biodiversity_index = self.environmental_data.get('biodiversity_index', 0.7)  # 0-1 scale
        
        # Environmental stress indicators
        self.water_stress_index = self.environmental_data.get('water_stress_index', 0.4)  # 0-1 scale
        self.land_degradation_index = self.environmental_data.get('land_degradation_index', 0.35)  # 0-1 scale
        self.salinity_intrusion = self.environmental_data.get('salinity_intrusion', 0.2)  # 0-1 scale
        
        # Climate adaptation parameters
        self.adaptation_investment = self.environmental_data.get('adaptation_investment', 0.01)  # % of GDP
        self.adaptation_effectiveness = self.environmental_data.get('adaptation_effectiveness', 0.5)  # 0-1 scale
        
        print("Environmental system initialized")
    
    def _init_subsystems(self):
        """Initialize all environmental subsystems."""
        print("Initializing environmental subsystems...")
        
        # Initialize climate system (temperature, rainfall, sea level rise)
        self.climate_system = ClimateSystem(
            config=self.config.get('climate', {}),
            environmental_data=self.environmental_data
        )
        
        # Initialize water system (rivers, groundwater, flooding)
        self.water_system = WaterSystem(
            config=self.config.get('water', {}),
            environmental_data=self.environmental_data
        )
        
        # Initialize disaster system (cyclones, floods, droughts)
        self.disaster_system = DisasterSystem(
            config=self.config.get('disasters', {}),
            environmental_data=self.environmental_data
        )
        
        # Initialize land system (land use, soil, erosion)
        self.land_system = LandSystem(
            config=self.config.get('land', {}),
            environmental_data=self.environmental_data
        )
        
        print("All environmental subsystems initialized.")
    
    def update_global_climate(self, year):
        """
        Update global climate parameters based on IPCC scenarios.
        
        Args:
            year (int): Current simulation year
            
        Returns:
            dict: Updated climate parameters
        """
        # Get climate scenario from config
        scenario = self.config.get('climate_scenario', 'rcp45')  # Default to RCP 4.5
        
        # Time since base year
        years_since_base = year - self.base_year
        
        # Simplified IPCC-based projections for different RCP scenarios
        if scenario == 'rcp26':  # Low emissions scenario
            temp_increase_rate = 0.01  # °C per year
            sea_level_rise_rate = 0.004  # m per year
            co2_increase_rate = 0.8  # ppm per year
        elif scenario == 'rcp45':  # Moderate emissions scenario
            temp_increase_rate = 0.025
            sea_level_rise_rate = 0.007
            co2_increase_rate = 1.5
        elif scenario == 'rcp85':  # High emissions scenario
            temp_increase_rate = 0.04
            sea_level_rise_rate = 0.01
            co2_increase_rate = 2.5
        else:
            raise ValueError(f"Unknown climate scenario: {scenario}")
        
        # Add some natural variability to the trends
        temp_variability = np.random.normal(0, 0.05)
        sea_level_variability = np.random.normal(0, 0.001)
        co2_variability = np.random.normal(0, 0.3)
        
        # Calculate new values with compound effects
        years_elapsed = year - self.current_year
        self.temperature_anomaly += (temp_increase_rate + temp_variability) * years_elapsed
        self.sea_level_rise += (sea_level_rise_rate + sea_level_variability) * years_elapsed
        self.co2_concentration += (co2_increase_rate + co2_variability) * years_elapsed
        
        # Update current year
        self.current_year = year
        
        # Return updated climate parameters
        climate_params = {
            'temperature_anomaly': self.temperature_anomaly,
            'sea_level_rise': self.sea_level_rise,
            'co2_concentration': self.co2_concentration,
        }
        
        return climate_params
    
    def update_regional_climate(self, global_params):
        """
        Downscale global climate parameters to regional Bangladesh conditions.
        
        Args:
            global_params (dict): Global climate parameters
            
        Returns:
            dict: Regional climate parameters
        """
        # Bangladesh typically experiences amplified warming compared to global average
        regional_temp_factor = 1.2
        regional_temp_anomaly = global_params['temperature_anomaly'] * regional_temp_factor
        
        # Monsoon changes with warming
        # Warming generally intensifies the monsoon but with more variability
        baseline_monsoon_intensity = 1.0
        monsoon_intensity_change = 0.05 * regional_temp_anomaly + np.random.normal(0, 0.1)
        self.monsoon_intensity = baseline_monsoon_intensity + monsoon_intensity_change
        
        # Gradual shift in monsoon timing (days) - generally delayed with warming
        monsoon_timing_change = 0.5 * regional_temp_anomaly + np.random.normal(0, 1)
        self.monsoon_timing_shift += monsoon_timing_change
        
        # Rainfall changes - generally increased total but more variability
        rainfall_pct_change = (0.03 * regional_temp_anomaly + np.random.normal(0, 0.05)) * 100
        baseline_rainfall = 2500.0  # mm/year
        self.regional_rainfall = baseline_rainfall * (1 + rainfall_pct_change / 100)
        
        # Return regional climate parameters
        regional_params = {
            'regional_temperature_anomaly': regional_temp_anomaly,
            'monsoon_intensity': self.monsoon_intensity,
            'monsoon_timing_shift': self.monsoon_timing_shift,
            'annual_rainfall': self.regional_rainfall,
            'rainfall_pct_change': rainfall_pct_change
        }
        
        return regional_params
    
    def update_environmental_quality(self, economic_system=None, demographic_system=None):
        """
        Update environmental quality indicators based on economic and demographic pressures.
        
        Args:
            economic_system: Economic system for industrial activity
            demographic_system: Demographic system for population pressure
            
        Returns:
            dict: Updated environmental quality indicators
        """
        # Base environmental quality trends (slow degradation)
        base_air_quality_change = 1.0  # AQI units per year (higher is worse)
        base_water_quality_change = -0.2  # percentage points per year (lower is worse)
        base_forest_change = -0.001  # percentage points per year (deforestation)
        base_biodiversity_change = -0.005  # index points per year (loss)
        
        # Economic activity impacts
        economic_pressure = 0.0
        if economic_system:
            # Industrial output increases pollution
            industrial_output = economic_system.get_sector_outputs().get('industrial', 50.0)
            technology_level = economic_system.technology_sector.technology_level if hasattr(economic_system, 'technology_sector') else 0.5
            
            # Technology improves efficiency and reduces pollution per unit output
            economic_pressure = 0.01 * (industrial_output / 50.0) * (1 - 0.5 * technology_level)
        
        # Demographic pressure
        demographic_pressure = 0.0
        if demographic_system:
            # Access total_population as an attribute, not a method
            population = demographic_system.total_population
            urbanization = demographic_system.urbanization_rate
            
            # Urban population creates more environmental pressure
            demographic_pressure = 0.005 * (population / 165000000) * (1 + urbanization)
        
        # Combined pressures
        environmental_pressure = economic_pressure + demographic_pressure
        
        # Update environmental quality indicators
        # Air quality (higher AQI is worse)
        self.air_quality_index += base_air_quality_change + 20 * environmental_pressure
        self.air_quality_index = max(min(self.air_quality_index, 500), 0)
        
        # Water quality (higher percentage is better)
        self.water_quality_index += base_water_quality_change - 15 * environmental_pressure
        self.water_quality_index = max(min(self.water_quality_index, 100), 0)
        
        # Forest coverage
        self.forest_coverage += base_forest_change - 0.005 * environmental_pressure
        self.forest_coverage = max(min(self.forest_coverage, 0.3), 0.05)
        
        # Biodiversity
        self.biodiversity_index += base_biodiversity_change - 0.03 * environmental_pressure
        self.biodiversity_index = max(min(self.biodiversity_index, 1.0), 0.0)
        
        # Return environmental quality indicators
        quality_indicators = {
            'air_quality_index': self.air_quality_index,
            'water_quality_index': self.water_quality_index,
            'forest_coverage': self.forest_coverage,
            'biodiversity_index': self.biodiversity_index,
            'environmental_pressure': environmental_pressure
        }
        
        return quality_indicators
    
    def update_environmental_stress(self, global_climate, regional_climate):
        """
        Update environmental stress indicators based on climate changes.
        
        Args:
            global_climate (dict): Global climate parameters
            regional_climate (dict): Regional climate parameters
            
        Returns:
            dict: Updated environmental stress indicators
        """
        # Sea level rise effects on salinity intrusion
        sea_level_effect = 0.1 * (global_climate['sea_level_rise'] - 0.3)
        
        # Rainfall effects on water stress
        rainfall_effect = -0.1 * (regional_climate['rainfall_pct_change'] / 100)
        
        # Temperature effects on land degradation
        temperature_effect = 0.05 * (regional_climate['regional_temperature_anomaly'] - 0.8)
        
        # Update water stress
        self.water_stress_index += rainfall_effect + 0.02 * temperature_effect
        self.water_stress_index = max(min(self.water_stress_index, 1.0), 0.0)
        
        # Update land degradation
        self.land_degradation_index += 0.01 + temperature_effect
        self.land_degradation_index = max(min(self.land_degradation_index, 1.0), 0.0)
        
        # Update salinity intrusion
        self.salinity_intrusion += 0.01 + sea_level_effect
        self.salinity_intrusion = max(min(self.salinity_intrusion, 1.0), 0.0)
        
        # Return environmental stress indicators
        stress_indicators = {
            'water_stress_index': self.water_stress_index,
            'land_degradation_index': self.land_degradation_index,
            'salinity_intrusion': self.salinity_intrusion,
            'sea_level_effect': sea_level_effect,
            'rainfall_effect': rainfall_effect,
            'temperature_effect': temperature_effect
        }
        
        return stress_indicators
    
    def update_adaptation(self, governance_system=None, economic_system=None):
        """
        Update climate adaptation parameters based on governance and economic inputs.
        
        Args:
            governance_system: Governance system for adaptation policies
            economic_system: Economic system for adaptation investments
            
        Returns:
            dict: Updated adaptation parameters
        """
        # Base adaptation trends
        base_investment_change = 0.001  # percentage points per year
        
        # Governance effects on adaptation
        governance_effect = 0.0
        if governance_system:
            climate_policy = governance_system.get_climate_policy()
            institutional_efficiency = governance_system.get_institutional_efficiency()
            
            governance_effect = 0.002 * climate_policy + 0.001 * institutional_efficiency
        
        # Economic effects on adaptation
        economic_effect = 0.0
        if economic_system:
            gdp = economic_system.gdp
            gdp_growth = economic_system.gdp_growth_rate
            
            # Higher GDP growth enables more adaptation investment
            economic_effect = 0.001 * gdp_growth
        
        # Update adaptation investment (% of GDP)
        self.adaptation_investment += base_investment_change + governance_effect + economic_effect
        self.adaptation_investment = max(min(self.adaptation_investment, 0.05), 0.0)
        
        # Update adaptation effectiveness based on investment and governance
        effectiveness_change = 0.02 * self.adaptation_investment
        if governance_system:
            effectiveness_change += 0.01 * governance_system.get_institutional_efficiency()
            
        self.adaptation_effectiveness += effectiveness_change
        self.adaptation_effectiveness = max(min(self.adaptation_effectiveness, 1.0), 0.0)
        
        # Return adaptation parameters
        adaptation_params = {
            'adaptation_investment': self.adaptation_investment,
            'adaptation_effectiveness': self.adaptation_effectiveness,
            'governance_effect': governance_effect,
            'economic_effect': economic_effect
        }
        
        return adaptation_params
    
    def generate_extreme_events(self, year, climate_params, stress_indicators):
        """
        Generate extreme weather events based on climate conditions.
        
        Args:
            year (int): Current simulation year
            climate_params (dict): Climate parameters
            stress_indicators (dict): Environmental stress indicators
            
        Returns:
            list: Extreme events generated
        """
        # Get extreme event generation from disaster subsystem
        extreme_events = self.disaster_system.generate_events(
            year=year,
            temperature_anomaly=climate_params.get('regional_temperature_anomaly', 0.0),
            sea_level_rise=climate_params.get('sea_level_rise', 0.0),
            monsoon_intensity=climate_params.get('monsoon_intensity', 1.0),
            water_stress=stress_indicators.get('water_stress_index', 0.5)
        )
        
        # Store events
        self.extreme_events = extreme_events
        
        # Calculate combined impact
        if extreme_events:
            self.extreme_event_impact = sum(event['impact'] for event in extreme_events)
        else:
            self.extreme_event_impact = 0.0
            
        return extreme_events
    
    def step(self, year, demographic_system=None, infrastructure_system=None, economic_system=None):
        """
        Advance the environmental system by one time step.
        
        Args:
            year (int): Current simulation year
            demographic_system: Demographic system for population pressure
            infrastructure_system: Infrastructure system for environmental interactions
            economic_system: Economic system for industrial activity
            
        Returns:
            dict: Results of the environmental step
        """
        print(f"Environmental system step: year {year}")
        
        # Update global climate parameters
        global_climate = self.update_global_climate(year)
        
        # Update regional climate conditions
        regional_climate = self.update_regional_climate(global_climate)
        
        # Climate system step
        climate_results = self.climate_system.step(
            year=year,
            global_climate=global_climate,
            regional_climate=regional_climate
        )
        
        # Update environmental quality
        quality_indicators = self.update_environmental_quality(
            economic_system=economic_system,
            demographic_system=demographic_system
        )
        
        # Update environmental stress indicators
        stress_indicators = self.update_environmental_stress(
            global_climate=global_climate,
            regional_climate=regional_climate
        )
        
        # Water system step
        water_results = self.water_system.step(
            year=year,
            climate_results=climate_results,
            demographic_system=demographic_system,
            infrastructure_system=infrastructure_system
        )
        
        # Land system step
        land_results = self.land_system.step(
            year=year,
            climate_results=climate_results,
            water_results=water_results,
            demographic_system=demographic_system,
            economic_system=economic_system
        )
        
        # Generate extreme events
        extreme_events = self.generate_extreme_events(
            year=year,
            climate_params=regional_climate,
            stress_indicators=stress_indicators
        )
        
        # Update adaptation parameters
        adaptation_params = self.update_adaptation(
            governance_system=None,  # Will be connected in simulation.py
            economic_system=economic_system
        )
        
        # Compile results
        results = {
            'year': year,
            'global_climate': global_climate,
            'regional_climate': regional_climate,
            'climate': climate_results,
            'water': water_results,
            'land': land_results,
            'quality_indicators': quality_indicators,
            'stress_indicators': stress_indicators,
            'extreme_events': extreme_events,
            'extreme_event_impact': self.extreme_event_impact,
            'adaptation': adaptation_params
        }
        
        return results
    
    def get_agricultural_productivity(self):
        """
        Get agricultural productivity modifier based on environmental conditions.
        
        Returns:
            float: Agricultural productivity modifier (1.0 is baseline)
        """
        # Factors affecting agricultural productivity
        rainfall_adequacy = 1.0
        if hasattr(self.climate_system, 'rainfall_adequacy'):
            rainfall_adequacy = self.climate_system.rainfall_adequacy
            
        flood_impact = 0.0
        if hasattr(self.water_system, 'flood_extent'):
            flood_impact = 0.2 * self.water_system.flood_extent
            
        drought_impact = 0.0
        if hasattr(self.water_system, 'drought_severity'):
            drought_impact = 0.3 * self.water_system.drought_severity
            
        salinity_impact = 0.0
        if hasattr(self.water_system, 'salinity_intrusion'):
            salinity_impact = 0.15 * self.water_system.salinity_intrusion
            
        temperature_impact = 0.0
        if hasattr(self.climate_system, 'temperature_anomaly'):
            # Moderate warming initially helps, but then harms
            temp_anomaly = self.climate_system.temperature_anomaly
            temperature_impact = 0.05 * temp_anomaly - 0.1 * (temp_anomaly ** 2)
        
        # Combined productivity modifier
        productivity_modifier = (
            1.0 + 
            0.3 * (rainfall_adequacy - 1.0) -
            flood_impact -
            drought_impact -
            salinity_impact +
            temperature_impact
        )
        
        # Adaptation partially offsets negative impacts
        if productivity_modifier < 1.0:
            adaptation_offset = (1.0 - productivity_modifier) * self.adaptation_effectiveness
            productivity_modifier += adaptation_offset
        
        return max(productivity_modifier, 0.5)  # Floor at 50% productivity
    
    def get_flood_impacts(self):
        """
        Get current flood impacts for other systems.
        
        Returns:
            float: Flood impact index (0-1 scale)
        """
        if hasattr(self.water_system, 'flood_extent'):
            return self.water_system.flood_extent
        else:
            return 0.0
    
    def get_cyclone_damage(self):
        """
        Get current cyclone damage for other systems.
        
        Returns:
            float: Cyclone damage index (0-1 scale)
        """
        if self.extreme_events:
            cyclones = [e for e in self.extreme_events if e['type'] == 'cyclone']
            if cyclones:
                return max(event['impact'] for event in cyclones)
        return 0.0
    
    def get_drought_severity(self):
        """
        Get current drought severity for other systems.
        
        Returns:
            float: Drought severity index (0-1 scale)
        """
        if hasattr(self.water_system, 'drought_severity'):
            return self.water_system.drought_severity
        else:
            return 0.0
    
    def get_environmental_indicators(self):
        """
        Get key environmental indicators for other systems.
        
        Returns:
            dict: Key environmental indicators
        """
        return {
            'temperature_anomaly': self.temperature_anomaly,
            'sea_level_rise': self.sea_level_rise,
            'annual_rainfall': self.regional_rainfall,
            'forest_coverage': self.forest_coverage,
            'water_stress_index': self.water_stress_index,
            'land_degradation_index': self.land_degradation_index,
            'salinity_intrusion': self.salinity_intrusion,
            'air_quality_index': self.air_quality_index,
            'water_quality_index': self.water_quality_index,
            'extreme_event_impact': self.extreme_event_impact
        }
