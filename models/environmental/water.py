#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Water system model for Bangladesh simulation.
This module handles river systems, flooding, groundwater, and water resources.
"""

import numpy as np
import pandas as pd
from scipy.stats import gamma


class WaterSystem:
    """
    Water system model representing Bangladesh's complex river network,
    groundwater resources, flooding dynamics, and water management.
    """
    
    def __init__(self, config, environmental_data):
        """
        Initialize the water system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the water system
            environmental_data (dict): Initial environmental data
        """
        self.config = config
        
        # River system parameters
        self.river_discharge = environmental_data.get('river_discharge', {
            'brahmaputra': 20000.0,  # m³/s - annual average
            'ganges': 11000.0,
            'meghna': 5000.0,
            'others': 3000.0
        })
        
        # Flooding parameters
        self.flood_extent = environmental_data.get('flood_extent', 0.2)  # Fraction of land area
        self.flood_duration = environmental_data.get('flood_duration', 45)  # Days
        self.flood_depth = environmental_data.get('flood_depth', 1.5)  # Meters average
        
        # Drought parameters
        self.drought_severity = environmental_data.get('drought_severity', 0.15)  # 0-1 scale
        self.drought_duration = environmental_data.get('drought_duration', 60)  # Days
        
        # Groundwater parameters
        self.groundwater_level = environmental_data.get('groundwater_level', 1.0)  # Normalized (1.0 is historical baseline)
        self.groundwater_recharge = environmental_data.get('groundwater_recharge', 0.9)  # Normalized
        self.groundwater_extraction = environmental_data.get('groundwater_extraction', 1.2)  # Normalized (>1 means over-extraction)
        
        # Water quality and stress indicators
        self.water_quality = environmental_data.get('water_quality', 0.65)  # 0-1 scale
        self.salinity_intrusion = environmental_data.get('salinity_intrusion', 0.3)  # 0-1 scale
        self.arsenic_contamination = environmental_data.get('arsenic_contamination', 0.25)  # Fraction of wells
        
        # Water management parameters
        self.irrigation_coverage = environmental_data.get('irrigation_coverage', 0.55)  # Fraction of cropland
        self.flood_protection = environmental_data.get('flood_protection', 0.3)  # Fraction of vulnerable areas
        self.water_treatment = environmental_data.get('water_treatment', 0.25)  # Fraction of water supply
        
        print("Water system initialized")
    
    def calculate_river_discharge(self, monthly_rainfall, upstream_flow=None):
        """
        Calculate monthly river discharge based on rainfall and upstream conditions.
        
        Args:
            monthly_rainfall (dict): Monthly rainfall amounts
            upstream_flow (dict): Upstream flow conditions (if any)
            
        Returns:
            dict: Monthly river discharge by major river
        """
        # Baseline discharge
        base_discharge = self.river_discharge.copy()
        
        # Rainfall contribution to flow (simplified model)
        rainfall_total = sum(monthly_rainfall.values())
        rainfall_factor = rainfall_total / 2500.0  # Normalized to average annual rainfall
        
        # Upstream conditions (simplified, would be more complex in reality)
        upstream_factor = 1.0
        if upstream_flow:
            upstream_factor = upstream_flow.get('flow_factor', 1.0)
        
        # Calculate monthly discharge patterns
        # Different rivers have different seasonal patterns
        monthly_patterns = {
            'brahmaputra': {
                'january': 0.4, 'february': 0.3, 'march': 0.3, 'april': 0.4, 
                'may': 0.6, 'june': 1.2, 'july': 2.0, 'august': 1.8, 
                'september': 1.4, 'october': 0.8, 'november': 0.5, 'december': 0.4
            },
            'ganges': {
                'january': 0.5, 'february': 0.4, 'march': 0.3, 'april': 0.3, 
                'may': 0.4, 'june': 0.8, 'july': 1.6, 'august': 1.8, 
                'september': 1.7, 'october': 1.2, 'november': 0.7, 'december': 0.6
            },
            'meghna': {
                'january': 0.6, 'february': 0.5, 'march': 0.4, 'april': 0.5, 
                'may': 0.7, 'june': 1.3, 'july': 1.8, 'august': 1.7, 
                'september': 1.5, 'october': 1.0, 'november': 0.7, 'december': 0.6
            },
            'others': {
                'january': 0.4, 'february': 0.3, 'march': 0.2, 'april': 0.3, 
                'may': 0.5, 'june': 1.0, 'july': 1.5, 'august': 1.6, 
                'september': 1.3, 'october': 0.9, 'november': 0.6, 'december': 0.5
            }
        }
        
        # Apply factors to calculate actual discharge
        monthly_discharge = {}
        for river, base in base_discharge.items():
            monthly_discharge[river] = {}
            
            # Apply rainfall and upstream factors with river-specific sensitivity
            if river == 'brahmaputra':
                # Brahmaputra is more sensitive to upstream conditions (Himalayan snowmelt)
                river_factor = 0.6 * rainfall_factor + 0.8 * upstream_factor
            elif river == 'ganges':
                # Ganges has significant upstream management (dams in India)
                river_factor = 0.4 * rainfall_factor + 0.9 * upstream_factor
            elif river == 'meghna':
                # Meghna is more sensitive to local rainfall
                river_factor = 0.8 * rainfall_factor + 0.4 * upstream_factor
            else:
                # Other rivers depend primarily on local rainfall
                river_factor = 0.9 * rainfall_factor + 0.2 * upstream_factor
            
            # Calculate monthly discharge with patterns and variability
            pattern = monthly_patterns[river]
            for month, pattern_factor in pattern.items():
                # Add some random variation to each month's flow
                month_variability = np.random.normal(1.0, 0.1)
                monthly_discharge[river][month] = base * pattern_factor * river_factor * month_variability
                
        return monthly_discharge
    
    def calculate_flooding(self, monthly_discharge, sea_level_rise, infrastructure=None):
        """
        Calculate flooding parameters based on river discharge and other factors.
        
        Args:
            monthly_discharge (dict): Monthly river discharge by major river
            sea_level_rise (float): Sea level rise in meters
            infrastructure (dict): Infrastructure parameters including flood protection
            
        Returns:
            dict: Flooding parameters
        """
        # Extract peak monsoon discharge (typically July-August)
        peak_months = ['july', 'august', 'september']
        peak_discharge = {}
        
        for river, monthly in monthly_discharge.items():
            peak_discharge[river] = max([monthly.get(month, 0) for month in peak_months])
        
        # Calculate total peak discharge
        total_peak_discharge = sum(peak_discharge.values())
        
        # Discharge threshold for significant flooding
        base_threshold = 70000.0  # m³/s - approximate combined flow threshold
        
        # Calculate flood extent based on discharge excess and sea level
        discharge_ratio = total_peak_discharge / base_threshold
        
        # Sea level effect (higher sea levels worsen flooding)
        sea_level_effect = 0.2 * (sea_level_rise / 0.3)  # Normalized to 0.3m reference
        
        # Infrastructure effect (flood protection reduces flooding)
        infrastructure_effect = 0.0
        if infrastructure:
            # Infrastructure system has water-related attributes defined in __init__
            # Based on the memory about the infrastructure system, it should have irrigation_coverage
            flood_protection = infrastructure.irrigation_coverage if hasattr(infrastructure, 'irrigation_coverage') else self.flood_protection
            embankment_condition = 0.6  # Default value
            sluice_gate_function = 0.7  # Default value
            
            # Combined infrastructure effect
            infrastructure_effect = -0.3 * flood_protection * embankment_condition * sluice_gate_function
        
        # Calculate flood extent (% of land area)
        base_flood_extent = max(0, 0.1 * (discharge_ratio - 0.9))
        self.flood_extent = min(0.6, base_flood_extent + sea_level_effect + infrastructure_effect)
        
        # Calculate flood duration (days)
        base_duration = 20 + 40 * (discharge_ratio - 0.9)
        self.flood_duration = max(0, min(90, base_duration - 10 * self.flood_protection))
        
        # Calculate flood depth (meters)
        base_depth = 0.7 + 1.5 * (discharge_ratio - 0.9)
        self.flood_depth = max(0, min(3.0, base_depth - 0.5 * self.flood_protection))
        
        # Return flooding parameters
        flood_params = {
            'extent': self.flood_extent,
            'duration': self.flood_duration,
            'depth': self.flood_depth,
            'discharge_ratio': discharge_ratio,
            'sea_level_effect': sea_level_effect,
            'infrastructure_effect': infrastructure_effect
        }
        
        return flood_params
    
    def calculate_drought(self, monthly_rainfall, temperature_stress):
        """
        Calculate drought parameters based on rainfall deficits and temperature.
        
        Args:
            monthly_rainfall (dict): Monthly rainfall amounts
            temperature_stress (float): Temperature stress index
            
        Returns:
            dict: Drought parameters
        """
        # Calculate dry season rainfall (November to April)
        dry_season_months = ['november', 'december', 'january', 'february', 'march', 'april']
        dry_season_rainfall = sum([monthly_rainfall.get(month, 0) for month in dry_season_months])
        
        # Normal dry season rainfall
        normal_dry_season = 300.0  # mm
        
        # Calculate rainfall deficit
        rainfall_ratio = dry_season_rainfall / normal_dry_season
        
        # Temperature effect (higher temperatures worsen drought through evaporation)
        temperature_effect = 0.3 * temperature_stress
        
        # Calculate drought severity (0-1 scale)
        base_severity = max(0, 0.2 * (1.0 - rainfall_ratio))
        self.drought_severity = min(1.0, base_severity + temperature_effect)
        
        # Calculate drought duration (days)
        base_duration = 20 + 60 * (1.0 - rainfall_ratio)
        self.drought_duration = max(0, min(120, base_duration + 20 * temperature_stress))
        
        # Return drought parameters
        drought_params = {
            'severity': self.drought_severity,
            'duration': self.drought_duration,
            'rainfall_ratio': rainfall_ratio,
            'temperature_effect': temperature_effect
        }
        
        return drought_params
    
    def update_groundwater(self, monthly_rainfall, demographic_system=None, infrastructure_system=None):
        """
        Update groundwater parameters based on rainfall, extraction, and salinity.
        
        Args:
            monthly_rainfall (dict): Monthly rainfall amounts
            demographic_system (obj): Demographic system for population pressure
            infrastructure_system (obj): Infrastructure system for water infrastructure
            
        Returns:
            dict: Groundwater parameters
        """
        # Calculate total annual rainfall
        annual_rainfall = sum(monthly_rainfall.values())
        
        # Calculate recharge factor based on rainfall
        recharge_factor = annual_rainfall / 2500.0  # Normalized to average annual rainfall
        
        # Population pressure on groundwater extraction
        extraction_factor = 1.0
        if demographic_system:
            # Population growth increases extraction
            population = demographic_system.total_population  # Access attribute directly
            urbanization = demographic_system.urbanization_rate  # Access attribute directly
            
            # Combined demographic effect
            extraction_factor = population / 165000000 * (1 + 0.5 * urbanization)
        
        # Infrastructure effect on groundwater
        infrastructure_effect = 0.0
        if infrastructure_system:
            # Water supply infrastructure reduces groundwater dependency
            water_supply = infrastructure_system.water_supply_coverage  # Access attribute directly
            water_treatment = infrastructure_system.water_treatment_capacity  # Access attribute directly
            
            # Surface water irrigation reduces groundwater use for agriculture
            surface_irrigation = infrastructure_system.irrigation_coverage  # Using irrigation_coverage as proxy
            
            # Combined infrastructure effect
            infrastructure_effect = -0.2 * water_supply - 0.1 * water_treatment - 0.2 * surface_irrigation
        
        # Update groundwater extraction
        self.groundwater_extraction = max(0.5, extraction_factor * (1 + infrastructure_effect))
        
        # Update groundwater recharge
        self.groundwater_recharge = max(0.5, recharge_factor)
        
        # Calculate net groundwater balance
        net_balance = self.groundwater_recharge - self.groundwater_extraction
        
        # Update groundwater level
        level_change_rate = 0.05 * net_balance
        self.groundwater_level = max(0.3, self.groundwater_level * (1 + level_change_rate))
        
        # Update salinity intrusion
        # Lower groundwater levels and sea level rise increase salinity
        if hasattr(self, 'sea_level_rise'):
            sea_level_effect = 0.1 * (self.sea_level_rise / 0.3)  # Normalized to 0.3m reference
        else:
            sea_level_effect = 0.0
            
        salinity_change = 0.02 - 0.05 * (self.groundwater_level - 1.0) + sea_level_effect
        self.salinity_intrusion = max(0.0, min(1.0, self.salinity_intrusion + salinity_change))
        
        # Return groundwater parameters
        groundwater_params = {
            'level': self.groundwater_level,
            'recharge': self.groundwater_recharge,
            'extraction': self.groundwater_extraction,
            'net_balance': net_balance,
            'salinity_intrusion': self.salinity_intrusion
        }
        
        return groundwater_params
    
    def update_water_quality(self, demographic_system=None, infrastructure_system=None):
        """
        Update water quality parameters based on population, infrastructure, and environmental factors.
        
        Args:
            demographic_system (obj): Demographic system for population pressure
            infrastructure_system (obj): Infrastructure system for water treatment
            
        Returns:
            dict: Water quality parameters
        """
        results = {}
        
        # Natural degradation/improvement factor
        natural_factor = np.random.normal(0.0, 0.02)  # Small random fluctuation
        
        # Population pressure on water quality
        demographic_effect = 0.0
        if demographic_system:
            # Calculate a population density proxy using total population
            population_density_proxy = demographic_system.total_population / 147570  # Area of Bangladesh in sq km
            urbanization = demographic_system.urbanization_rate  # Access attribute directly
            
            # Negative effect of population and urbanization on water quality
            demographic_effect = -0.03 * (population_density_proxy / 1000) * (1 + urbanization)
        
        # Infrastructure effect on water quality
        infrastructure_effect = 0.0
        if infrastructure_system:
            # Access directly available attributes
            water_treatment = infrastructure_system.water_treatment_capacity  # Access attribute directly
            sanitation = infrastructure_system.sanitation_coverage  # Access attribute directly
            
            # Use a default value for waste_management since it might not be available
            waste_management = 0.4  # Default value
            if hasattr(infrastructure_system, 'waste_management'):
                waste_management = infrastructure_system.waste_management
            
            # Positive effect of infrastructure on water quality
            infrastructure_effect = 0.05 * water_treatment + 0.03 * sanitation + 0.02 * waste_management
        
        # Update water quality based on all factors
        water_quality_change = natural_factor + demographic_effect + infrastructure_effect
        self.water_quality = max(0.1, min(1.0, self.water_quality + water_quality_change))
        
        # Update salinity intrusion (affected by sea level and extraction)
        # Simplified model for now
        self.salinity_intrusion = self.salinity_intrusion + 0.005 * (1 + self.groundwater_extraction - self.groundwater_recharge)
        self.salinity_intrusion = max(0.0, min(1.0, self.salinity_intrusion))
        
        # Update arsenic contamination (slower changing)
        arsenic_change = 0.001 * (self.groundwater_extraction - 1.0)
        self.arsenic_contamination = max(0.0, min(1.0, self.arsenic_contamination + arsenic_change))
        
        # Compile results
        results = {
            'water_quality': self.water_quality,
            'salinity_intrusion': self.salinity_intrusion,
            'arsenic_contamination': self.arsenic_contamination
        }
        
        return results
    
    def update_water_management(self, infrastructure_system=None, economic_system=None):
        """
        Update water management parameters based on infrastructure investments and economic capacity.
        
        Args:
            infrastructure_system (obj): Infrastructure system for water infrastructure
            economic_system (obj): Economic system for investments
            
        Returns:
            dict: Water management parameters
        """
        # Infrastructure investments
        infrastructure_effect = 0.0
        if infrastructure_system:
            # Use irrigation_coverage which we know exists from previous inspection
            water_investment = 0.01  # Default value
            irrigation_investment = 0.01  # Default value
            flood_protection_investment = 0.01  # Default value
            
            # Use attributes if available, otherwise use defaults
            if hasattr(infrastructure_system, 'annual_infrastructure_investment'):
                # Use a fraction of the overall infrastructure investment for water
                investment_allocation = 0.15  # Assume 15% goes to water
                water_investment = infrastructure_system.annual_infrastructure_investment * investment_allocation
                irrigation_investment = water_investment * 0.5  # Half of water investment to irrigation
                flood_protection_investment = water_investment * 0.3  # 30% to flood protection
            
            # Combined infrastructure effect
            infrastructure_effect = 0.03 * water_investment + 0.02 * irrigation_investment + 0.04 * flood_protection_investment
        
        # Economic capacity
        economic_effect = 0.0
        if economic_system:
            # Access GDP growth rate attribute if available
            gdp_growth = 0.04  # Default value of 4% GDP growth
            if hasattr(economic_system, 'gdp_growth_rate'):
                gdp_growth = economic_system.gdp_growth_rate
            
            # Economic growth enables more investments
            economic_effect = 0.01 * gdp_growth
        
        # Update water management parameters
        self.irrigation_coverage = min(0.9, self.irrigation_coverage + 0.01 * infrastructure_effect)
        self.flood_protection = min(0.8, self.flood_protection + 0.02 * infrastructure_effect)
        self.water_treatment = min(0.7, self.water_treatment + 0.015 * infrastructure_effect)
        
        # Return management parameters
        management_params = {
            'irrigation_coverage': self.irrigation_coverage,
            'flood_protection': self.flood_protection,
            'water_treatment': self.water_treatment,
            'infrastructure_effect': infrastructure_effect,
            'economic_effect': economic_effect
        }
        
        return management_params
    
    def step(self, year, climate_results, demographic_system=None, infrastructure_system=None):
        """
        Advance the water system by one time step.
        
        Args:
            year (int): Current simulation year
            climate_results (dict): Climate system results
            demographic_system (obj): Demographic system for population pressure
            infrastructure_system (obj): Infrastructure system for water infrastructure
            
        Returns:
            dict: Water system results
        """
        print(f"Water system step: year {year}")
        
        # Extract climate results
        monthly_rainfall = climate_results.get('monthly_rainfall', {})
        temperature_stress = climate_results.get('temperature_stress', 0.0)
        sea_level_rise = 0.3  # Default
        if hasattr(climate_results, 'global_climate'):
            sea_level_rise = climate_results.get('global_climate', {}).get('sea_level_rise', 0.3)
        
        # Calculate river discharge
        monthly_discharge = self.calculate_river_discharge(monthly_rainfall)
        
        # Calculate flooding parameters
        flood_params = self.calculate_flooding(monthly_discharge, sea_level_rise, infrastructure_system)
        
        # Calculate drought parameters
        drought_params = self.calculate_drought(monthly_rainfall, temperature_stress)
        
        # Update groundwater
        groundwater_params = self.update_groundwater(monthly_rainfall, demographic_system, infrastructure_system)
        
        # Update water quality
        quality_params = self.update_water_quality(demographic_system, infrastructure_system)
        
        # Update water management
        management_params = self.update_water_management(infrastructure_system, None)  # Economic system will be connected later
        
        # Compile results
        results = {
            'year': year,
            'monthly_discharge': monthly_discharge,
            'flooding': flood_params,
            'drought': drought_params,
            'groundwater': groundwater_params,
            'water_quality': quality_params,
            'water_management': management_params,
            'irrigation_coverage': self.irrigation_coverage,
            'flood_protection': self.flood_protection,
            'water_treatment': self.water_treatment,
            'salinity_intrusion': self.salinity_intrusion
        }
        
        return results
