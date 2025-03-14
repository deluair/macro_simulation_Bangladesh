#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Land system model for Bangladesh simulation.
This module handles land use, land cover change, soil quality, and erosion dynamics.
"""

import numpy as np
import pandas as pd


class LandSystem:
    """
    Land system model representing Bangladesh's land use, soil, and erosion dynamics.
    """
    
    def __init__(self, config, environmental_data):
        """
        Initialize the land system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters step by step,for the land system
            environmental_data (dict): Initial environmental data
        """
        self.config = config
        
        # Land cover parameters (fractions of total land area)
        self.land_cover = environmental_data.get('land_cover', {
            'cropland': 0.65,            # 65% agricultural land
            'forest': 0.11,              # 11% forest
            'urban': 0.08,               # 8% urban areas
            'water_bodies': 0.04,        # 4% permanent water bodies
            'wetlands': 0.03,            # 3% wetlands
            'grassland': 0.03,           # 3% grassland
            'mangroves': 0.01,           # 1% mangroves (Sundarbans)
            'barren': 0.02,              # 2% barren land
            'other': 0.03                # 3% other land types
        })
        
        # Land use parameters (how land is used)
        self.land_use = environmental_data.get('land_use', {
            'rice_cultivation': 0.45,    # 45% of land area (70% of cropland)
            'other_crops': 0.20,         # 20% of land area (30% of cropland)
            'residential': 0.05,         # 5% residential areas
            'industrial': 0.01,          # 1% industrial areas
            'commercial': 0.02,          # 2% commercial areas
            'conservation': 0.04,        # 4% conservation areas
            'aquaculture': 0.03,         # 3% aquaculture
            'fallow': 0.05,              # 5% fallow land
            'other_uses': 0.15           # 15% other uses
        })
        
        # Cropland parameters
        self.cropping_intensity = environmental_data.get('cropping_intensity', 1.9)  # Crops per year (>1 means multiple crops)
        self.irrigation_coverage = environmental_data.get('irrigation_coverage', 0.55)  # Fraction of cropland irrigated
        
        # Soil parameters
        self.soil_quality = environmental_data.get('soil_quality', {
            'fertility': 0.6,            # 0-1 scale
            'organic_matter': 0.5,       # 0-1 scale
            'erosion_vulnerability': 0.7,  # 0-1 scale (higher is more vulnerable)
            'salinity': 0.3,             # 0-1 scale (higher is more saline)
            'acidity': 0.4               # 0-1 scale (higher is more acidic)
        })
        
        # Degradation parameters
        self.land_degradation = environmental_data.get('land_degradation', {
            'erosion_rate': 0.015,       # Fraction of topsoil lost per year
            'salinization_rate': 0.01,   # Increase in saline area fraction per year
            'nutrient_depletion': 0.02,  # Fraction of fertility lost per year without inputs
            'waterlogging': 0.1          # Fraction of cropland affected by waterlogging
        })
        
        # Deforestation and land transition rates (annual %)
        self.transition_rates = environmental_data.get('transition_rates', {
            'deforestation': 0.005,      # 0.5% of forest lost per year
            'urbanization': 0.01,        # 1% increase in urban area per year
            'agricultural_expansion': 0.002,  # 0.2% increase in cropland per year
            'wetland_loss': 0.01,        # 1% of wetlands lost per year
            'mangrove_loss': 0.005       # 0.5% of mangroves lost per year
        })
        
        # Land productivity
        self.land_productivity = environmental_data.get('land_productivity', {
            'cropland': 1.0,             # Normalized to 1.0 (baseline)
            'forest': 1.0,
            'grassland': 1.0,
            'wetlands': 1.0,
            'mangroves': 1.0
        })
        
        print("Land system initialized")
    
    def update_land_cover(self, economic_system=None, demographic_system=None):
        """
        Update land cover fractions based on economic and demographic pressures.
        
        Args:
            economic_system: Economic system for economic pressures
            demographic_system: Demographic system for population pressures
            
        Returns:
            dict: Updated land cover fractions
        """
        # Base land cover change rates
        base_change = {
            'cropland': -0.001,          # Slight decline due to urban expansion
            'forest': -0.005,            # Deforestation
            'urban': 0.005,              # Urban expansion
            'water_bodies': 0.0,         # Relatively stable
            'wetlands': -0.005,          # Wetland loss due to development
            'grassland': -0.001,         # Conversion to other uses
            'mangroves': -0.003,         # Mangrove loss
            'barren': 0.001,             # Slight increase due to degradation
            'other': 0.0
        }
        
        # Economic pressure (GDP growth increases development pressure)
        economic_factor = 1.0
        if economic_system:
            gdp_growth = getattr(economic_system, 'gdp_growth_rate', 0.06)
            economic_factor = 1.0 + (gdp_growth - 0.05) * 3.0  # Scale factor
        
        # Demographic pressure (population growth and urbanization)
        demographic_factor = 1.0
        urbanization_rate = 0.03
        if demographic_system:
            population_growth = demographic_system.population_growth_rate  # Access attribute directly
            urbanization_rate = demographic_system.urbanization_rate  # Access attribute directly
            
            demographic_factor = 1.0 + (population_growth - 0.01) * 5.0  # Scale factor
        
        # Adjust change rates by economic and demographic factors
        adjusted_change = {}
        for land_type, change in base_change.items():
            if land_type == 'urban':
                # Urban expansion driven by both economic growth and urbanization
                adjusted_change[land_type] = change * economic_factor * demographic_factor * (1.0 + 5.0 * urbanization_rate)
            elif land_type == 'cropland':
                # Cropland affected by economic growth and population
                adjusted_change[land_type] = change * demographic_factor * 0.5
            elif land_type in ['forest', 'wetlands', 'mangroves']:
                # Natural areas decline with economic and demographic pressure
                adjusted_change[land_type] = change * economic_factor * demographic_factor
            else:
                adjusted_change[land_type] = change
        
        # Apply changes
        old_land_cover = self.land_cover.copy()
        for land_type, change in adjusted_change.items():
            self.land_cover[land_type] += change
            
            # Ensure not negative
            self.land_cover[land_type] = max(0.0, self.land_cover[land_type])
        
        # Normalize to ensure sum equals 1
        total = sum(self.land_cover.values())
        for land_type in self.land_cover:
            self.land_cover[land_type] /= total
        
        # Calculate changes
        changes = {land_type: self.land_cover[land_type] - old_land_cover[land_type] 
                  for land_type in self.land_cover}
        
        return {
            'land_cover': self.land_cover,
            'changes': changes
        }
    
    def update_land_use(self, agricultural_system=None, infrastructure_system=None):
        """
        Update land use parameters based on agricultural and infrastructure developments.
        
        Args:
            agricultural_system: Agricultural system for farming practices
            infrastructure_system: Infrastructure system for development
            
        Returns:
            dict: Updated land use parameters
        """
        # Base land use change rates
        base_change = {
            'rice_cultivation': -0.002,  # Gradual diversification away from rice
            'other_crops': 0.003,        # Increase in other crops
            'residential': 0.002,        # More residential development
            'industrial': 0.001,         # Slow industrial growth
            'commercial': 0.001,         # Commercial expansion
            'conservation': 0.0,         # Conservation relatively stable
            'aquaculture': 0.001,        # Growing aquaculture
            'fallow': -0.001,            # Declining fallow land
            'other_uses': -0.005
        }
        
        # Agricultural factors
        ag_factor = 1.0
        if agricultural_system:
            # Crop diversification and intensification trends
            # Use default values since we don't know the available attributes
            crop_diversification = 0.05  # Default value
            intensification = 0.03  # Default value
            
            # Try to use attributes directly if available
            if hasattr(agricultural_system, 'crop_diversification'):
                crop_diversification = agricultural_system.crop_diversification
            if hasattr(agricultural_system, 'intensification'):
                intensification = agricultural_system.intensification
            
            ag_factor = 1.0 + 2.0 * crop_diversification
            
            # Adjust specific ag land uses
            base_change['rice_cultivation'] *= (1.0 - crop_diversification)
            base_change['other_crops'] *= (1.0 + crop_diversification)
            base_change['fallow'] *= (1.0 - intensification)
        
        # Infrastructure development factors
        infra_factor = 1.0
        if infrastructure_system:
            # Urban and industrial development
            urban_investment = 0.01  # Default value
            industrial_investment = 0.01  # Default value
            
            # Use appropriate attributes if available
            if hasattr(infrastructure_system, 'annual_infrastructure_investment'):
                # Estimate urban and industrial investment from overall infrastructure investment
                urban_investment = infrastructure_system.annual_infrastructure_investment * 0.1  # Assume 10% goes to urban
                industrial_investment = infrastructure_system.annual_infrastructure_investment * 0.05  # Assume 5% goes to industrial
            
            infra_factor = 1.0 + 2.0 * urban_investment
            
            # Adjust specific infrastructure land uses
            base_change['residential'] *= (1.0 + 2.0 * urban_investment)
            base_change['industrial'] *= (1.0 + 3.0 * industrial_investment)
            base_change['commercial'] *= (1.0 + 2.0 * urban_investment)
        
        # Apply changes
        old_land_use = self.land_use.copy()
        
        # First ensure all keys in base_change exist in self.land_use
        for use_type in base_change.keys():
            if use_type not in self.land_use:
                # Initialize missing keys with reasonable defaults
                if use_type == 'rice_cultivation':
                    self.land_use[use_type] = 0.45  # 45% as mentioned in __init__
                    old_land_use[use_type] = 0.45  # Update old_land_use too
                elif use_type == 'other_crops':
                    self.land_use[use_type] = 0.20  # 20% as mentioned in __init__
                    old_land_use[use_type] = 0.20  # Update old_land_use too
                else:
                    self.land_use[use_type] = 0.01  # Small default value
                    old_land_use[use_type] = 0.01  # Update old_land_use too
        
        # Then apply changes
        for use_type, change in base_change.items():
            self.land_use[use_type] += change * ag_factor * infra_factor
            
            # Ensure not negative
            self.land_use[use_type] = max(0.0, self.land_use[use_type])
        
        # Normalize to ensure sum equals 1
        total = sum(self.land_use.values())
        for use_type in self.land_use:
            self.land_use[use_type] /= total
        
        # Update cropping intensity (increasing over time)
        self.cropping_intensity = min(3.0, self.cropping_intensity * 1.005)
        
        # Update irrigation coverage
        if infrastructure_system:
            irrigation_investment = infrastructure_system.annual_infrastructure_investment * 0.05  # Assume 5% goes to irrigation
            self.irrigation_coverage = min(0.8, self.irrigation_coverage * (1.0 + 0.1 * irrigation_investment))
        else:
            self.irrigation_coverage = min(0.8, self.irrigation_coverage * 1.005)
        
        # Calculate changes
        changes = {use_type: self.land_use[use_type] - old_land_use[use_type] 
                  for use_type in self.land_use}
        
        return {
            'land_use': self.land_use,
            'cropping_intensity': self.cropping_intensity,
            'irrigation_coverage': self.irrigation_coverage,
            'changes': changes
        }
    
    def update_soil_quality(self, climate_results, water_results, agricultural_system=None):
        """
        Update soil quality parameters based on climate, water, and agricultural practices.
        
        Args:
            climate_results (dict): Climate system results
            water_results (dict): Water system results
            agricultural_system: Agricultural system for farming practices
            
        Returns:
            dict: Updated soil quality parameters
        """
        # Extract relevant climate and water parameters
        temperature_stress = climate_results.get('temperature_stress', 0.0)
        rainfall_adequacy = climate_results.get('rainfall_adequacy', 1.0)
        flood_extent = water_results.get('flooding', {}).get('extent', 0.0)
        drought_severity = water_results.get('drought', {}).get('severity', 0.0)
        salinity_intrusion = water_results.get('groundwater', {}).get('salinity_intrusion', 0.0)
        
        # Base soil quality change rates
        base_change = {
            'fertility': -0.005,         # Natural fertility decline
            'organic_matter': -0.005,    # Organic matter loss
            'erosion_vulnerability': 0.005,  # Increasing vulnerability
            'salinity': 0.003,           # Increasing salinity
            'acidity': 0.002             # Increasing acidity
        }
        
        # Climate effects
        climate_effects = {
            'fertility': -0.01 * temperature_stress - 0.02 * drought_severity,
            'organic_matter': -0.01 * temperature_stress,
            'erosion_vulnerability': 0.02 * (1.0 - rainfall_adequacy) + 0.01 * flood_extent,
            'salinity': 0.03 * salinity_intrusion,
            'acidity': 0.01 * (rainfall_adequacy - 1.0) ** 2  # Too much or too little rain can affect acidity
        }
        
        # Agricultural management effects
        ag_effects = {
            'fertility': 0.0,
            'organic_matter': 0.0,
            'erosion_vulnerability': 0.0,
            'salinity': 0.0,
            'acidity': 0.0
        }
        
        if agricultural_system:
            # Agricultural practices can improve or degrade soil
            organic_farming = agricultural_system.get_organic_farming_rate()
            conservation_tillage = agricultural_system.get_conservation_tillage_rate()
            chemical_intensity = agricultural_system.get_chemical_intensity()
            
            # Positive practices improve soil quality
            ag_effects['fertility'] = 0.02 * organic_farming - 0.01 * chemical_intensity
            ag_effects['organic_matter'] = 0.03 * organic_farming + 0.02 * conservation_tillage
            ag_effects['erosion_vulnerability'] = -0.04 * conservation_tillage
            ag_effects['salinity'] = -0.01 * organic_farming + 0.02 * chemical_intensity
            ag_effects['acidity'] = -0.01 * organic_farming + 0.02 * chemical_intensity
        
        # Apply changes
        for parameter, base in base_change.items():
            climate_effect = climate_effects.get(parameter, 0.0)
            ag_effect = ag_effects.get(parameter, 0.0)
            
            # Calculate total change
            total_change = base + climate_effect + ag_effect
            
            # Apply change
            self.soil_quality[parameter] += total_change
            
            # Ensure within bounds [0,1]
            self.soil_quality[parameter] = max(0.0, min(1.0, self.soil_quality[parameter]))
        
        # Update land degradation rates based on soil quality
        self.land_degradation['erosion_rate'] = 0.01 + 0.02 * self.soil_quality['erosion_vulnerability']
        self.land_degradation['salinization_rate'] = 0.005 + 0.02 * self.soil_quality['salinity']
        self.land_degradation['nutrient_depletion'] = 0.01 + 0.03 * (1.0 - self.soil_quality['fertility'])
        self.land_degradation['waterlogging'] = 0.05 + 0.1 * flood_extent
        
        return {
            'soil_quality': self.soil_quality,
            'land_degradation': self.land_degradation,
            'climate_effects': climate_effects,
            'agricultural_effects': ag_effects
        }
    
    def update_land_productivity(self, climate_results, water_results, agricultural_system=None):
        """
        Update land productivity based on climate, water, soil, and agricultural practices.
        
        Args:
            climate_results (dict): Climate system results
            water_results (dict): Water system results
            agricultural_system: Agricultural system for farming practices
            
        Returns:
            dict: Updated land productivity
        """
        # Extract relevant parameters
        temperature_stress = climate_results.get('temperature_stress', 0.0)
        rainfall_adequacy = climate_results.get('rainfall_adequacy', 1.0)
        flood_extent = water_results.get('flooding', {}).get('extent', 0.0)
        drought_severity = water_results.get('drought', {}).get('severity', 0.0)
        
        # Base productivity change rates (gradual improvements with technology)
        base_change = {
            'cropland': 0.01,            # 1% annual productivity growth
            'forest': 0.005,             # 0.5% forest productivity growth
            'grassland': 0.005,          # 0.5% grassland productivity growth
            'wetlands': 0.0,             # Wetlands stable
            'mangroves': 0.005           # 0.5% mangrove productivity growth
        }
        
        # Climate effects on productivity
        climate_effects = {
            'cropland': -0.2 * temperature_stress - 0.3 * drought_severity - 0.2 * flood_extent + 0.2 * (rainfall_adequacy - 1.0),
            'forest': -0.1 * temperature_stress - 0.2 * drought_severity,
            'grassland': -0.15 * temperature_stress - 0.25 * drought_severity,
            'wetlands': -0.05 * drought_severity + 0.1 * flood_extent,
            'mangroves': -0.1 * temperature_stress + 0.05 * flood_extent
        }
        
        # Soil quality effects (mainly for cropland)
        soil_effects = {
            'cropland': 0.2 * self.soil_quality['fertility'] + 0.1 * self.soil_quality['organic_matter'] - 0.1 * self.soil_quality['salinity'],
            'forest': 0.1 * self.soil_quality['fertility'],
            'grassland': 0.1 * self.soil_quality['fertility'] - 0.05 * self.soil_quality['salinity'],
            'wetlands': 0.0,
            'mangroves': -0.1 * self.soil_quality['salinity']  # Mangroves adapted to some salinity
        }
        
        # Agricultural management effects
        ag_effects = {
            'cropland': 0.0,
            'forest': 0.0,
            'grassland': 0.0,
            'wetlands': 0.0,
            'mangroves': 0.0
        }
        
        if agricultural_system:
            technology_level = agricultural_system.get_technology_level()
            irrigation_efficiency = agricultural_system.get_irrigation_efficiency()
            sustainable_practices = agricultural_system.get_sustainable_practices()
            
            # Agricultural practices mainly affect cropland
            ag_effects['cropland'] = 0.1 * technology_level + 0.05 * irrigation_efficiency + 0.05 * sustainable_practices
            
            # Forestry management affects forests
            forestry_management = agricultural_system.get_forestry_management()
            ag_effects['forest'] = 0.05 * forestry_management
        
        # Apply changes
        for land_type, base in base_change.items():
            climate_effect = climate_effects.get(land_type, 0.0)
            soil_effect = soil_effects.get(land_type, 0.0)
            ag_effect = ag_effects.get(land_type, 0.0)
            
            # Calculate total change
            total_change = base + climate_effect + soil_effect + ag_effect
            
            # Apply change
            self.land_productivity[land_type] *= (1.0 + total_change)
            
            # Ensure reasonable bounds
            self.land_productivity[land_type] = max(0.5, min(2.0, self.land_productivity[land_type]))
        
        return {
            'land_productivity': self.land_productivity,
            'climate_effects': climate_effects,
            'soil_effects': soil_effects,
            'agricultural_effects': ag_effects
        }
    
    def step(self, year, climate_results, water_results, demographic_system=None, economic_system=None):
        """
        Advance the land system by one time step.
        
        Args:
            year (int): Current simulation year
            climate_results (dict): Climate system results
            water_results (dict): Water system results
            demographic_system: Demographic system for population pressure
            economic_system: Economic system for economic pressure
            
        Returns:
            dict: Land system results
        """
        print(f"Land system step: year {year}")
        
        # Update land cover
        land_cover_results = self.update_land_cover(
            economic_system=economic_system,
            demographic_system=demographic_system
        )
        
        # Update land use
        land_use_results = self.update_land_use(
            agricultural_system=None,  # Will be connected later
            infrastructure_system=None  # Will be connected later
        )
        
        # Update soil quality
        soil_results = self.update_soil_quality(
            climate_results=climate_results,
            water_results=water_results,
            agricultural_system=None  # Will be connected later
        )
        
        # Update land productivity
        productivity_results = self.update_land_productivity(
            climate_results=climate_results,
            water_results=water_results,
            agricultural_system=None  # Will be connected later
        )
        
        # Compile results
        results = {
            'year': year,
            'land_cover': land_cover_results,
            'land_use': land_use_results,
            'soil_quality': soil_results,
            'land_productivity': productivity_results,
            'transition_rates': self.transition_rates
        }
        
        return results
