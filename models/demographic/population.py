#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demographic system model for Bangladesh simulation.
This module implements population dynamics, migration patterns, education,
and skill distribution affecting productivity and innovation.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
import geopandas as gpd


class DemographicSystem:
    """
    Demographic system model representing Bangladesh's population dynamics and characteristics.
    """
    
    def __init__(self, config, data_loader):
        """
        Initialize the demographic system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the demographic system
            data_loader (DataLoader): Data loading utility for demographic data
        """
        self.config = config
        self.data_loader = data_loader
        
        # Load initial demographic data
        self.demographic_data = data_loader.load_demographic_data()
        
        # Set up time-related variables
        self.current_year = config.get('start_year', 2023)
        self.time_step = config.get('time_step', 1.0)
        self.base_year = config.get('base_year', 2000)
        
        # Population parameters
        self.total_population = self.demographic_data.get('total_population', 169_000_000)  # 2023 estimate
        self.population_growth_rate = self.demographic_data.get('population_growth_rate', 0.01)  # Annual growth rate
        self.urbanization_rate = self.demographic_data.get('urbanization_rate', 0.37)  # Urban population percentage
        self.urban_growth_rate = self.demographic_data.get('urban_growth_rate', 0.03)  # Annual urban growth
        
        # Age structure 
        self.age_groups = ['0-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
        self.age_distribution = self.demographic_data.get('age_distribution', {
            '0-14': 0.27,  # 27% of population under 14
            '15-24': 0.19,
            '25-34': 0.18,
            '35-44': 0.14,
            '45-54': 0.11,
            '55-64': 0.07,
            '65+': 0.04
        })
        
        # Gender distribution
        self.gender_ratio = self.demographic_data.get('gender_ratio', 0.98)  # Males per female
        
        # Education parameters
        self.education_levels = ['no_education', 'primary', 'secondary', 'tertiary']
        self.education_distribution = self.demographic_data.get('education_distribution', {
            'no_education': 0.25,
            'primary': 0.30,
            'secondary': 0.35,
            'tertiary': 0.10
        })
        self.literacy_rate = self.demographic_data.get('literacy_rate', 0.74)  # Overall literacy rate
        
        # Skill parameters
        self.skill_levels = ['unskilled', 'semi_skilled', 'skilled', 'highly_skilled']
        self.skill_distribution = self.demographic_data.get('skill_distribution', {
            'unskilled': 0.30,
            'semi_skilled': 0.40,
            'skilled': 0.25,
            'highly_skilled': 0.05
        })
        
        # Regional distribution (divisions)
        self.regions = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh']
        self.region_population = self.demographic_data.get('region_population', {
            'Dhaka': 0.31,
            'Chittagong': 0.20,
            'Rajshahi': 0.12,
            'Khulna': 0.10,
            'Barisal': 0.06,
            'Sylhet': 0.07,
            'Rangpur': 0.08,
            'Mymensingh': 0.06
        })
        
        # Rural-urban distribution per region
        self.region_urbanization = self.demographic_data.get('region_urbanization', {
            'Dhaka': 0.60,  # 60% urban in Dhaka division
            'Chittagong': 0.45,
            'Rajshahi': 0.30,
            'Khulna': 0.35,
            'Barisal': 0.25,
            'Sylhet': 0.30,
            'Rangpur': 0.23,
            'Mymensingh': 0.20
        })
        
        # Migration parameters
        self.rural_urban_migration_rate = self.demographic_data.get('rural_urban_migration_rate', 0.02)  # Annual rate
        self.international_migration_rate = self.demographic_data.get('international_migration_rate', 0.005)  # Annual rate
        self.remittance_per_migrant = self.demographic_data.get('remittance_per_migrant', 1500)  # USD per year
        self.internal_displacement_rate = self.demographic_data.get('internal_displacement_rate', 0.001)  # Annual rate
        
        # Health parameters
        self.life_expectancy = self.demographic_data.get('life_expectancy', 72.5)  # Years
        self.infant_mortality = self.demographic_data.get('infant_mortality', 30.0)  # Per 1000 live births
        self.maternal_mortality = self.demographic_data.get('maternal_mortality', 173.0)  # Per 100,000 live births
        self.fertility_rate = self.demographic_data.get('fertility_rate', 2.0)  # Children per woman
        
        # Initialize spatial population grid if in config
        if config.get('use_spatial_grid', True):
            self._init_spatial_grid()
        
        print("Demographic system initialized")
    
    def _init_spatial_grid(self):
        """Initialize the spatial population grid for Bangladesh."""
        print("Initializing spatial population grid...")
        
        # Load Bangladesh GIS data
        try:
            self.bd_gdf = self.data_loader.load_gis_data('bangladesh_divisions')
        except:
            print("Warning: Could not load GIS data, using simplified grid")
            # Create simplified grid if GIS data not available
            self.grid_size = self.config.get('grid_size', (50, 50))
            self.population_grid = np.zeros(self.grid_size)
            
            # Fill grid with population distribution (higher in urban centers)
            # Dhaka area has highest density
            center_x, center_y = self.grid_size[0] // 2, self.grid_size[1] // 2
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    # Distance from Dhaka (center)
                    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    # Exponential decay of population density with distance
                    self.population_grid[x, y] = np.exp(-0.1 * dist_from_center)
            
            # Normalize grid to sum to total population
            self.population_grid = self.population_grid / self.population_grid.sum() * self.total_population
            return
            
        # If GIS data available, initialize population per division
        self.division_population = {}
        for region in self.regions:
            # Calculate population for this region
            region_pop = self.total_population * self.region_population[region]
            self.division_population[region] = region_pop
            
            # Apply urban/rural split within division
            urban_pop = region_pop * self.region_urbanization[region]
            rural_pop = region_pop - urban_pop
            
            # Store in division data
            self.bd_gdf.loc[self.bd_gdf['division'] == region, 'population'] = region_pop
            self.bd_gdf.loc[self.bd_gdf['division'] == region, 'urban_pop'] = urban_pop
            self.bd_gdf.loc[self.bd_gdf['division'] == region, 'rural_pop'] = rural_pop
            
        # Calculate population density per division
        self.bd_gdf['density'] = self.bd_gdf['population'] / self.bd_gdf['area_km2']
        
        print(f"Spatial grid initialized with {len(self.regions)} divisions")
    
    def update_population(self, year, environmental_system, economic_system):
        """
        Update population size and distribution based on births, deaths, and migration.
        
        Args:
            year (int): Current simulation year
            environmental_system (EnvironmentalSystem): Environmental system state
            economic_system (EconomicSystem): Economic system state
            
        Returns:
            dict: Updated population parameters
        """
        # Calculate time step for this update
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'total_population': self.total_population}
            
        # --- Fertility and birth calculation ---
        # Adjust fertility based on economic development and female education
        gdp_per_capita = economic_system.gdp_per_capita if hasattr(economic_system, 'gdp_per_capita') else 2500
        female_education = self.education_distribution['secondary'] + self.education_distribution['tertiary']
        
        # Fertility decreases with higher GDP and female education
        fertility_adjustment = -0.02 * np.log(gdp_per_capita / 2500) - 0.5 * (female_education - 0.45)
        adjusted_fertility = max(1.2, min(6.0, self.fertility_rate + fertility_adjustment))
        
        # --- Mortality calculation ---
        # Adjust mortality based on health indicators, economic factors, and disasters
        disaster_impact = environmental_system.calculate_overall_impact() if hasattr(environmental_system, 'calculate_overall_impact') else 0
        
        # Mortality increases with disasters and decreases with economic development
        mortality_adjustment = 0.05 * disaster_impact - 0.03 * np.log(gdp_per_capita / 2500)
        adjusted_mortality = max(0.005, min(0.03, 0.01 + mortality_adjustment))
        
        # Calculate natural growth
        natural_growth_rate = (adjusted_fertility / 2.1 - 1) * 0.01 - adjusted_mortality
        
        # --- Migration calculation ---
        # International outmigration
        economic_opportunity = max(0, min(1, (gdp_per_capita - 2000) / 10000))
        political_stability = 0.6  # Can be replaced with governance system value
        environmental_stress = disaster_impact
        
        # International migration affected by economic, environmental, and political factors
        international_migration_adjustment = (
            -0.2 * economic_opportunity  # Better economy means less emigration
            + 0.3 * environmental_stress  # More disasters means more emigration
            - 0.1 * political_stability   # More stability means less emigration
        )
        adjusted_intl_migration = max(0.001, min(0.01, 
            self.international_migration_rate + international_migration_adjustment))
        
        # Calculate net growth rate accounting for natural growth and international migration
        net_growth_rate = natural_growth_rate - adjusted_intl_migration
        
        # Update total population
        old_population = self.total_population
        self.total_population *= (1 + net_growth_rate) ** time_delta
        
        # --- Update urbanization ---
        # Adjust rural-urban migration based on economic and environmental factors
        economic_urban_premium = economic_system.urban_rural_wage_ratio if hasattr(economic_system, 'urban_rural_wage_ratio') else 1.5
        environmental_rural_stress = disaster_impact * 0.7  # Rural areas often more affected
        
        # Rural-urban migration affected by economic and environmental factors
        rural_urban_adjustment = (
            0.2 * (economic_urban_premium - 1.5)  # Higher urban wages drive migration
            + 0.3 * environmental_rural_stress     # Rural environmental stress drives migration
        )
        adjusted_rural_urban = max(0.005, min(0.04, 
            self.rural_urban_migration_rate + rural_urban_adjustment))
        
        # Update urbanization rate
        rural_pop_pct = 1 - self.urbanization_rate
        migration_to_urban = rural_pop_pct * adjusted_rural_urban * time_delta
        urban_growth = self.urbanization_rate * (net_growth_rate + 0.01) * time_delta  # Urban areas grow faster
        
        self.urbanization_rate = min(0.9, self.urbanization_rate + migration_to_urban + urban_growth)
        
        # Store current values for next iteration
        self.current_year = year
        self.population_growth_rate = net_growth_rate
        self.rural_urban_migration_rate = adjusted_rural_urban
        self.fertility_rate = adjusted_fertility
        
        # Return updated values
        return {
            'total_population': self.total_population,
            'population_growth': (self.total_population - old_population) / old_population,
            'urbanization_rate': self.urbanization_rate,
            'fertility_rate': self.fertility_rate,
            'international_migration_rate': adjusted_intl_migration,
            'rural_urban_migration_rate': adjusted_rural_urban
        }
    
    def update_education_skills(self, year, economic_system, governance_system):
        """
        Update education and skill levels based on economic development and governance.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            governance_system (GovernanceSystem): Governance system state
            
        Returns:
            dict: Updated education and skill distribution
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'education_distribution': self.education_distribution}
        
        # Education investment
        education_investment = economic_system.education_investment if hasattr(economic_system, 'education_investment') else 0.03
        education_effectiveness = governance_system.education_effectiveness if hasattr(governance_system, 'education_effectiveness') else 0.5
        
        # Apply education transitions based on investment and effectiveness
        education_transition_rate = 0.01 * education_investment * education_effectiveness * time_delta
        
        # Update education distribution with transitions to higher levels
        new_education = {}
        new_education['no_education'] = max(0.05, self.education_distribution['no_education'] - education_transition_rate)
        new_education['primary'] = max(0.15, self.education_distribution['primary'] + 
                                     0.7 * education_transition_rate - 0.3 * education_transition_rate)
        new_education['secondary'] = max(0.1, self.education_distribution['secondary'] + 
                                       0.3 * education_transition_rate - 0.2 * education_transition_rate)
        new_education['tertiary'] = min(0.4, self.education_distribution['tertiary'] + 0.2 * education_transition_rate)
        
        # Normalize to ensure sum is 1.0
        total = sum(new_education.values())
        for level in new_education:
            new_education[level] /= total
            
        self.education_distribution = new_education
        
        # Update skill distribution based on education and economic opportunities
        skill_demand = economic_system.skill_demand if hasattr(economic_system, 'skill_demand') else {
            'unskilled': 0.3,
            'semi_skilled': 0.4,
            'skilled': 0.25,
            'highly_skilled': 0.05
        }
        
        # Weighted transition to match economic demand and education
        skill_transition_rate = 0.02 * time_delta
        edu_skill_factor = (self.education_distribution['secondary'] + 
                          2 * self.education_distribution['tertiary'])
        
        # Update skill distribution with transitions toward economic demand
        new_skills = {}
        for skill in self.skill_levels:
            demand_gap = skill_demand[skill] - self.skill_distribution[skill]
            education_effect = 0 
            if skill == 'skilled':
                education_effect = edu_skill_factor * 0.1
            elif skill == 'highly_skilled':
                education_effect = edu_skill_factor * 0.2
                
            adjustment = skill_transition_rate * (demand_gap + education_effect)
            new_skills[skill] = max(0.01, min(0.6, self.skill_distribution[skill] + adjustment))
            
        # Normalize skill distribution
        total = sum(new_skills.values())
        for skill in new_skills:
            new_skills[skill] /= total
            
        self.skill_distribution = new_skills
        
        # Update literacy rate based on education
        self.literacy_rate = min(1.0, max(0.3, 
            0.1 + 0.5 * self.education_distribution['primary'] + 
            0.95 * self.education_distribution['secondary'] + 
            1.0 * self.education_distribution['tertiary']))
        
        # Return updated values
        return {
            'education_distribution': self.education_distribution,
            'skill_distribution': self.skill_distribution,
            'literacy_rate': self.literacy_rate
        }
    
    def update_spatial_distribution(self, year, environmental_system, economic_system, infrastructure_system):
        """
        Update spatial distribution of population across regions.
        
        Args:
            year (int): Current simulation year
            environmental_system (EnvironmentalSystem): Environmental system state
            economic_system (EconomicSystem): Economic system state
            infrastructure_system (InfrastructureSystem): Infrastructure system state
            
        Returns:
            dict: Updated regional population distribution
        """
        # Get regional disaster impacts if available
        regional_disasters = {}
        if hasattr(environmental_system, 'get_regional_impacts'):
            regional_disasters = environmental_system.get_regional_impacts()
        else:
            # Fallback to default values if method not available
            for region in self.regions:
                regional_disasters[region] = 0.1  # Default moderate impact
        
        # Get regional economic opportunities if available
        regional_economy = {}
        if hasattr(economic_system, 'get_regional_gdp'):
            regional_gdp = economic_system.get_regional_gdp()
            for region in self.regions:
                regional_economy[region] = regional_gdp.get(region, 1.0) / economic_system.gdp_per_capita
        else:
            # Fallback values - Dhaka and Chittagong have better opportunities
            for region in self.regions:
                if region in ['Dhaka', 'Chittagong']:
                    regional_economy[region] = 1.5
                else:
                    regional_economy[region] = 0.8
        
        # Get infrastructure quality if available
        regional_infrastructure = {}
        if hasattr(infrastructure_system, 'get_regional_infrastructure'):
            regional_infrastructure = infrastructure_system.get_regional_infrastructure()
        else:
            # Fallback to default values
            for region in self.regions:
                if region in ['Dhaka', 'Chittagong']:
                    regional_infrastructure[region] = 0.7  # Better infrastructure
                else:
                    regional_infrastructure[region] = 0.4  # Less developed infrastructure
        
        # Calculate migration factors for each region
        migration_factors = {}
        for region in self.regions:
            # Combine factors affecting migration:
            # - Higher economic opportunity attracts migrants
            # - Better infrastructure attracts migrants
            # - Environmental disasters repel migrants
            migration_factors[region] = (
                2.0 * regional_economy[region] +
                1.0 * regional_infrastructure[region] -
                3.0 * regional_disasters[region]
            )
        
        # Normalize migration factors to create a migration distribution
        total_factor = sum(migration_factors.values())
        if total_factor != 0:
            for region in self.regions:
                migration_factors[region] /= total_factor
        
        # Calculate new population distribution
        # 90% of current distribution + 10% based on migration factors
        new_region_population = {}
        for region in self.regions:
            current_pop = self.region_population[region]
            migration_effect = migration_factors[region] - current_pop
            new_region_population[region] = current_pop + 0.1 * migration_effect
        
        # Normalize to ensure sum is 1.0
        total = sum(new_region_population.values())
        for region in self.regions:
            new_region_population[region] /= total
        
        # Update regional populations
        self.region_population = new_region_population
        
        # Update spatial grid if available
        if hasattr(self, 'bd_gdf'):
            for region in self.regions:
                # Calculate population for this region
                region_pop = self.total_population * self.region_population[region]
                
                # Apply urban/rural split within division
                urban_pop = region_pop * self.region_urbanization[region]
                rural_pop = region_pop - urban_pop
                
                # Update division data
                self.bd_gdf.loc[self.bd_gdf['division'] == region, 'population'] = region_pop
                self.bd_gdf.loc[self.bd_gdf['division'] == region, 'urban_pop'] = urban_pop
                self.bd_gdf.loc[self.bd_gdf['division'] == region, 'rural_pop'] = rural_pop
            
            # Recalculate population density
            self.bd_gdf['density'] = self.bd_gdf['population'] / self.bd_gdf['area_km2']
        
        return {
            'region_population': self.region_population,
            'urbanization_by_region': self.region_urbanization
        }
    
    def calculate_labor_force(self):
        """
        Calculate labor force size and characteristics.
        
        Returns:
            dict: Labor force characteristics
        """
        # Labor force participation rates by age group and gender
        labor_participation = {
            '0-14': 0.05,  # Child labor still exists but at low rates
            '15-24': 0.65,
            '25-34': 0.85,
            '35-44': 0.90,
            '45-54': 0.85,
            '55-64': 0.60,
            '65+': 0.30
        }
        
        # Male vs female participation rate differences
        gender_participation_ratio = 0.65  # Female participation as fraction of male
        
        # Calculate total labor force
        labor_force = 0
        for age_group, part_rate in labor_participation.items():
            age_group_pop = self.total_population * self.age_distribution[age_group]
            # Account for gender differences in participation
            males = age_group_pop / (1 + 1/self.gender_ratio)
            females = age_group_pop - males
            
            male_workers = males * part_rate
            female_workers = females * part_rate * gender_participation_ratio
            
            labor_force += male_workers + female_workers
        
        # Labor force by skill level
        labor_by_skill = {}
        for skill in self.skill_levels:
            labor_by_skill[skill] = labor_force * self.skill_distribution[skill]
        
        # Urban vs rural labor
        urban_labor = labor_force * self.urbanization_rate
        rural_labor = labor_force - urban_labor
        
        # Calculate human capital index (0-1 scale)
        # Based on education, skills, health
        human_capital_index = (
            0.3 * self.education_distribution['tertiary'] +
            0.2 * self.education_distribution['secondary'] +
            0.3 * self.skill_distribution['highly_skilled'] +
            0.2 * self.skill_distribution['skilled'] +
            0.1 * ((self.life_expectancy - 50) / 40)  # Normalized life expectancy contribution
        )
        
        return {
            'total_labor_force': labor_force,
            'labor_force_participation_rate': labor_force / self.total_population,
            'urban_labor': urban_labor,
            'rural_labor': rural_labor,
            'labor_by_skill': labor_by_skill,
            'human_capital_index': human_capital_index
        }
    
    def calculate_migration_flows(self):
        """
        Calculate detailed migration flows and remittances.
        
        Returns:
            dict: Migration flows and remittances
        """
        # International migration destinations and rates
        destinations = {
            'middle_east': 0.45,  # 45% of migrants go to Middle East
            'southeast_asia': 0.15,
            'europe': 0.10,
            'north_america': 0.20,
            'other': 0.10
        }
        
        # Estimate international migrant stock
        intl_migrant_stock = self.total_population * 0.03  # About 3% of population abroad
        new_migrants = self.total_population * self.international_migration_rate
        
        # Calculate remittances by destination
        remittance_by_destination = {}
        total_remittance = 0
        
        remittance_factors = {
            'middle_east': 1.0,  # Baseline 
            'southeast_asia': 0.8,
            'europe': 2.0,
            'north_america': 2.5,
            'other': 1.2
        }
        
        for dest, rate in destinations.items():
            migrants_in_dest = intl_migrant_stock * rate
            remit_per_migrant = self.remittance_per_migrant * remittance_factors[dest]
            remittance = migrants_in_dest * remit_per_migrant
            
            remittance_by_destination[dest] = remittance
            total_remittance += remittance
        
        # Internal migration - rural to urban
        rural_pop = self.total_population * (1 - self.urbanization_rate)
        rural_urban_migrants = rural_pop * self.rural_urban_migration_rate
        
        # Region to region migration flows based on regional population distribution
        region_migration_flows = {}
        for origin in self.regions:
            region_migration_flows[origin] = {}
            origin_pop = self.total_population * self.region_population[origin]
            
            for destination in self.regions:
                if origin == destination:
                    region_migration_flows[origin][destination] = 0
                    continue
                
                # Higher migration to more populous regions
                destination_factor = self.region_population[destination] * 2
                
                # Dhaka has stronger pull
                if destination == 'Dhaka':
                    destination_factor *= 1.5
                
                # Calculate flow from origin to destination
                migration_rate = 0.002 * destination_factor  # 0.2% base internal migration rate adjusted by factors
                migrants = origin_pop * migration_rate
                
                region_migration_flows[origin][destination] = migrants
        
        return {
            'international_migrants': intl_migrant_stock,
            'new_international_migrants': new_migrants,
            'international_destinations': destinations,
            'total_remittances': total_remittance,
            'remittances_by_destination': remittance_by_destination,
            'rural_urban_migrants': rural_urban_migrants,
            'region_migration_flows': region_migration_flows
        }
    
    def step(self, year, environmental_system=None, economic_system=None, governance_system=None, infrastructure_system=None):
        """
        Advance the demographic system by one time step.
        
        Args:
            year (int): Current simulation year
            environmental_system: Environmental system state
            economic_system: Economic system state
            governance_system: Governance system state
            infrastructure_system: Infrastructure system state
            
        Returns:
            dict: Demographic system results
        """
        print(f"Updating demographic system for year {year}...")
        
        # Update population size and urbanization
        population_update = self.update_population(
            year, 
            environmental_system, 
            economic_system
        )
        
        # Update education and skills
        education_update = self.update_education_skills(
            year,
            economic_system,
            governance_system
        )
        
        # Update spatial distribution 
        spatial_update = self.update_spatial_distribution(
            year,
            environmental_system,
            economic_system,
            infrastructure_system
        )
        
        # Calculate labor force characteristics
        labor_force = self.calculate_labor_force()
        
        # Calculate migration flows and remittances
        migration_flows = self.calculate_migration_flows()
        
        # Combine all updates
        results = {
            'year': year,
            'population': self.total_population,
            'urbanization_rate': self.urbanization_rate,
            'growth_rate': self.population_growth_rate,
            'fertility_rate': self.fertility_rate,
            'life_expectancy': self.life_expectancy,
            'education': self.education_distribution,
            'skills': self.skill_distribution,
            'literacy_rate': self.literacy_rate,
            'labor_force': labor_force,
            'migration': migration_flows,
            'regional_distribution': self.region_population
        }
        
        print(f"Demographic system updated for year {year}")
        return results
