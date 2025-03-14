#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader utility for the Bangladesh simulation model.
This module handles loading initial data for all components of the simulation.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path


class DataLoader:
    """
    Utility for loading and preprocessing data for the simulation model.
    Handles loading economic, demographic, environmental, infrastructure,
    and governance data from various file formats.
    """
    
    def __init__(self, data_dir):
        """
        Initialize the data loader with path to data directory.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        
        # Verify data directory exists
        if not self.data_dir.exists():
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Created data directory: {self.data_dir}")
        
        # Check for required data files
        self._verify_data_files()
        
        print(f"DataLoader initialized with data directory: {data_dir}")
    
    def _verify_data_files(self):
        """Verify that required data files exist, create samples if missing."""
        
        # Define the expected data files by system
        expected_files = {
            'economic': ['economic_data.csv', 'trade_data.csv', 'sector_data.json'],
            'demographic': ['population_data.csv', 'migration_data.csv'],
            'environmental': ['climate_data.csv', 'disaster_records.csv', 'land_use.csv'],
            'infrastructure': ['transport_data.csv', 'energy_data.csv', 'water_data.csv'],
            'governance': ['governance_indicators.csv', 'policy_data.json']
        }
        
        # Verify each system's data files
        for system, files in expected_files.items():
            system_dir = self.data_dir / system
            
            # Create system directory if it doesn't exist
            if not system_dir.exists():
                os.makedirs(system_dir, exist_ok=True)
                print(f"Created {system} data directory")
            
            # Check for each expected file
            for file in files:
                file_path = system_dir / file
                if not file_path.exists():
                    # Create sample data file if missing
                    self._create_sample_data(system, file, file_path)
                    print(f"Created sample data file: {file_path}")
    
    def _create_sample_data(self, system, filename, file_path):
        """Create sample data file with placeholder data."""
        
        ext = file_path.suffix.lower()
        
        if ext == '.csv':
            # Create sample CSV data
            if system == 'economic':
                if 'economic_data' in filename:
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'gdp_billion_usd': [50 + i*10 for i in range(24)],
                        'gdp_growth': [0.05 + np.random.normal(0, 0.01) for _ in range(24)],
                        'inflation': [0.06 + np.random.normal(0, 0.02) for _ in range(24)],
                        'exchange_rate': [70 + i for i in range(24)]
                    })
                else:  # trade_data
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'exports_billion_usd': [10 + i*2 for i in range(24)],
                        'imports_billion_usd': [15 + i*2 for i in range(24)],
                        'remittances_billion_usd': [5 + i*0.5 for i in range(24)]
                    })
            
            elif system == 'demographic':
                if 'population_data' in filename:
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'total_population_million': [130 + i*2 for i in range(24)],
                        'urban_population_pct': [0.25 + i*0.005 for i in range(24)],
                        'fertility_rate': [3.0 - i*0.05 for i in range(24)][:24]
                    })
                else:  # migration_data
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'internal_migrants_million': [1 + i*0.1 for i in range(24)],
                        'international_migrants_million': [0.5 + i*0.05 for i in range(24)]
                    })
            
            elif system == 'environmental':
                if 'climate_data' in filename:
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'temperature_anomaly': [0.5 + i*0.02 for i in range(24)],
                        'precipitation_change_pct': [np.random.normal(0, 5) for _ in range(24)],
                        'sea_level_rise_mm': [i*3 for i in range(24)]
                    })
                elif 'disaster_records' in filename:
                    # Generate random disaster data
                    years = []
                    types = []
                    impacts = []
                    deaths = []
                    damages = []
                    
                    disaster_types = ['flood', 'cyclone', 'drought', 'landslide']
                    for year in range(2000, 2024):
                        n_disasters = np.random.randint(1, 4)
                        for _ in range(n_disasters):
                            years.append(year)
                            types.append(np.random.choice(disaster_types))
                            impacts.append(np.random.uniform(0.01, 0.2))
                            deaths.append(np.random.randint(0, 5000))
                            damages.append(np.random.uniform(10, 2000))
                    
                    df = pd.DataFrame({
                        'year': years,
                        'disaster_type': types,
                        'impact_severity': impacts,
                        'deaths': deaths,
                        'economic_damage_million_usd': damages
                    })
                else:  # land_use
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'agricultural_pct': [60 - i*0.2 for i in range(24)],
                        'urban_pct': [5 + i*0.3 for i in range(24)],
                        'forest_pct': [15 - i*0.1 for i in range(24)],
                        'wetland_pct': [10 - i*0.05 for i in range(24)],
                        'other_pct': [10 + i*0.05 for i in range(24)]
                    })
            
            elif system == 'infrastructure':
                if 'transport_data' in filename:
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'road_network_km': [20000 + i*1000 for i in range(24)],
                        'railway_network_km': [2500 + i*50 for i in range(24)],
                        'road_quality_index': [0.3 + i*0.01 for i in range(24)],
                        'port_capacity_teu': [500000 + i*100000 for i in range(24)]
                    })
                elif 'energy_data' in filename:
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'electricity_coverage_pct': [40 + i*2 for i in range(24)],
                        'power_generation_mw': [5000 + i*500 for i in range(24)],
                        'renewable_pct': [1 + i*0.5 for i in range(24)],
                        'gas_pct': [80 - i*0.5 for i in range(24)]
                    })
                else:  # water_data
                    df = pd.DataFrame({
                        'year': range(2000, 2024),
                        'water_coverage_pct': [70 + i*1 for i in range(24)],
                        'sanitation_coverage_pct': [50 + i*1.5 for i in range(24)],
                        'irrigation_coverage_pct': [40 + i*1 for i in range(24)]
                    })
            
            elif system == 'governance':
                # governance_indicators
                df = pd.DataFrame({
                    'year': range(2000, 2024),
                    'institutional_effectiveness': [0.3 + i*0.01 for i in range(24)],
                    'corruption_index': [0.6 - i*0.01 for i in range(24)],
                    'political_stability': [0.4 + np.random.normal(0, 0.05) for _ in range(24)],
                    'regulatory_quality': [0.35 + i*0.01 for i in range(24)]
                })
            
            df.to_csv(file_path, index=False)
            
        elif ext == '.json':
            # Create sample JSON data
            if system == 'economic' and 'sector_data' in filename:
                data = {
                    'sectors': {
                        'agriculture': {
                            'gdp_share': 0.15,
                            'employment_share': 0.40,
                            'growth_rate': 0.03,
                            'exports_share': 0.05
                        },
                        'manufacturing': {
                            'gdp_share': 0.30,
                            'employment_share': 0.20,
                            'growth_rate': 0.08,
                            'exports_share': 0.80
                        },
                        'services': {
                            'gdp_share': 0.50,
                            'employment_share': 0.35,
                            'growth_rate': 0.06,
                            'exports_share': 0.15
                        },
                        'informal': {
                            'gdp_share': 0.05,
                            'employment_share': 0.05,
                            'growth_rate': 0.04,
                            'exports_share': 0.00
                        }
                    }
                }
            elif system == 'governance' and 'policy_data' in filename:
                data = {
                    'policies': [
                        {
                            'name': 'Vision 2041',
                            'start_year': 2021,
                            'end_year': 2041,
                            'targets': {
                                'gdp_growth': 0.09,
                                'poverty_reduction': 0.8,
                                'electricity_coverage': 1.0,
                                'infrastructure_investment': 0.12
                            }
                        },
                        {
                            'name': 'Climate Action Plan',
                            'start_year': 2020,
                            'end_year': 2035,
                            'targets': {
                                'renewable_energy_share': 0.4,
                                'emissions_reduction': 0.3,
                                'climate_resilience_investment': 0.05
                            }
                        }
                    ]
                }
            else:
                # Generic empty JSON
                data = {'data': 'sample'}
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def load_economic_data(self):
        """
        Load economic data for simulation.
        
        Returns:
            dict: Economic data parameters
        """
        economic_dir = self.data_dir / 'economic'
        
        # Load CSV data
        try:
            econ_df = pd.read_csv(economic_dir / 'economic_data.csv')
            trade_df = pd.read_csv(economic_dir / 'trade_data.csv')
            
            # Load most recent year for initialization
            latest_econ = econ_df.iloc[-1].to_dict()
            latest_trade = trade_df.iloc[-1].to_dict()
            
            # Load sector data from JSON
            with open(economic_dir / 'sector_data.json', 'r') as f:
                sector_data = json.load(f)
                
            # Combine data
            economic_data = {
                'initial_gdp': latest_econ.get('gdp_billion_usd', 400),
                'initial_gdp_growth': latest_econ.get('gdp_growth', 0.06),
                'initial_inflation_rate': latest_econ.get('inflation', 0.055),
                'initial_exchange_rate': latest_econ.get('exchange_rate', 85),
                'initial_exports': latest_trade.get('exports_billion_usd', 40),
                'initial_imports': latest_trade.get('imports_billion_usd', 50),
                'remittances': latest_trade.get('remittances_billion_usd', 20),
                'sector_data': sector_data.get('sectors', {})
            }
            
            return economic_data
        
        except Exception as e:
            print(f"Error loading economic data: {e}")
            # Return default values
            return {
                'initial_gdp': 400,
                'initial_gdp_growth': 0.06,
                'initial_inflation_rate': 0.055,
                'initial_exchange_rate': 85,
                'initial_exports': 40,
                'initial_imports': 50,
                'remittances': 20,
                'sector_data': {}
            }
    
    def load_demographic_data(self):
        """
        Load demographic data for simulation.
        
        Returns:
            dict: Demographic data parameters
        """
        demographic_dir = self.data_dir / 'demographic'
        
        try:
            # Load population data
            pop_df = pd.read_csv(demographic_dir / 'population_data.csv')
            migration_df = pd.read_csv(demographic_dir / 'migration_data.csv')
            
            # Get latest data
            latest_pop = pop_df.iloc[-1].to_dict()
            latest_migration = migration_df.iloc[-1].to_dict()
            
            # Create demographic data dictionary
            demographic_data = {
                'total_population': latest_pop.get('total_population_million', 169) * 1_000_000,
                'population_growth_rate': 0.01,  # Annual rate
                'urbanization_rate': latest_pop.get('urban_population_pct', 0.37),
                'fertility_rate': latest_pop.get('fertility_rate', 2.0),
                'internal_migrants': latest_migration.get('internal_migrants_million', 2.0) * 1_000_000,
                'international_migrants': latest_migration.get('international_migrants_million', 0.8) * 1_000_000,
                
                # Estimated age distribution
                'age_distribution': {
                    '0-14': 0.27,
                    '15-24': 0.19,
                    '25-34': 0.18,
                    '35-44': 0.14,
                    '45-54': 0.11,
                    '55-64': 0.07,
                    '65+': 0.04
                },
                
                # Regional population distribution (estimated)
                'region_population': {
                    'Dhaka': 0.31,
                    'Chittagong': 0.20,
                    'Rajshahi': 0.12,
                    'Khulna': 0.10,
                    'Barisal': 0.06,
                    'Sylhet': 0.07,
                    'Rangpur': 0.08,
                    'Mymensingh': 0.06
                }
            }
            
            return demographic_data
            
        except Exception as e:
            print(f"Error loading demographic data: {e}")
            # Return default values
            return {
                'total_population': 169_000_000,
                'population_growth_rate': 0.01,
                'urbanization_rate': 0.37,
                'fertility_rate': 2.0
            }
    
    def load_environmental_data(self):
        """
        Load environmental data for simulation.
        
        Returns:
            dict: Environmental data parameters
        """
        env_dir = self.data_dir / 'environmental'
        
        try:
            # Load climate data
            climate_df = pd.read_csv(env_dir / 'climate_data.csv')
            disaster_df = pd.read_csv(env_dir / 'disaster_records.csv')
            land_df = pd.read_csv(env_dir / 'land_use.csv')
            
            # Get latest climate data
            latest_climate = climate_df.iloc[-1].to_dict()
            latest_land = land_df.iloc[-1].to_dict()
            
            # Analyze disaster frequency
            disaster_stats = disaster_df.groupby('disaster_type').agg(
                frequency=('year', 'count'),
                avg_impact=('impact_severity', 'mean'),
                avg_deaths=('deaths', 'mean'),
                avg_damage=('economic_damage_million_usd', 'mean')
            ).to_dict('index')
            
            # Create environmental data dictionary
            environmental_data = {
                'temperature_anomaly': latest_climate.get('temperature_anomaly', 0.8),
                'precipitation_change_pct': latest_climate.get('precipitation_change_pct', 0),
                'sea_level_rise': latest_climate.get('sea_level_rise_mm', 50) / 1000,  # Convert to meters
                
                'land_use': {
                    'agricultural': latest_land.get('agricultural_pct', 55) / 100,
                    'urban': latest_land.get('urban_pct', 12) / 100,
                    'forest': latest_land.get('forest_pct', 13) / 100,
                    'wetland': latest_land.get('wetland_pct', 7) / 100,
                    'other': latest_land.get('other_pct', 13) / 100
                },
                
                'disaster_parameters': {
                    d_type: {
                        'annual_frequency': stats['frequency'] / len(disaster_df['year'].unique()),
                        'avg_severity': stats['avg_impact'],
                        'avg_deaths': stats['avg_deaths'],
                        'avg_damage_million_usd': stats['avg_damage']
                    } for d_type, stats in disaster_stats.items()
                }
            }
            
            return environmental_data
            
        except Exception as e:
            print(f"Error loading environmental data: {e}")
            # Return default values
            return {
                'temperature_anomaly': 0.8,
                'sea_level_rise': 0.05,  # meters
                'land_use': {
                    'agricultural': 0.55,
                    'urban': 0.12,
                    'forest': 0.13
                }
            }
    
    def load_infrastructure_data(self):
        """
        Load infrastructure data for simulation.
        
        Returns:
            dict: Infrastructure data parameters
        """
        infra_dir = self.data_dir / 'infrastructure'
        
        try:
            # Load infrastructure data
            transport_df = pd.read_csv(infra_dir / 'transport_data.csv')
            energy_df = pd.read_csv(infra_dir / 'energy_data.csv')
            water_df = pd.read_csv(infra_dir / 'water_data.csv')
            
            # Get latest data
            latest_transport = transport_df.iloc[-1].to_dict()
            latest_energy = energy_df.iloc[-1].to_dict()
            latest_water = water_df.iloc[-1].to_dict()
            
            # Create infrastructure data dictionary
            infrastructure_data = {
                # Transport
                'road_network_km': latest_transport.get('road_network_km', 40000),
                'railway_network_km': latest_transport.get('railway_network_km', 3500),
                'road_quality_index': latest_transport.get('road_quality_index', 0.5),
                'port_capacity_teu': latest_transport.get('port_capacity_teu', 2_500_000),
                
                # Energy
                'electricity_coverage': latest_energy.get('electricity_coverage_pct', 85) / 100,
                'power_generation_mw': latest_energy.get('power_generation_mw', 15000),
                'energy_mix': {
                    'gas': latest_energy.get('gas_pct', 70) / 100,
                    'coal': 0.05,
                    'oil': 0.08,
                    'hydro': 0.01,
                    'solar': latest_energy.get('renewable_pct', 10) / 200,  # Half of renewable
                    'wind': latest_energy.get('renewable_pct', 10) / 200,   # Half of renewable
                    'nuclear': 0.07,
                    'other': 0.01
                },
                
                # Water
                'water_supply_coverage': latest_water.get('water_coverage_pct', 90) / 100,
                'sanitation_coverage': latest_water.get('sanitation_coverage_pct', 80) / 100,
                'irrigation_coverage': latest_water.get('irrigation_coverage_pct', 60) / 100,
                
                # Telecom (estimated)
                'mobile_coverage': 0.95,
                'internet_coverage': 0.70,
                'broadband_penetration': 0.30
            }
            
            return infrastructure_data
            
        except Exception as e:
            print(f"Error loading infrastructure data: {e}")
            # Return default values
            return {
                'electricity_coverage': 0.85,
                'road_quality_index': 0.5,
                'water_supply_coverage': 0.9
            }
    
    def load_governance_data(self):
        """
        Load governance data for simulation.
        
        Returns:
            dict: Governance data parameters
        """
        gov_dir = self.data_dir / 'governance'
        
        try:
            # Load governance data
            gov_df = pd.read_csv(gov_dir / 'governance_indicators.csv')
            
            # Load policy data from JSON
            with open(gov_dir / 'policy_data.json', 'r') as f:
                policy_data = json.load(f)
            
            # Get latest data
            latest_gov = gov_df.iloc[-1].to_dict()
            
            # Create governance data dictionary
            governance_data = {
                'institutional_effectiveness': latest_gov.get('institutional_effectiveness', 0.4),
                'corruption_level': latest_gov.get('corruption_index', 0.5),
                'political_stability': latest_gov.get('political_stability', 0.45),
                'regulatory_quality': latest_gov.get('regulatory_quality', 0.45),
                'policies': policy_data.get('policies', [])
            }
            
            return governance_data
            
        except Exception as e:
            print(f"Error loading governance data: {e}")
            # Return default values
            return {
                'institutional_effectiveness': 0.4,
                'corruption_level': 0.5,
                'political_stability': 0.45
            }
    
    def load_gis_data(self, layer_name):
        """
        Load GIS data for spatial analysis.
        
        Args:
            layer_name (str): Name of the GIS layer to load
        
        Returns:
            geopandas.GeoDataFrame: Spatial data frame
        """
        gis_dir = self.data_dir / 'gis'
        
        # Create directory if it doesn't exist
        if not gis_dir.exists():
            os.makedirs(gis_dir, exist_ok=True)
        
        try:
            # For Bangladesh administrative boundaries
            if layer_name == 'bangladesh_divisions':
                file_path = gis_dir / 'bd_divisions.geojson'
                
                # Check if file exists
                if not file_path.exists():
                    # Generate simple fake GeoJSON for demo
                    dummy_geometry = self._create_dummy_bd_geometry()
                    with open(file_path, 'w') as f:
                        f.write(dummy_geometry)
                    print(f"Created dummy GIS data: {file_path}")
                
                # Load GeoJSON
                gdf = gpd.read_file(file_path)
                return gdf
                
            else:
                raise FileNotFoundError(f"Unknown GIS layer: {layer_name}")
                
        except Exception as e:
            print(f"Error loading GIS data: {e}")
            raise
    
    def _create_dummy_bd_geometry(self):
        """Create a dummy GeoJSON representation of Bangladesh divisions."""
        
        # Very simplified GeoJSON for demonstration purposes
        return '''{
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"division": "Dhaka", "area_km2": 20000, "population": 40000000},
                    "geometry": {"type": "Polygon", "coordinates": [[[90.0, 23.5], [90.5, 23.5], [90.5, 24.0], [90.0, 24.0], [90.0, 23.5]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"division": "Chittagong", "area_km2": 18000, "population": 30000000},
                    "geometry": {"type": "Polygon", "coordinates": [[[91.0, 22.5], [92.0, 22.5], [92.0, 23.0], [91.0, 23.0], [91.0, 22.5]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"division": "Rajshahi", "area_km2": 15000, "population": 20000000},
                    "geometry": {"type": "Polygon", "coordinates": [[[88.5, 24.0], [89.5, 24.0], [89.5, 25.0], [88.5, 25.0], [88.5, 24.0]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"division": "Khulna", "area_km2": 14000, "population": 18000000},
                    "geometry": {"type": "Polygon", "coordinates": [[[89.0, 22.0], [90.0, 22.0], [90.0, 23.0], [89.0, 23.0], [89.0, 22.0]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"division": "Barisal", "area_km2": 10000, "population": 10000000},
                    "geometry": {"type": "Polygon", "coordinates": [[[90.0, 22.0], [91.0, 22.0], [91.0, 22.5], [90.0, 22.5], [90.0, 22.0]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"division": "Sylhet", "area_km2": 12000, "population": 12000000},
                    "geometry": {"type": "Polygon", "coordinates": [[[91.0, 24.0], [92.0, 24.0], [92.0, 25.0], [91.0, 25.0], [91.0, 24.0]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"division": "Rangpur", "area_km2": 13000, "population": 15000000},
                    "geometry": {"type": "Polygon", "coordinates": [[[89.0, 25.0], [90.0, 25.0], [90.0, 26.0], [89.0, 26.0], [89.0, 25.0]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"division": "Mymensingh", "area_km2": 10500, "population": 11000000},
                    "geometry": {"type": "Polygon", "coordinates": [[[90.0, 24.0], [91.0, 24.0], [91.0, 25.0], [90.0, 25.0], [90.0, 24.0]]]}
                }
            ]
        }'''
