#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main simulation runner for the Bangladesh Integrated Socioeconomic and Environmental System Model.
This file orchestrates the simulation by initializing all system components, running the simulation
steps, and producing the final outputs and visualizations.
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Import model components
from models.economic.economy import EconomicSystem
from models.environmental.environment import EnvironmentalSystem
from models.demographic.population import DemographicSystem
from models.infrastructure.infrastructure import InfrastructureSystem
from models.governance.governance import GovernanceSystem

# Import utilities
from utils.data_loader import DataLoader
from utils.config_manager import ConfigManager
from utils.result_processor import ResultProcessor
from utils.validation import ValidationMetrics
from utils.monte_carlo import MonteCarloSimulator
from utils.system_integrator import SystemIntegrator
from utils.visualizer import Visualizer

# Import visualization tools
from visualization.dashboard import Dashboard
from visualization.geo_visualizer import GeoVisualizer
from visualization.network_visualizer import NetworkVisualizer
from visualization.time_series_plotter import TimeSeriesPlotter


class BangladeshSimulation:
    """
    Main simulation class that coordinates all components of the Bangladesh integrated system model.
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the simulation with configuration settings.
        
        Args:
            config_path (str): Path to the main configuration file
        """
        print(f"Initializing Bangladesh Integrated System Simulation...")
        self.start_time = time.time()
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Set random seed for reproducibility
        np.random.seed(self.config['simulation']['random_seed'])
        
        # Initialize data loader
        self.data_loader = DataLoader(self.config['data']['data_dir'])
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Setup result processor
        self.result_processor = ResultProcessor(
            output_dir=self.config['output']['output_dir'],
            simulation_id=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        # Initialize Monte Carlo simulator if enabled
        if self.config['simulation']['monte_carlo']['enabled']:
            self.monte_carlo = MonteCarloSimulator(
                self,
                n_runs=self.config['simulation']['monte_carlo']['n_runs'],
                parameters=self.config['simulation']['monte_carlo']['parameters']
            )
        else:
            self.monte_carlo = None
            
        print(f"Simulation initialized in {time.time() - self.start_time:.2f} seconds")
    
    def _init_subsystems(self):
        """Initialize all subsystems of the simulation."""
        print("Initializing subsystems...")
        
        # Initialize economic system
        self.economic_system = EconomicSystem(
            config=self.config['economic'],
            data_loader=self.data_loader
        )
        
        # Initialize environmental system
        self.environmental_system = EnvironmentalSystem(
            config=self.config['environmental'],
            data_loader=self.data_loader
        )
        
        # Initialize demographic system
        self.demographic_system = DemographicSystem(
            config=self.config['demographic'],
            data_loader=self.data_loader
        )
        
        # Initialize infrastructure system
        self.infrastructure_system = InfrastructureSystem(
            config=self.config['infrastructure'],
            data_loader=self.data_loader
        )
        
        # Initialize governance system
        self.governance_system = GovernanceSystem(
            config=self.config['governance'],
            data_loader=self.data_loader
        )
        
        # Initialize system integrator
        self.system_integrator = SystemIntegrator(
            config=self.config
        )
        
        print("All subsystems initialized.")
    
    def step(self):
        """Execute one simulation step of the simulation, updating all systems."""
        # Log the current step
        print(f"Executing simulation step for year {self.current_year}")
        
        # Step 1: Update the environmental system first as it provides climate inputs
        if self.environmental_system:
            env_results = self.environmental_system.step(
                year=self.current_year,
                demographic_system=self.demographic_system,
                infrastructure_system=self.infrastructure_system,
                economic_system=self.economic_system
            )
            self.history['environmental'].append(env_results)
        
        # Step 2: Update the demographic system
        if self.demographic_system:
            demo_results = self.demographic_system.step(
                year=self.current_year,
                environmental_system=self.environmental_system,
                economic_system=self.economic_system,
                infrastructure_system=self.infrastructure_system
            )
            self.history['demographic'].append(demo_results)
        
        # Step 3: Update the economic system with inputs from other systems
        if self.economic_system:
            econ_results = self.economic_system.step(
                year=self.current_year,
                environmental_system=self.environmental_system,
                demographic_system=self.demographic_system,
                infrastructure_system=self.infrastructure_system,
                governance_system=self.governance_system
            )
            self.history['economic'].append(econ_results)
        
        # Step 4: Update the infrastructure system
        if self.infrastructure_system:
            infra_results = self.infrastructure_system.step(
                year=self.current_year,
                environmental_system=self.environmental_system,
                demographic_system=self.demographic_system,
                governance_system=self.governance_system
            )
            self.history['infrastructure'].append(infra_results)
        
        # Step 5: Update the governance system with inputs from all other systems
        if self.governance_system:
            gov_results = self.governance_system.step(
                year=self.current_year,
                economic_system=self.economic_system,
                environmental_system=self.environmental_system,
                demographic_system=self.demographic_system,
                infrastructure_system=self.infrastructure_system
            )
            self.history['governance'].append(gov_results)
        
        # Calculate and record integrated metrics for this time step
        integrated_metrics = self._calculate_integrated_metrics()
        self.history['integrated_metrics'].append(integrated_metrics)
        
        # Update the current year for the next time step
        self.current_year += 1
        
        return {
            'year': self.current_year - 1,
            'integrated_metrics': integrated_metrics,
            'environmental': self.history['environmental'][-1] if self.history['environmental'] else None,
            'demographic': self.history['demographic'][-1] if self.history['demographic'] else None,
            'economic': self.history['economic'][-1] if self.history['economic'] else None,
            'infrastructure': self.history['infrastructure'][-1] if self.history['infrastructure'] else None,
            'governance': self.history['governance'][-1] if self.history['governance'] else None
        }
        
    def _calculate_integrated_metrics(self):
        """Calculate integrated metrics across all systems for the current time step."""
        metrics = {'year': self.current_year}
        
        # Human Development Index components
        if self.demographic_system and hasattr(self.demographic_system, 'life_expectancy'):
            metrics['life_expectancy'] = self.demographic_system.life_expectancy
            
        if self.demographic_system and hasattr(self.demographic_system, 'education_index'):
            metrics['education_index'] = self.demographic_system.education_index
            
        if self.economic_system and hasattr(self.economic_system, 'gdp_per_capita'):
            metrics['gdp_per_capita'] = self.economic_system.gdp_per_capita
        
        # Calculate Human Development Index if components are available
        if all(key in metrics for key in ['life_expectancy', 'education_index', 'gdp_per_capita']):
            # Normalize life expectancy (25-85 years range)
            life_exp_index = (metrics['life_expectancy'] - 25) / (85 - 25)
            
            # GDP per capita index (log scale, $100-$75,000 range)
            income_index = (np.log(metrics['gdp_per_capita']) - np.log(100)) / (np.log(75000) - np.log(100))
            
            # Education index already normalized
            edu_index = metrics['education_index']
            
            # Calculate HDI as geometric mean of the three indices
            metrics['human_development_index'] = (life_exp_index * income_index * edu_index) ** (1/3)
        
        # Environmental sustainability indicator
        if self.environmental_system:
            if hasattr(self.environmental_system, 'climate_vulnerability_index'):
                metrics['climate_vulnerability'] = self.environmental_system.climate_vulnerability_index
                
            if hasattr(self.environmental_system, 'biodiversity_index'):
                metrics['biodiversity'] = self.environmental_system.biodiversity_index
        
        # Infrastructure development 
        if self.infrastructure_system and hasattr(self.infrastructure_system, 'get_infrastructure_quality_index'):
            metrics['infrastructure_quality'] = self.infrastructure_system.get_infrastructure_quality_index()
        
        # Governance quality
        if self.governance_system and hasattr(self.governance_system, 'governance_effectiveness'):
            metrics['governance_effectiveness'] = self.governance_system.governance_effectiveness
            
        # Sustainable Development Index
        # Combines HDI with environmental sustainability
        if 'human_development_index' in metrics and 'climate_vulnerability' in metrics:
            # Invert climate vulnerability so higher is better
            env_sustainability = 1 - metrics['climate_vulnerability']
            
            # Combine HDI and environmental sustainability
            metrics['sustainable_development_index'] = (
                metrics['human_development_index'] * 0.7 + env_sustainability * 0.3
            )
        
        # Social inequality metrics if available
        if self.economic_system and hasattr(self.economic_system, 'gini_coefficient'):
            metrics['inequality'] = self.economic_system.gini_coefficient
        
        # Regional development disparity
        if self.infrastructure_system and hasattr(self.infrastructure_system, 'regional_infrastructure'):
            # Calculate coefficient of variation of regional infrastructure metrics
            regional_values = [
                region_data.get('electricity_coverage', 0) 
                for region, region_data in self.infrastructure_system.regional_infrastructure.items()
            ]
            if regional_values:
                metrics['regional_disparity'] = np.std(regional_values) / np.mean(regional_values) if np.mean(regional_values) > 0 else 0
        
        return metrics
    
    def run(self):
        """
        Run the simulation for the specified number of time steps.
        """
        print(f"Starting simulation run...")
        start_year = self.config['simulation']['start_year']
        end_year = self.config['simulation']['end_year']
        time_step = self.config['simulation']['time_step']
        
        # Create time points for simulation
        self.time_points = np.arange(start_year, end_year + time_step, time_step)
        total_steps = len(self.time_points) - 1
        
        # Store results for each time step
        self.results = {
            'economic': [],
            'environmental': [],
            'demographic': [],
            'infrastructure': [],
            'governance': []
        }
        
        # Initialize current year
        self.current_year = start_year
        
        # Initialize history
        self.history = {
            'environmental': [],
            'demographic': [],
            'economic': [],
            'infrastructure': [],
            'governance': [],
            'integrated_metrics': []
        }
        
        # Run simulation for each time step
        for t in range(total_steps):
            next_year = self.time_points[t + 1]
            print(f"Simulating period {self.current_year} to {next_year} (Step {t+1}/{total_steps})")
            
            # Execute one simulation step
            step_results = self.step()
            
            # Store results
            self.results['environmental'].append(step_results['environmental'])
            self.results['demographic'].append(step_results['demographic'])
            self.results['economic'].append(step_results['economic'])
            self.results['infrastructure'].append(step_results['infrastructure'])
            self.results['governance'].append(step_results['governance'])
        
        # Process and save results
        self.result_processor.process_results(self.results, self.time_points[:-1])
        
        print(f"Simulation completed in {time.time() - self.start_time:.2f} seconds")
        return self.results
    
    def run_monte_carlo(self):
        """Run Monte Carlo simulation if enabled."""
        if self.monte_carlo:
            print("Starting Monte Carlo simulation...")
            mc_results = self.monte_carlo.run()
            self.result_processor.process_monte_carlo_results(mc_results)
            print("Monte Carlo simulation completed.")
            return mc_results
        else:
            print("Monte Carlo simulation not enabled in config.")
            return None
    
    def validate(self, validation_data=None):
        """
        Validate simulation results against historical data.
        
        Args:
            validation_data (dict, optional): Historical data to validate against.
                If None, will load from the data directory.
        """
        print("Validating simulation results...")
        
        # Load validation data if not provided
        if validation_data is None:
            validation_data = self.data_loader.load_validation_data()
        
        # Initialize validation metrics
        validator = ValidationMetrics(self.results, validation_data, self.time_points[:-1])
        validation_results = validator.compute_all_metrics()
        
        # Save validation results
        self.result_processor.save_validation_results(validation_results)
        
        print("Validation completed.")
        return validation_results
    
    def visualize(self):
        """Create visualizations of simulation results."""
        print("Generating visualizations...")
        
        # Create time series plots
        ts_plotter = TimeSeriesPlotter(self.results, self.time_points[:-1])
        ts_plots = ts_plotter.create_plots()
        self.result_processor.save_plots(ts_plots, 'time_series')
        
        # Create geospatial visualizations
        geo_viz = GeoVisualizer(self.results, self.config['visualization']['map_settings'])
        geo_plots = geo_viz.create_maps()
        self.result_processor.save_plots(geo_plots, 'geospatial')
        
        # Create network visualizations
        network_viz = NetworkVisualizer(
            infrastructure=self.results['infrastructure'],
            economic=self.results['economic']
        )
        network_plots = network_viz.create_network_plots()
        self.result_processor.save_plots(network_plots, 'networks')
        
        # Create interactive dashboard
        dashboard_path = Dashboard(
            self.results, 
            self.time_points[:-1],
            output_dir=self.config['output']['output_dir']
        )
        
        print(f"Visualizations saved to {self.config['output']['output_dir']}")
        print(f"Interactive dashboard available at {dashboard_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Bangladesh Integrated System Simulation')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Run Monte Carlo simulation')
    parser.add_argument('--validate', action='store_true',
                        help='Validate results against historical data')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip visualization generation')
    return parser.parse_args()


def main():
    """Main function to run the simulation."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize simulation
    simulation = BangladeshSimulation(config_path=args.config)
    
    # Run simulation
    results = simulation.run()
    
    # Run Monte Carlo simulation if requested
    if args.monte_carlo:
        mc_results = simulation.run_monte_carlo()
    
    # Validate results if requested
    if args.validate:
        validation_results = simulation.validate()
    
    # Create visualizations unless disabled
    if not args.no_visualize:
        simulation.visualize()
    
    print("Simulation process completed successfully.")


if __name__ == "__main__":
    main()
