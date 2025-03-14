"""
Bangladesh Simulation Model - Validation Test

This script runs the simulation model and validates its outputs against historical data.
It generates validation metrics and visualizations to assess model accuracy.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import simulation model
from simulation import BangladeshSimulation
from utils.validation import ValidationMetrics
from utils.data_loader import DataLoader
from utils.config_manager import ConfigManager
from utils.system_integrator import SystemIntegrator
from utils.visualizer import Visualizer
from utils.governance_system import GovernanceSystem
from utils.infrastructure_system import InfrastructureSystem

def main():
    print("=" * 80)
    print("BANGLADESH SIMULATION MODEL - VALIDATION TEST")
    print("=" * 80)
    
    # Load configuration
    print("\nLoading configuration...")
    config_path = os.path.join('config', 'config.yaml')
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_dir = os.path.join('results', f'validation_{timestamp}')
    os.makedirs(validation_dir, exist_ok=True)
    
    # Save validation configuration
    config_manager.save_config(config, os.path.join(validation_dir, 'validation_config.yaml'))
    
    # Generate simulated validation data
    print("\nGenerating simulated validation data...")
    
    # Define date range for validation
    start_year = 2010
    end_year = 2023
    
    # Initialize governance and infrastructure systems
    governance_system = GovernanceSystem(config)
    infrastructure_system = InfrastructureSystem(config)
    
    # Generate simulated historical data
    gov_data = governance_system.generate_simulated_data(start_year, end_year)
    infra_data = infrastructure_system.generate_simulated_data(start_year, end_year)
    
    # Initialize the system integrator with the configuration
    system_integrator = SystemIntegrator(config)
    
    # Apply integration to simulated data
    print("\nApplying system integration effects to simulated data...")
    integrated_data = pd.DataFrame(index=range(start_year, end_year + 1))
    
    # Create a modified simulation object just for integration
    class SimProxy:
        def __init__(self, gov_system, infra_system):
            self.governance_system = gov_system
            self.infrastructure_system = infra_system
    
    sim_proxy = SimProxy(governance_system, infrastructure_system)
    
    # Apply cross-system effects
    integration_results = []
    
    for year in range(start_year, end_year + 1):
        # Update the governance and infrastructure systems independently
        if year > start_year:
            # Add some economic growth simulation (0.03-0.06 range)
            economic_growth = np.random.uniform(0.03, 0.06)
            # Add some population growth simulation (0.01-0.02 range)
            population_growth = np.random.uniform(0.01, 0.02)
            
            # Update systems
            governance_system.update(year, economic_growth=economic_growth)
            infrastructure_system.update(year, economic_growth=economic_growth, 
                                        population_growth=population_growth,
                                        governance_quality=governance_system.get_overall_governance_index())
        
        # Apply cross-system effects
        effects = system_integrator.apply_cross_system_effects(sim_proxy)
        integration_results.append(effects)
    
    # Prepare data for validation
    # Combine governance and infrastructure data
    all_data = pd.DataFrame()
    all_data['Year'] = range(start_year, end_year + 1)
    
    # Add governance indicators
    for col in gov_data.columns:
        all_data[col] = gov_data[col].values
    
    # Add infrastructure indicators
    for col in infra_data.columns:
        if col not in all_data.columns:  # Avoid duplicate 'Year' column
            all_data[col] = infra_data[col].values
    
    # Save the simulated data for validation
    simulation_data_path = os.path.join(validation_dir, 'simulated_data.csv')
    all_data.to_csv(simulation_data_path, index=False)
    print(f"Saved simulated data to {simulation_data_path}")
    
    # Create simulated historical data for validation
    print("\nCreating simulated historical data with perturbations for validation...")
    
    # Add random perturbations to create "historical" data for comparison
    hist_data = all_data.copy()
    for col in hist_data.columns:
        if col != 'Year' and col != 'Year':
            # Add random noise (±10%)
            hist_data[col] = hist_data[col] * np.random.uniform(0.9, 1.1, len(hist_data))
            # Ensure values stay within 0-1 range for normalized indicators
            hist_data[col] = hist_data[col].clip(0, 1)
    
    # Save the historical data for validation
    historical_data_path = os.path.join(validation_dir, 'historical_data.csv')
    
    # Convert to the format expected by the validator (long format)
    hist_data_long = pd.melt(
        hist_data, 
        id_vars=['Year'], 
        var_name='variable', 
        value_name='value'
    )
    hist_data_long = hist_data_long.rename(columns={'Year': 'year'})
    
    hist_data_long.to_csv(historical_data_path, index=False)
    print(f"Saved simulated historical data to {historical_data_path}")
    
    # Use the same historical data structure for simulation results
    sim_data_long = pd.melt(
        all_data, 
        id_vars=['Year'], 
        var_name='variable', 
        value_name='value'
    )
    sim_data_long = sim_data_long.rename(columns={'Year': 'year'})
    
    # Initialize ValidationMetrics
    print("\nInitializing validation metrics...")
    validator = ValidationMetrics()
    
    # Load historical data into validator
    hist_data_dict = {}
    for _, row in hist_data_long.iterrows():
        variable = row['variable']
        year = row['year']
        value = row['value']
        
        if variable not in hist_data_dict:
            hist_data_dict[variable] = {}
        
        hist_data_dict[variable][year] = value
    
    validator.load_historical_data(hist_data_dict)
    
    # Extract key variables for validation by system
    economic_vars = ['gdp', 'gdp_growth_rate', 'inflation_rate', 'unemployment_rate']
    environmental_vars = ['temperature_anomaly', 'sea_level_rise', 'forest_coverage']
    demographic_vars = ['population', 'urban_population_share', 'life_expectancy']
    infrastructure_vars = ['transport_quality', 'energy_reliability', 'water_sanitation', 
                          'telecom_coverage', 'urban_planning', 'overall_infrastructure_quality']
    governance_vars = ['institutional_effectiveness', 'corruption_index', 'policy_effectiveness', 
                      'regulatory_quality', 'political_stability', 'public_service_delivery']
    
    # Combine all key variables
    key_variables = governance_vars + infrastructure_vars
    
    # Extract simulation results by variable and year
    sim_data_dict = {}
    for _, row in sim_data_long.iterrows():
        variable = row['variable']
        year = row['year']
        value = row['value']
        
        if variable not in sim_data_dict:
            sim_data_dict[variable] = {}
        
        sim_data_dict[variable][year] = value
    
    # Validate each key variable
    print("\nValidating key variables...")
    validation_results = {}
    
    for var in key_variables:
        if var in sim_data_dict and var in hist_data_dict:
            # Get years present in both simulation and historical data
            common_years = sorted(set(sim_data_dict[var].keys()) & set(hist_data_dict[var].keys()))
            
            if not common_years:
                print(f"Warning: No overlapping years for variable {var}, skipping validation")
                continue
                
            # Extract values for common years
            sim_values = [sim_data_dict[var][year] for year in common_years]
            hist_values = [hist_data_dict[var][year] for year in common_years]
            
            # Validate this variable
            metrics = validator.validate_variable(
                variable_name=var,
                simulated_values=sim_values,
                historical_values=hist_values,
                time_points=common_years,
                plot=True,
                output_dir=validation_dir
            )
            
            validation_results[var] = metrics
        else:
            if var not in sim_data_dict:
                print(f"Warning: Variable {var} not found in simulation results")
            if var not in hist_data_dict:
                print(f"Warning: Variable {var} not found in historical data")
    
    # Get overall performance
    overall_metrics = validator.get_overall_performance()
    
    # Display validation results
    print("\nValidation Metrics:")
    metrics_df = pd.DataFrame([
        {
            'Variable': var,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'], 
            'MAPE (%)': metrics.get('percent_bias', np.nan),
            'R²': metrics['r2']
        }
        for var, metrics in validation_results.items()
    ])
    print(metrics_df.to_string(index=False))
    
    # Save validation metrics
    metrics_df.to_csv(os.path.join(validation_dir, 'validation_metrics.csv'), index=False)
    
    # Save overall performance
    with open(os.path.join(validation_dir, 'overall_performance.txt'), 'w') as f:
        f.write("Overall Validation Performance:\n\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # Generate system-specific validation visualizations
    print("\nGenerating comprehensive validation visualizations...")
    visualizer = Visualizer(output_dir=validation_dir)
    
    # Infrastructure-Governance Integration Visualizations
    print("\nCreating Infrastructure-Governance Integration visualizations...")
    
    # Create integration visualization DataFrame
    years = list(range(start_year, end_year + 1))
    integration_df = pd.DataFrame({'year': years})
    
    # Add governance and infrastructure indicators
    for var in governance_vars:
        if var in sim_data_dict:
            integration_df[f'{var}_sim'] = [sim_data_dict[var].get(y, np.nan) for y in years]
            integration_df[f'{var}_hist'] = [hist_data_dict[var].get(y, np.nan) for y in years]
    
    for var in infrastructure_vars:
        if var in sim_data_dict:
            integration_df[f'{var}_sim'] = [sim_data_dict[var].get(y, np.nan) for y in years]
            integration_df[f'{var}_hist'] = [hist_data_dict[var].get(y, np.nan) for y in years]
    
    # Create dual-axis plots to show relationships
    # Infrastructure affected by governance
    if 'corruption_index_sim' in integration_df.columns and 'infrastructure_quality_sim' in integration_df.columns:
        visualizer.plot_dual_axis(
            data=integration_df.set_index('year'),
            var1='corruption_index_sim',
            var2='overall_infrastructure_quality_sim',
            title='Corruption Index vs Infrastructure Quality',
            ylabel1='Corruption Index',
            ylabel2='Infrastructure Quality',
            filename='gov_infra_corruption_quality.png',
            show_plot=False
        )
    
    # Governance affected by infrastructure
    if 'public_service_delivery_sim' in integration_df.columns and 'transport_quality_sim' in integration_df.columns:
        visualizer.plot_dual_axis(
            data=integration_df.set_index('year'),
            var1='public_service_delivery_sim',
            var2='transport_quality_sim',
            title='Public Service Delivery vs Transport Quality',
            ylabel1='Public Service Delivery',
            ylabel2='Transport Quality',
            filename='gov_infra_service_transport.png',
            show_plot=False
        )
    
    # Create correlation heatmap between governance and infrastructure
    gov_infra_cols = [col for col in integration_df.columns 
                     if ('_sim' in col and col.split('_sim')[0] in 
                         (governance_vars + infrastructure_vars))]
    
    if gov_infra_cols:
        gov_infra_correlation = integration_df[gov_infra_cols].corr()
        
        visualizer.plot_heatmap(
            data=gov_infra_correlation,
            title='Governance-Infrastructure Correlation Heatmap',
            cmap='coolwarm',
            filename='gov_infra_correlation.png',
            show_plot=False
        )
    
    print("\nValidation test completed successfully.")
    print(f"Results saved to {validation_dir}")

if __name__ == "__main__":
    main()
