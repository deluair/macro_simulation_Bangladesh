#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bangladesh Simulation Model - Infrastructure-Governance Integration Test

This script tests the integration between the governance and infrastructure systems,
demonstrating how they interact and provide feedback to each other through the
system integrator.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import required modules
from utils.governance_system import GovernanceSystem
from utils.infrastructure_system import InfrastructureSystem
from utils.system_integrator import SystemIntegrator
from utils.config_manager import ConfigManager
from utils.visualizer import Visualizer

def main():
    """Run a demonstration of governance-infrastructure integration."""
    print("=" * 80)
    print("BANGLADESH SIMULATION MODEL - GOVERNANCE-INFRASTRUCTURE INTEGRATION TEST")
    print("=" * 80)
    
    # Load configuration
    print("\nLoading configuration...")
    config_path = os.path.join('config', 'config.yaml')
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'integration_test_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define simulation period
    start_year = 2010
    end_year = 2030
    years = list(range(start_year, end_year + 1))
    
    print(f"\nRunning integration simulation from {start_year} to {end_year}...")
    
    # Initialize governance and infrastructure systems
    governance_system = GovernanceSystem(config)
    infrastructure_system = InfrastructureSystem(config)
    
    # Initialize system integrator
    system_integrator = SystemIntegrator(config)
    
    # Create a simulation proxy for the integrator to use
    class SimProxy:
        def __init__(self, gov_system, infra_system):
            self.governance_system = gov_system
            self.infrastructure_system = infra_system
    
    sim_proxy = SimProxy(governance_system, infrastructure_system)
    
    # Create results storage
    results = {
        'Year': years,
        
        # Governance indicators
        'institutional_effectiveness': [],
        'corruption_index': [],
        'policy_effectiveness': [],
        'regulatory_quality': [],
        'political_stability': [],
        'public_service_delivery': [],
        'governance_index': [],
        
        # Infrastructure indicators
        'transport_quality': [],
        'energy_reliability': [],
        'water_sanitation': [],
        'telecom_coverage': [],
        'urban_planning': [],
        'infrastructure_quality': [],
        
        # Integration effects
        'gov_to_infra_effects': [],
        'infra_to_gov_effects': []
    }
    
    # Run simulation
    for year in years:
        print(f"Simulating year {year}...")
        
        # Record the current state
        gov_indicators = governance_system.get_indicators()
        infra_indicators = infrastructure_system.get_indicators()
        
        # Store governance indicators
        results['institutional_effectiveness'].append(gov_indicators['institutional_effectiveness'])
        results['corruption_index'].append(gov_indicators['corruption_index'])
        results['policy_effectiveness'].append(gov_indicators['policy_effectiveness'])
        results['regulatory_quality'].append(gov_indicators['regulatory_quality'])
        results['political_stability'].append(gov_indicators['political_stability'])
        results['public_service_delivery'].append(gov_indicators['public_service_delivery'])
        results['governance_index'].append(governance_system.get_overall_governance_index())
        
        # Store infrastructure indicators
        results['transport_quality'].append(infra_indicators['transport_quality'])
        results['energy_reliability'].append(infra_indicators['energy_reliability'])
        results['water_sanitation'].append(infra_indicators['water_sanitation'])
        results['telecom_coverage'].append(infra_indicators['telecom_coverage'])
        results['urban_planning'].append(infra_indicators['urban_planning'])
        results['infrastructure_quality'].append(infrastructure_system.get_overall_quality())
        
        # Apply system integration effects
        effects = system_integrator.apply_cross_system_effects(sim_proxy)
        
        # Record integration effect magnitudes
        if 'governance_infrastructure' in effects:
            gov_to_infra_effect = sum(effects['governance_infrastructure']
                                     ['governance_to_infrastructure'].values())
            infra_to_gov_effect = sum(effects['governance_infrastructure']
                                     ['infrastructure_to_governance'].values())
            
            results['gov_to_infra_effects'].append(gov_to_infra_effect)
            results['infra_to_gov_effects'].append(infra_to_gov_effect)
        else:
            results['gov_to_infra_effects'].append(0)
            results['infra_to_gov_effects'].append(0)
        
        # Update systems for next year if not the last year
        if year < end_year:
            # Simulate economic growth (ranges from 3-6%)
            economic_growth = np.random.uniform(0.03, 0.06)
            
            # Simulate population growth (ranges from 1-2%)
            population_growth = np.random.uniform(0.01, 0.02)
            
            # Update systems with economic and population growth
            governance_system.update(
                year=year, 
                economic_growth=economic_growth
            )
            
            infrastructure_system.update(
                year=year,
                economic_growth=economic_growth,
                population_growth=population_growth,
                governance_quality=governance_system.get_overall_governance_index()
            )
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_csv_path = os.path.join(results_dir, 'integration_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved integration results to {results_csv_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = Visualizer(output_dir=results_dir)
    
    # Set up the DataFrame with year as index for visualizer
    df = results_df.set_index('Year')
    
    # 1. Time series: Governance Index and Infrastructure Quality
    visualizer.plot_time_series(
        data=df,
        variables=['governance_index', 'infrastructure_quality'],
        title="Governance and Infrastructure Quality Over Time",
        ylabel="Quality Index (0-1)",
        legend_labels=['Governance Index', 'Infrastructure Quality'],
        filename="gov_infra_indices.png",
        show_plot=False
    )
    
    # 2. Time series: Corruption and Infrastructure Quality
    visualizer.plot_time_series(
        data=df,
        variables=['corruption_index', 'infrastructure_quality'],
        title="Corruption Index vs Infrastructure Quality",
        ylabel="Index Value (0-1)",
        legend_labels=['Corruption Index', 'Infrastructure Quality'],
        filename="corruption_infrastructure.png",
        show_plot=False
    )
    
    # 3. Time series: Policy Effectiveness and Transport Quality
    visualizer.plot_time_series(
        data=df,
        variables=['policy_effectiveness', 'transport_quality'],
        title="Policy Effectiveness vs Transport Quality",
        ylabel="Index Value (0-1)",
        legend_labels=['Policy Effectiveness', 'Transport Quality'],
        filename="policy_transport.png",
        show_plot=False
    )
    
    # 4. Time series: Institutional Effectiveness and Energy Reliability
    visualizer.plot_time_series(
        data=df,
        variables=['institutional_effectiveness', 'energy_reliability'],
        title="Institutional Effectiveness vs Energy Reliability",
        ylabel="Index Value (0-1)",
        legend_labels=['Institutional Effectiveness', 'Energy Reliability'],
        filename="institutional_energy.png",
        show_plot=False
    )
    
    # 5. Correlation heatmap of governance and infrastructure indicators
    gov_infra_indicators = [
        'institutional_effectiveness', 'corruption_index', 'policy_effectiveness',
        'regulatory_quality', 'transport_quality', 'energy_reliability',
        'water_sanitation', 'telecom_coverage', 'urban_planning'
    ]
    
    # Only include indicators that are in the DataFrame
    available_indicators = [ind for ind in gov_infra_indicators if ind in df.columns]
    
    correlation_matrix = df[available_indicators].corr()
    
    visualizer.plot_heatmap(
        data=correlation_matrix,
        title="Correlation Between Governance and Infrastructure Indicators",
        cmap="coolwarm",
        annot=True,
        filename="gov_infra_correlation.png",
        show_plot=False
    )
    
    # 6. Plot the magnitude of integration effects over time
    visualizer.plot_time_series(
        data=df,
        variables=['gov_to_infra_effects', 'infra_to_gov_effects'],
        title="Magnitude of Integration Effects Over Time",
        ylabel="Effect Magnitude",
        legend_labels=['Governance → Infrastructure', 'Infrastructure → Governance'],
        filename="integration_effects.png",
        show_plot=False
    )
    
    # Try to create a radar chart if the visualizer supports it
    try:
        # Extract the first, middle, and last year data for comparison
        first_year = start_year
        middle_year = start_year + (end_year - start_year) // 2
        last_year = end_year
        
        # Define the indicators to include
        radar_indicators = [
            'institutional_effectiveness', 'corruption_index', 'policy_effectiveness',
            'transport_quality', 'energy_reliability', 'water_sanitation'
        ]
        
        # Create data for radar chart
        radar_data = {
            f'{first_year}': df.loc[first_year, radar_indicators].to_dict(),
            f'{middle_year}': df.loc[middle_year, radar_indicators].to_dict(),
            f'{last_year}': df.loc[last_year, radar_indicators].to_dict()
        }
        
        # Call radar chart method if available
        if hasattr(visualizer, 'plot_radar_chart'):
            visualizer.plot_radar_chart(
                data=radar_data,
                title="Governance-Infrastructure Indicators Over Time",
                filename="gov_infra_radar.png",
                show_plot=False
            )
    except Exception as e:
        print(f"Note: Radar chart generation skipped: {e}")
    
    print(f"\nIntegration test completed. Results and visualizations saved to {results_dir}")
    print("\nKey findings:")
    
    # Calculate some statistics to report
    governance_change = results_df['governance_index'].iloc[-1] - results_df['governance_index'].iloc[0]
    infrastructure_change = results_df['infrastructure_quality'].iloc[-1] - results_df['infrastructure_quality'].iloc[0]
    
    # Calculate correlation between governance and infrastructure
    gov_infra_corr = results_df['governance_index'].corr(results_df['infrastructure_quality'])
    
    print(f"1. Overall governance index changed by {governance_change:.4f} over the simulation period")
    print(f"2. Overall infrastructure quality changed by {infrastructure_change:.4f} over the simulation period")
    print(f"3. Correlation between governance and infrastructure indices: {gov_infra_corr:.4f}")
    
    # Calculate the strongest influences
    gov_indicators = ['institutional_effectiveness', 'corruption_index', 'policy_effectiveness', 
                     'regulatory_quality', 'political_stability']
    infra_components = ['transport_quality', 'energy_reliability', 'water_sanitation', 
                       'telecom_coverage', 'urban_planning']
    
    # Find strongest correlation between governance indicators and infrastructure components
    max_corr = 0
    max_gov = ""
    max_infra = ""
    
    for gov in gov_indicators:
        for infra in infra_components:
            if gov in results_df.columns and infra in results_df.columns:
                corr = abs(results_df[gov].corr(results_df[infra]))
                if corr > max_corr:
                    max_corr = corr
                    max_gov = gov
                    max_infra = infra
    
    if max_gov and max_infra:
        print(f"4. Strongest relationship: {max_gov} and {max_infra} (correlation: {max_corr:.4f})")
    
    print("\nTo further explore these results, check the visualizations in the results directory.")

if __name__ == "__main__":
    main()
