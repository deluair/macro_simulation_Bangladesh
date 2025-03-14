"""
Bangladesh Integrated Socioeconomic and Environmental Simulation
Main Execution Script

This script runs the full Bangladesh simulation model for future projections,
generates visualizations, and saves results to disk.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
from typing import Dict, Any, List
import logging
from pathlib import Path
import json
from scipy import ndimage  # Added for gaussian_filter used in heat maps

# Import simulation model
from models.simulation import BangladeshSimulation
from utils.config_manager import ConfigManager
from utils.result_processor import ResultProcessor
from utils.visualizer import Visualizer
from utils.html_report_generator import HTMLReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str = 'config/simulation_config.yaml') -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation(config: Dict[str, Any]) -> BangladeshSimulation:
    """Run the simulation for the specified duration."""
    simulation = BangladeshSimulation(config)
    duration = config['simulation']['duration']
    
    logger.info(f"Starting simulation for {duration} years")
    
    for year in range(duration):
        simulation.step()
        if (year + 1) % 5 == 0:
            logger.info(f"Completed year {year + 1}/{duration}")
    
    return simulation

def create_development_trajectory_plot(simulation: BangladeshSimulation) -> None:
    """Create a plot showing the development trajectory over time."""
    history = simulation.get_history()
    years = [state['time_step'] for state in history]
    
    plt.figure(figsize=(12, 8))
    plt.plot(years, [state['state']['development_index'] for state in history], 
             label='Development Index', linewidth=2)
    plt.plot(years, [state['state']['sustainability_index'] for state in history], 
             label='Sustainability Index', linewidth=2)
    plt.plot(years, [state['state']['resilience_index'] for state in history], 
             label='Resilience Index', linewidth=2)
    plt.plot(years, [state['state']['wellbeing_index'] for state in history], 
             label='Wellbeing Index', linewidth=2)
    
    plt.title('Bangladesh Development Trajectory', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Index Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig('results/development_trajectory.png')
    plt.close()

def create_sector_analysis_plot(simulation: BangladeshSimulation) -> None:
    """Create a plot showing sector performance over time."""
    history = simulation.get_history()
    years = [state['time_step'] for state in history]
    
    plt.figure(figsize=(12, 8))
    for sector in simulation.models['economic'].sectors:
        sector_data = [state['models']['economic']['sectors'][sector]['gdp_share'] 
                      for state in history]
        plt.plot(years, sector_data, label=sector.capitalize(), linewidth=2)
    
    plt.title('Economic Sector Performance', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('GDP Share', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig('results/sector_analysis.png')
    plt.close()

def create_environmental_impact_plot(simulation: BangladeshSimulation) -> None:
    """Create a plot showing environmental indicators over time."""
    history = simulation.get_history()
    years = [state['time_step'] for state in history]
    
    plt.figure(figsize=(12, 8))
    plt.plot(years, [state['models']['environmental'].state['flood_risk'] 
                    for state in history], label='Flood Risk', linewidth=2)
    plt.plot(years, [state['models']['environmental'].state['crop_yield_index'] 
                    for state in history], label='Crop Yield Index', linewidth=2)
    plt.plot(years, [state['models']['environmental'].state['water_stress_index'] 
                    for state in history], label='Water Stress Index', linewidth=2)
    
    plt.title('Environmental Indicators', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Index Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig('results/environmental_impact.png')
    plt.close()

def create_demographic_analysis_plot(simulation: BangladeshSimulation) -> None:
    """Create a plot showing demographic indicators over time."""
    history = simulation.get_history()
    years = [state['time_step'] for state in history]
    
    plt.figure(figsize=(12, 8))
    plt.plot(years, [state['models']['demographic'].state['human_development_index'] 
                    for state in history], label='Human Development Index', linewidth=2)
    plt.plot(years, [state['models']['demographic'].state['urbanization_rate'] 
                    for state in history], label='Urbanization Rate', linewidth=2)
    plt.plot(years, [state['models']['demographic'].state['social_cohesion_index'] 
                    for state in history], label='Social Cohesion Index', linewidth=2)
    
    plt.title('Demographic Indicators', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Index Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig('results/demographic_analysis.png')
    plt.close()

def create_infrastructure_analysis_plot(simulation: BangladeshSimulation) -> None:
    """Create a plot showing infrastructure indicators over time."""
    history = simulation.get_history()
    years = [state['time_step'] for state in history]
    
    plt.figure(figsize=(12, 8))
    plt.plot(years, [state['models']['infrastructure'].state['infrastructure_quality_index'] 
                    for state in history], label='Infrastructure Quality', linewidth=2)
    plt.plot(years, [state['models']['infrastructure'].state['connectivity_index'] 
                    for state in history], label='Connectivity Index', linewidth=2)
    plt.plot(years, [state['models']['infrastructure'].state['efficiency_index'] 
                    for state in history], label='Efficiency Index', linewidth=2)
    
    plt.title('Infrastructure Development', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Index Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig('results/infrastructure_analysis.png')
    plt.close()

def create_governance_analysis_plot(simulation: BangladeshSimulation) -> None:
    """Create a plot showing governance indicators over time."""
    history = simulation.get_history()
    years = [state['time_step'] for state in history]
    
    plt.figure(figsize=(12, 8))
    plt.plot(years, [state['models']['governance'].state['governance_effectiveness_index'] 
                    for state in history], label='Governance Effectiveness', linewidth=2)
    plt.plot(years, [state['models']['governance'].state['social_progress_index'] 
                    for state in history], label='Social Progress', linewidth=2)
    plt.plot(years, [state['models']['governance'].state['institutional_quality'] 
                    for state in history], label='Institutional Quality', linewidth=2)
    
    plt.title('Governance Indicators', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Index Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig('results/governance_analysis.png')
    plt.close()

def generate_summary_report(simulation: BangladeshSimulation) -> None:
    """Generate a summary report of the simulation results."""
    history = simulation.get_history()
    final_state = history[-1]
    
    report = f"""
    Bangladesh Development Simulation Summary Report
    ==============================================
    
    Final Year Results:
    ------------------
    Development Index: {final_state['state']['development_index']:.3f}
    Sustainability Index: {final_state['state']['sustainability_index']:.3f}
    Resilience Index: {final_state['state']['resilience_index']:.3f}
    Wellbeing Index: {final_state['state']['wellbeing_index']:.3f}
    
    Economic Indicators:
    -------------------
    Total GDP: ${final_state['models']['economic'].state['total_gdp']:,.2f}
    Unemployment Rate: {final_state['models']['economic'].state['unemployment_rate']:.2%}
    Trade Balance: ${final_state['models']['economic'].state['trade_balance']:,.2f}
    
    Environmental Indicators:
    ------------------------
    Flood Risk: {final_state['models']['environmental'].state['flood_risk']:.3f}
    Crop Yield Index: {final_state['models']['environmental'].state['crop_yield_index']:.3f}
    Water Stress Index: {final_state['models']['environmental'].state['water_stress_index']:.3f}
    
    Demographic Indicators:
    ----------------------
    Population: {final_state['models']['demographic'].population['total']:,.0f}
    Urban Share: {final_state['models']['demographic'].population['urban_share']:.2%}
    Human Development Index: {final_state['models']['demographic'].state['human_development_index']:.3f}
    
    Infrastructure Indicators:
    -------------------------
    Infrastructure Quality: {final_state['models']['infrastructure'].state['infrastructure_quality_index']:.3f}
    Connectivity Index: {final_state['models']['infrastructure'].state['connectivity_index']:.3f}
    Efficiency Index: {final_state['models']['infrastructure'].state['efficiency_index']:.3f}
    
    Governance Indicators:
    ---------------------
    Governance Effectiveness: {final_state['models']['governance'].state['governance_effectiveness_index']:.3f}
    Social Progress: {final_state['models']['governance'].state['social_progress_index']:.3f}
    Institutional Quality: {final_state['models']['governance'].state['institutional_quality']:.3f}
    """
    
    with open('results/summary_report.txt', 'w') as f:
        f.write(report)

def main():
    """Main entry point for running the simulation."""
    logger.info("Starting Bangladesh Development Simulation")
    
    # Create output directories if they don't exist
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Run simulation
    simulation = run_simulation(config)
    
    # Process results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    simulation_id = f"sim_{timestamp}"
    results_dir = output_dir / simulation_id
    results_dir.mkdir(exist_ok=True)
    
    # Create processor with correct parameters
    processor = ResultProcessor(output_dir=str(results_dir), simulation_id=simulation_id)
    
    # Generate results dataframe and statistics
    processor.process_simulation_results(simulation)
    results_df = processor.get_results_dataframe()
    summary_stats = processor.get_summary_statistics()
    
    # Save results to CSV
    results_df.to_csv(results_dir / 'simulation_results.csv')
    
    # Save summary statistics to JSON
    import json
    with open(results_dir / 'summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create plots directory
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    visualizer = Visualizer(output_dir=str(results_dir))
    visualize_results(results_df, results_dir, visualizer)
    
    # Create section-specific plot directories
    for section in ['economic', 'demographic', 'environmental', 'infrastructure', 'governance']:
        section_dir = plots_dir / section
        section_dir.mkdir(exist_ok=True)
        
        # Generate section-specific plots
        if hasattr(visualizer, f'create_{section}_plots'):
            # Extract data for this section
            section_data = {}
            
            # Extract years from results
            years = sorted(set(results_df.index))
            section_data['years'] = years
            
            # Extract data for specific sections
            if section == 'economic':
                section_data['gdp_projection'] = results_df.get(('economic', 'total_gdp'), []) if ('economic', 'total_gdp') in results_df.columns else []
                
                # Define sectors from results if available
                sectors = [col[1] for col in results_df.columns if col[0] == 'economic' and 'sector_' in col[1]]
                if sectors:
                    section_data['sectors'] = [s.replace('sector_', '') for s in sectors]
                    section_data['sector_contributions'] = [results_df.iloc[-1].get(('economic', s), 0) for s in sectors]
                
                # Employment by sector
                if section_data.get('sectors') and ('economic', 'employment_rate') in results_df.columns:
                    # Generate employment by sector based on latest data
                    emp_data = [results_df.iloc[-1].get(('economic', 'employment_' + s.replace('sector_', '')), 0) for s in sectors]
                    section_data['employment_by_sector'] = emp_data
                
                # Add trade balance if available
                if ('economic', 'trade_balance') in results_df.columns:
                    section_data['trade_balance'] = results_df[('economic', 'trade_balance')].tolist()
                
            elif section == 'demographic':
                if ('demographic', 'total_population') in results_df.columns:
                    section_data['population'] = results_df[('demographic', 'total_population')].tolist()
                
                # Define age groups if available
                if ('demographic', 'age_group_0_14') in results_df.columns:
                    section_data['age_groups'] = ['0-14', '15-24', '25-39', '40-54', '55-64', '65+']
                    age_cols = ['age_group_0_14', 'age_group_15_24', 'age_group_25_39', 
                               'age_group_40_54', 'age_group_55_64', 'age_group_65_plus']
                    # Use latest values
                    section_data['age_distribution'] = [results_df.iloc[-1].get(('demographic', col), 0) for col in age_cols]
                
                # Urban vs rural population
                if ('demographic', 'urban_population') in results_df.columns and ('demographic', 'rural_population') in results_df.columns:
                    section_data['urban_population'] = results_df[('demographic', 'urban_population')].tolist()
                    section_data['rural_population'] = results_df[('demographic', 'rural_population')].tolist()
                
                # HDI if available
                if ('demographic', 'human_development_index') in results_df.columns:
                    section_data['hdi'] = results_df[('demographic', 'human_development_index')].tolist()
                
            elif section == 'environmental':
                if ('environmental', 'temperature') in results_df.columns:
                    section_data['temperature'] = results_df[('environmental', 'temperature')].tolist()
                
                # Flood risk data if available
                if ('environmental', 'flood_risk_high') in results_df.columns:
                    section_data['flood_risk_categories'] = ['Very High', 'High', 'Medium', 'Low', 'Very Low']
                    risk_cols = ['flood_risk_very_high', 'flood_risk_high', 'flood_risk_medium', 
                                'flood_risk_low', 'flood_risk_very_low']
                    # Use latest values
                    section_data['flood_risk_percentages'] = [results_df.iloc[-1].get(('environmental', col), 0) for col in risk_cols]
                
                # Water quality data if available
                if ('environmental', 'water_quality_rivers') in results_df.columns:
                    section_data['water_sources'] = ['Rivers', 'Lakes', 'Groundwater', 'Rainwater', 'Processed']
                    water_cols = ['water_quality_rivers', 'water_quality_lakes', 'water_quality_groundwater', 
                                 'water_quality_rainwater', 'water_quality_processed']
                    # Use latest values
                    section_data['water_quality'] = [results_df.iloc[-1].get(('environmental', col), 0) for col in water_cols]
                
                # Environmental health index if available
                if ('environmental', 'environmental_health_index') in results_df.columns:
                    section_data['environmental_health'] = results_df[('environmental', 'environmental_health_index')].tolist()
                
            elif section == 'infrastructure':
                if ('infrastructure', 'investment_percent_gdp') in results_df.columns:
                    section_data['infrastructure_investment'] = results_df[('infrastructure', 'investment_percent_gdp')].tolist()
                
                # Infrastructure quality by type
                if ('infrastructure', 'quality_roads') in results_df.columns:
                    section_data['infrastructure_types'] = ['Roads', 'Bridges', 'Ports', 'Electricity', 'Water', 'Telecom']
                    quality_cols = ['quality_roads', 'quality_bridges', 'quality_ports', 
                                   'quality_electricity', 'quality_water', 'quality_telecom']
                    # Use latest values
                    section_data['infrastructure_quality'] = [results_df.iloc[-1].get(('infrastructure', col), 0) for col in quality_cols]
                
                # Connectivity index if available
                if ('infrastructure', 'connectivity_index') in results_df.columns:
                    section_data['connectivity_index'] = results_df[('infrastructure', 'connectivity_index')].tolist()
                
                # Resilience index if available
                if ('infrastructure', 'resilience_index') in results_df.columns:
                    section_data['resilience_index'] = results_df[('infrastructure', 'resilience_index')].tolist()
                
            elif section == 'governance':
                if ('governance', 'effectiveness_index') in results_df.columns:
                    section_data['governance_effectiveness'] = results_df[('governance', 'effectiveness_index')].tolist()
                
                # Policy implementation success by area
                if ('governance', 'policy_economic') in results_df.columns:
                    section_data['policy_areas'] = ['Economic', 'Social', 'Environmental', 'Education', 'Healthcare', 'Infrastructure']
                    policy_cols = ['policy_economic', 'policy_social', 'policy_environmental', 
                                  'policy_education', 'policy_healthcare', 'policy_infrastructure']
                    # Use latest values
                    section_data['policy_implementation'] = [results_df.iloc[-1].get(('governance', col), 0) for col in policy_cols]
                
                # Corruption index if available
                if ('governance', 'corruption_index') in results_df.columns:
                    section_data['corruption_index'] = results_df[('governance', 'corruption_index')].tolist()
                
                # Institutional quality if available
                if ('governance', 'institutional_quality') in results_df.columns:
                    section_data['institutional_quality'] = results_df[('governance', 'institutional_quality')].tolist()
            
            # Call the section-specific plot method with the extracted data
            getattr(visualizer, f'create_{section}_plots')(section_data, section_dir)
        
        # Generate composite plots with the same data
        if hasattr(visualizer, f'create_{section}_composite_plot'):
            # Use the same section data prepared above
            getattr(visualizer, f'create_{section}_composite_plot')(section_data, plots_dir / f'{section}_composite.png')
    
    # Process integrated model data
    logger.info("Processing integrated model data")
    integrated_model_data = process_all_models(simulation)
    
    # NEW CODE: Generate geographic visualizations
    logger.info("Generating geographic visualizations")
    geo_viz_paths = generate_geo_visualizations(simulation, results_dir, integrated_model_data)
    
    # Perform cross-model sensitivity analysis
    logger.info("Performing cross-model sensitivity analysis")
    sensitivity_results = analyze_cross_model_sensitivities(
        integrated_model_data['integrated_data'], 
        integrated_model_data['time_steps']
    )
    
    # Generate HTML report
    logger.info("Generating HTML report")
    report_generator = HTMLReportGenerator(output_dir, simulation_id)
    report_path = report_generator.generate_report("Bangladesh Development Simulation Results", geo_viz_paths)
    
    logger.info(f"Report generated: {report_path}")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("Simulation completed successfully")
    
    return {
        'simulation': simulation,
        'results_df': results_df,
        'summary_stats': summary_stats,
        'report_path': report_path,
        'results_dir': results_dir,
        'integrated_model_data': integrated_model_data,
        'sensitivity_results': sensitivity_results,
        'geo_visualization_paths': geo_viz_paths
    }

def visualize_results(results_df, output_dir, visualizer):
    """Generate visualizations for standard simulation results."""
    
    # Create time series data format
    time_series_data = {}
    
    # Handle multi-index DataFrame from ResultProcessor
    if isinstance(results_df.columns, pd.MultiIndex):
        # Extract data from multi-index DataFrame using more efficient methods
        for component, variable in results_df.columns:
            var_name = f"{component}_{variable}"
            
            if var_name not in time_series_data:
                time_series_data[var_name] = {'years': [], 'values': []}
            
            # Get values for this variable
            series = results_df[(component, variable)]
            
            # Add non-NA values to time series data using list comprehensions
            non_na_values = [(year, value) for year, value in series.items() if not pd.isna(value)]
            if non_na_values:  # Check if we have any values
                years, values = zip(*non_na_values)
                time_series_data[var_name]['years'] = list(years)
                time_series_data[var_name]['values'] = list(values)
    else:
        # Handle old format (if still needed) with more efficient approach
        # Group by variable first to avoid repeated dictionary lookups
        for var, group in results_df.groupby('variable'):
            time_series_data[var] = {
                'years': group['year'].tolist(),
                'values': group['value'].tolist()
            }
    
    # Economic indicators
    economic_vars = ['economic_total_gdp', 'economic_gdp_growth', 'economic_inflation_rate', 'economic_unemployment_rate']
    econ_df = pd.DataFrame({
        'year': sorted(set([year for var in economic_vars if var in time_series_data 
                          for year in time_series_data[var]['years']]))
    }).set_index('year')
    
    for var in economic_vars:
        if var in time_series_data:
            years = time_series_data[var]['years']
            values = time_series_data[var]['values']
            
            # Create sorted pairs
            year_value_pairs = sorted(zip(years, values), key=lambda x: x[0])
            sorted_years, sorted_values = zip(*year_value_pairs) if year_value_pairs else ([], [])
            
            # Add to dataframe
            for year, value in zip(sorted_years, sorted_values):
                econ_df.loc[year, var] = value
    
    # Plot economic indicators
    if not econ_df.empty and any(var in econ_df.columns for var in economic_vars):
        visualizer.plot_time_series(
            data=econ_df,
            variables=[var for var in economic_vars if var in econ_df.columns],
            title='Economic Indicators Projection',
            ylabel='Value',
            filename='economic_indicators.png',
            show_plot=False
        )
    
    # GDP and Growth Rate
    if 'gdp' in time_series_data and 'gdp_growth_rate' in time_series_data:
        gdp_growth_df = pd.DataFrame({
            'year': sorted(set(time_series_data['gdp']['years'] + time_series_data['gdp_growth_rate']['years']))
        }).set_index('year')
        
        # Add GDP data
        for year, value in sorted(zip(time_series_data['gdp']['years'], time_series_data['gdp']['values'])):
            gdp_growth_df.loc[year, 'gdp'] = value
        
        # Add Growth Rate data
        for year, value in sorted(zip(time_series_data['gdp_growth_rate']['years'], time_series_data['gdp_growth_rate']['values'])):
            gdp_growth_df.loc[year, 'gdp_growth_rate'] = value
        
        visualizer.plot_dual_axis(
            data=gdp_growth_df,
            primary_var='gdp',
            secondary_var='gdp_growth_rate',
            title='GDP and Growth Rate Projection',
            primary_label='GDP (Billion USD)',
            secondary_label='GDP Growth Rate (%)',
            primary_color='blue',
            secondary_color='red',
            filename='gdp_projection.png',
            show_plot=False
        )
    
    # Population and Urbanization
    if 'population' in time_series_data and 'urban_population_share' in time_series_data:
        pop_urban_df = pd.DataFrame({
            'year': sorted(set(time_series_data['population']['years'] + time_series_data['urban_population_share']['years']))
        }).set_index('year')
        
        # Add population data
        for year, value in sorted(zip(time_series_data['population']['years'], time_series_data['population']['values'])):
            pop_urban_df.loc[year, 'population'] = value / 1000000  # Convert to millions
        
        # Add urbanization data
        for year, value in sorted(zip(time_series_data['urban_population_share']['years'], time_series_data['urban_population_share']['values'])):
            pop_urban_df.loc[year, 'urban_population_share'] = value
        
        visualizer.plot_dual_axis(
            data=pop_urban_df,
            primary_var='population',
            secondary_var='urban_population_share',
            title='Population and Urbanization Projection',
            primary_label='Population (Millions)',
            secondary_label='Urban Population (%)',
            primary_color='green',
            secondary_color='magenta',
            filename='population_projection.png',
            show_plot=False
        )
    
    # Environmental Indicators
    env_vars = ['temperature_anomaly', 'sea_level_rise', 'forest_coverage']
    env_df = pd.DataFrame({
        'year': sorted(set([year for var in env_vars if var in time_series_data 
                          for year in time_series_data[var]['years']]))
    }).set_index('year')
    
    for var in env_vars:
        if var in time_series_data:
            years = time_series_data[var]['years']
            values = time_series_data[var]['values']
            
            # Create sorted pairs
            year_value_pairs = sorted(zip(years, values), key=lambda x: x[0])
            sorted_years, sorted_values = zip(*year_value_pairs) if year_value_pairs else ([], [])
            
            # Add to dataframe
            for year, value in zip(sorted_years, sorted_values):
                env_df.loc[year, var] = value
    
    # Plot environmental indicators
    if not env_df.empty and any(var in env_df.columns for var in env_vars):
        visualizer.plot_time_series(
            data=env_df,
            variables=[var for var in env_vars if var in env_df.columns],
            title='Environmental Indicators Projection',
            ylabel='Value',
            filename='environmental_projection.png',
            show_plot=False
        )
    
    # Infrastructure Development
    infra_vars = ['electricity_coverage', 'water_supply_coverage', 'telecom_coverage', 'road_network', 'urban_planning_quality']
    infra_df = pd.DataFrame({
        'year': sorted(set([year for var in infra_vars if var in time_series_data 
                          for year in time_series_data[var]['years']]))
    }).set_index('year')
    
    for var in infra_vars:
        if var in time_series_data:
            years = time_series_data[var]['years']
            values = time_series_data[var]['values']
            
            # Create sorted pairs
            year_value_pairs = sorted(zip(years, values), key=lambda x: x[0])
            sorted_years, sorted_values = zip(*year_value_pairs) if year_value_pairs else ([], [])
            
            # Add to dataframe
            for year, value in zip(sorted_years, sorted_values):
                infra_df.loc[year, var] = value
    
    # Plot infrastructure indicators
    if not infra_df.empty and any(var in infra_df.columns for var in infra_vars):
        visualizer.plot_time_series(
            data=infra_df,
            variables=[var for var in infra_vars if var in infra_df.columns],
            title='Infrastructure Development Projection',
            ylabel='Coverage (%)',
            filename='infrastructure_projection.png',
            show_plot=False
        )
    
    # Governance Indicators
    gov_vars = ['institutional_effectiveness', 'corruption_index', 'regulatory_quality']
    gov_df = pd.DataFrame({
        'year': sorted(set([year for var in gov_vars if var in time_series_data 
                          for year in time_series_data[var]['years']]))
    }).set_index('year')
    
    for var in gov_vars:
        if var in time_series_data:
            years = time_series_data[var]['years']
            values = time_series_data[var]['values']
            
            # Create sorted pairs
            year_value_pairs = sorted(zip(years, values), key=lambda x: x[0])
            sorted_years, sorted_values = zip(*year_value_pairs) if year_value_pairs else ([], [])
            
            # Add to dataframe
            for year, value in zip(sorted_years, sorted_values):
                gov_df.loc[year, var] = value
    
    # Plot governance indicators
    if not gov_df.empty and any(var in gov_df.columns for var in gov_vars):
        visualizer.plot_time_series(
            data=gov_df,
            variables=[var for var in gov_vars if var in gov_df.columns],
            title='Governance Indicators Projection',
            ylabel='Index Value',
            filename='governance_projection.png',
            show_plot=False
        )
    
    # Governance-Infrastructure Interaction
    if (not gov_df.empty and any(var in gov_df.columns for var in gov_vars) and 
        not infra_df.empty and any(var in infra_df.columns for var in infra_vars)):
        
        # Get the latest year data for both governance and infrastructure
        latest_years = sorted(set(gov_df.index) & set(infra_df.index))
        if latest_years:
            latest_year = max(latest_years)
            
            # Create radar chart data for the latest year
            gov_infra_radar_data = {}
            
            # Add governance variables
            for var in gov_vars:
                if var in gov_df.columns:
                    gov_infra_radar_data[var] = gov_df.loc[latest_year, var]
            
            # Add infrastructure variables
            for var in infra_vars:
                if var in infra_df.columns:
                    gov_infra_radar_data[var] = infra_df.loc[latest_year, var]
            
            # Create radar chart
            if gov_infra_radar_data:
                radar_data = pd.DataFrame({
                    'category': list(gov_infra_radar_data.keys()),
                    'value': list(gov_infra_radar_data.values())
                })
                
                visualizer.plot_radar(
                    data=radar_data,
                    categories='category',
                    values='value',
                    title=f'Governance-Infrastructure Status ({latest_year})',
                    filename='governance_infrastructure_radar.png',
                    show_plot=False
                )
        
        # Create correlation heatmap between governance and infrastructure variables
        # First, ensure we have common years to calculate correlations
        common_years = sorted(set(gov_df.index) & set(infra_df.index))
        
        if len(common_years) > 1:
            # Extract governance and infrastructure data for common years
            gov_vars_present = [var for var in gov_vars if var in gov_df.columns]
            infra_vars_present = [var for var in infra_vars if var in infra_df.columns]
            
            if gov_vars_present and infra_vars_present:
                # Calculate correlation matrix
                gov_data = gov_df.loc[common_years, gov_vars_present]
                infra_data = infra_df.loc[common_years, infra_vars_present]
                
                # Combine the dataframes
                combined_df = pd.concat([gov_data, infra_data], axis=1)
                
                # Calculate correlation
                corr_matrix = combined_df.corr()
                
                # Extract just the correlations between governance and infrastructure
                gov_infra_corr = corr_matrix.loc[gov_vars_present, infra_vars_present]
                
                # Plot heatmap
                visualizer.plot_heatmap(
                    data=gov_infra_corr,
                    title='Governance-Infrastructure Correlation',
                    cmap='coolwarm',
                    center=0,
                    filename='governance_infrastructure_correlation.png',
                    show_plot=False
                )

def visualize_monte_carlo_results(mc_results, mc_summary, output_dir, visualizer):
    """Generate visualizations for Monte Carlo simulation results."""
    
    # Extract key variables from Monte Carlo results
    key_vars = ['gdp', 'population', 'electricity_coverage', 'institutional_effectiveness']
    
    # Process Monte Carlo results for each key variable
    for var in key_vars:
        if var in mc_summary:
            # Get years
            years = sorted(mc_summary[var].keys())
            
            # Extract mean, lower_bound, upper_bound for each year
            means = [mc_summary[var][year]['mean'] for year in years]
            lower_bounds = [mc_summary[var][year]['lower_bound'] for year in years]
            upper_bounds = [mc_summary[var][year]['upper_bound'] for year in years]
            
            # Create DataFrame
            mc_df = pd.DataFrame({
                'year': years,
                'mean': means,
                'lower_bound': lower_bounds,
                'upper_bound': upper_bounds
            }).set_index('year')
            
            # Plot Monte Carlo results with uncertainty bands
            visualizer.plot_uncertainty_band(
                data=mc_df,
                mean_col='mean',
                lower_col='lower_bound',
                upper_col='upper_bound',
                title=f'Monte Carlo Simulation: {var.replace("_", " ").title()}',
                ylabel=var.replace("_", " ").title(),
                filename=f'monte_carlo_{var}.png',
                show_plot=False
            )
    
    # Check if we have governance and infrastructure variables
    if ('institutional_effectiveness' in mc_summary and 
        'corruption_index' in mc_summary and 
        'electricity_coverage' in mc_summary and
        'water_supply_coverage' in mc_summary):
        
        # Get the final year projection
        years = sorted(mc_summary['gdp'].keys())
        final_year = years[-1]
        
        # Extract governance-infrastructure correlation distribution
        gov_vars = ['institutional_effectiveness', 'corruption_index', 'regulatory_quality']
        infra_vars = ['electricity_coverage', 'water_supply_coverage', 'telecom_coverage']
        
        gov_infra_correlations = []
        
        # Calculate governance-infrastructure correlations across Monte Carlo runs
        for run_id, run_results in mc_results.items():
            # Extract time series for this run
            run_gov_data = {var: [] for var in gov_vars if var in run_results}
            run_infra_data = {var: [] for var in infra_vars if var in run_results}
            run_years = sorted(run_results.keys())
            
            # Extract values for each variable over time
            for year in run_years:
                for var in gov_vars:
                    if var in run_results[year]:
                        run_gov_data[var].append(run_results[year][var])
                
                for var in infra_vars:
                    if var in run_results[year]:
                        run_infra_data[var].append(run_results[year][var])
            
            # Calculate correlation for each governance-infrastructure pair
            for gov_var in run_gov_data:
                for infra_var in run_infra_data:
                    if len(run_gov_data[gov_var]) > 1 and len(run_infra_data[infra_var]) > 1:
                        # Ensure equal length
                        min_len = min(len(run_gov_data[gov_var]), len(run_infra_data[infra_var]))
                        corr = np.corrcoef(
                            run_gov_data[gov_var][:min_len], 
                            run_infra_data[infra_var][:min_len]
                        )[0, 1]
                        
                        gov_infra_correlations.append({
                            'governance': gov_var,
                            'infrastructure': infra_var,
                            'correlation': corr
                        })
        
        # Create boxplot of governance-infrastructure correlations
        if gov_infra_correlations:
            corr_df = pd.DataFrame(gov_infra_correlations)
            
            # Create variable pair labels
            corr_df['pair'] = corr_df['governance'].str.replace('_', ' ').str.title() + ' - ' + corr_df['infrastructure'].str.replace('_', ' ').str.title()
            
            # Group by pair and create boxplot
            visualizer.plot_boxplot(
                data=corr_df,
                categories='pair',
                values='correlation',
                title='Governance-Infrastructure Correlation Distribution (Monte Carlo)',
                xlabel='Variable Pairs',
                ylabel='Correlation Coefficient',
                filename='monte_carlo_gov_infra_correlation.png',
                show_plot=False
            )

def process_all_models(simulation, processor=None):
    """
    Process data from all models efficiently using functional programming patterns.
    
    This function extracts key metrics from all models in parallel using map and list comprehensions,
    creating a comprehensive dataset that combines outputs from all model components.
    
    Args:
        simulation: BangladeshSimulation instance
        processor: Optional ResultProcessor instance for additional processing
        
    Returns:
        dict: Integrated data from all models
    """
    # Get all model history
    history = simulation.get_history()
    
    # Define model types
    model_types = ['economic', 'demographic', 'environmental', 'infrastructure', 'governance']
    
    # Extract time steps
    try:
        time_steps = [state['time_step'] for state in history]
    except (TypeError, KeyError):
        # If history is not a list of dictionaries with time_step keys,
        # create a list of time steps based on the current_step
        time_steps = list(range(simulation.current_step))
    
    # Define common key metrics for each model type
    model_key_metrics = {
        'economic': ['total_gdp', 'gdp_growth', 'inflation_rate', 'unemployment_rate', 'trade_balance'],
        'demographic': ['total_population', 'birth_rate', 'death_rate', 'migration_rate', 'urbanization_rate'],
        'environmental': ['carbon_emissions', 'forest_coverage', 'water_stress', 'air_quality_index', 'natural_disaster_risk'],
        'infrastructure': ['energy_access', 'transport_quality', 'water_sanitation', 'internet_access', 'housing_quality'],
        'governance': ['stability_index', 'corruption_index', 'policy_effectiveness', 'rule_of_law', 'regulatory_quality']
    }
    
    # Extract data for all models using map and functional approach
    integrated_data = {model_type: {} for model_type in model_types}
    
    # Process each model type
    for model_type in model_types:
        # Check if the simulation has a models attribute
        if hasattr(simulation, 'models') and model_type in simulation.models:
            # Get metrics for this model
            metrics = model_key_metrics[model_type]
            
            # Initialize data structure for each metric
            for metric in metrics:
                integrated_data[model_type][metric] = {
                    'values': [],
                    'years': time_steps.copy()
                }
            
            # Use map to extract metric values across all time steps
            for metric in metrics:
                try:
                    # Use list comprehension to extract values for each time step
                    values = [state['models'][model_type].state.get(metric, np.nan) 
                             for state in history if 'models' in state]
                    integrated_data[model_type][metric]['values'] = values
                except (KeyError, AttributeError, TypeError):
                    # Handle missing data
                    integrated_data[model_type][metric]['values'] = [np.nan] * len(time_steps)
        else:
            # If the simulation doesn't have a models attribute, try to access the system directly
            system_attr = f"{model_type}_system"
            if hasattr(simulation, system_attr):
                # Get metrics for this model
                metrics = model_key_metrics[model_type]
                
                # Initialize data structure for each metric
                for metric in metrics:
                    integrated_data[model_type][metric] = {
                        'values': [],
                        'years': time_steps.copy()
                    }
                
                # Try to get the system's state
                system = getattr(simulation, system_attr)
                if hasattr(system, 'get_state'):
                    state = system.get_state()
                    for metric in metrics:
                        try:
                            # Use the current state for all time steps
                            value = state.get(metric, np.nan)
                            integrated_data[model_type][metric]['values'] = [value] * len(time_steps)
                        except (KeyError, AttributeError, TypeError):
                            # Handle missing data
                            integrated_data[model_type][metric]['values'] = [np.nan] * len(time_steps)
    
    # Calculate inter-model correlations
    correlations = {}
    
    # Define pairs of metrics to correlate across models
    correlation_pairs = [
        ('economic.total_gdp', 'environmental.carbon_emissions'),
        ('economic.gdp_growth', 'governance.stability_index'),
        ('demographic.urbanization_rate', 'infrastructure.energy_access'),
        ('environmental.water_stress', 'demographic.migration_rate'),
        ('governance.corruption_index', 'economic.trade_balance')
    ]
    
    # Calculate correlations
    for pair in correlation_pairs:
        model1, metric1 = pair[0].split('.')
        model2, metric2 = pair[1].split('.')
        
        if (model1 in integrated_data and metric1 in integrated_data[model1] and
            model2 in integrated_data and metric2 in integrated_data[model2]):
            
            values1 = integrated_data[model1][metric1]['values']
            values2 = integrated_data[model2][metric2]['values']
            
            # Ensure values are numeric and of the same length
            values1 = [float(v) if v is not None and not pd.isna(v) else np.nan for v in values1]
            values2 = [float(v) if v is not None and not pd.isna(v) else np.nan for v in values2]
            
            # Filter out NaN values
            valid_pairs = [(v1, v2) for v1, v2 in zip(values1, values2) 
                          if not (pd.isna(v1) or pd.isna(v2))]
            
            if valid_pairs:
                valid_values1, valid_values2 = zip(*valid_pairs)
                
                # Calculate correlation
                try:
                    corr = np.corrcoef(valid_values1, valid_values2)[0, 1]
                    correlations[f"{pair[0]}_vs_{pair[1]}"] = corr
                except:
                    correlations[f"{pair[0]}_vs_{pair[1]}"] = np.nan
    
    return {
        'integrated_data': integrated_data,
        'correlations': correlations,
        'time_steps': time_steps
    }

def analyze_cross_model_sensitivities(integrated_data, time_steps):
    """
    Analyze sensitivities between models using map and functional programming.
    
    This function calculates how changes in metrics from one model affect metrics in other models,
    using rolling windows to measure elasticities and sensitivities.
    
    Args:
        integrated_data: Dictionary of integrated model data from process_all_models
        time_steps: List of time steps corresponding to the data points
        
    Returns:
        dict: Cross-model sensitivity metrics
    """
    # Define the key models and metrics to analyze
    key_model_metrics = {
        'economic': ['total_gdp', 'gdp_growth'],
        'demographic': ['total_population', 'urbanization_rate'],
        'environmental': ['carbon_emissions', 'water_stress'],
        'infrastructure': ['energy_access', 'transport_quality'],
        'governance': ['stability_index', 'policy_effectiveness']
    }
    
    # Initialize results dictionary
    sensitivities = {}
    elasticities = {}
    
    # Define all possible cross-model relationships
    model_pairs = [(m1, m2) for m1 in key_model_metrics.keys() for m2 in key_model_metrics.keys() if m1 != m2]
    
    # Calculate period-to-period changes for all metrics
    changes = {}
    for model, metrics in key_model_metrics.items():
        if model in integrated_data:
            changes[model] = {}
            for metric in metrics:
                if metric in integrated_data[model]:
                    values = integrated_data[model][metric]['values']
                    
                    # Calculate percentage changes between consecutive periods
                    # Using functional approach with map and zip
                    pct_changes = list(map(
                        lambda pair: (pair[1] - pair[0]) / pair[0] if pair[0] != 0 and not pd.isna(pair[0]) and not pd.isna(pair[1]) else np.nan,
                        zip(values[:-1], values[1:])
                    ))
                    
                    # Store percentage changes with corresponding time periods
                    changes[model][metric] = {
                        'periods': time_steps[1:],  # Skip the first period
                        'pct_changes': pct_changes
                    }
    
    # Calculate sensitivities between model metrics
    for model1, model2 in model_pairs:
        if model1 in changes and model2 in changes:
            for metric1 in key_model_metrics[model1]:
                for metric2 in key_model_metrics[model2]:
                    if metric1 in changes[model1] and metric2 in changes[model2]:
                        # Get percentage changes
                        pct_changes1 = changes[model1][metric1]['pct_changes']
                        pct_changes2 = changes[model2][metric2]['pct_changes']
                        periods = changes[model1][metric1]['periods']
                        
                        # Ensure same length
                        min_len = min(len(pct_changes1), len(pct_changes2))
                        if min_len > 1:
                            # Filter out NaN values
                            valid_pairs = [(p, c1, c2) for p, c1, c2 in zip(periods[:min_len], pct_changes1[:min_len], pct_changes2[:min_len])
                                         if not (pd.isna(c1) or pd.isna(c2))]
                            
                            if valid_pairs:
                                valid_periods, valid_changes1, valid_changes2 = zip(*valid_pairs)
                                
                                # Calculate elasticity (% change in metric2 / % change in metric1)
                                # Using map function for efficient calculation
                                elasticity_values = list(map(
                                    lambda pair: pair[1] / pair[0] if pair[0] != 0 else np.nan,
                                    zip(valid_changes1, valid_changes2)
                                ))
                                
                                # Calculate average elasticity (excluding extremes)
                                filtered_elasticities = [e for e in elasticity_values if not pd.isna(e) and abs(e) < 10]  # Filter out extreme values
                                
                                if filtered_elasticities:
                                    avg_elasticity = sum(filtered_elasticities) / len(filtered_elasticities)
                                    elasticities[f"{model1}.{metric1}_to_{model2}.{metric2}"] = avg_elasticity
                                    
                                    # Store individual elasticity values for time series analysis
                                    sensitivities[f"{model1}.{metric1}_to_{model2}.{metric2}"] = {
                                        'periods': valid_periods,
                                        'elasticities': elasticity_values
                                    }
    
    # Identify the most significant cross-model relationships
    significant_relationships = []
    
    if elasticities:
        # Sort elasticities by absolute value
        sorted_elasticities = sorted(
            [(k, v) for k, v in elasticities.items() if not pd.isna(v)],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Take top relationships
        top_n = min(10, len(sorted_elasticities))
        significant_relationships = sorted_elasticities[:top_n]
    
    return {
        'elasticities': elasticities,
        'sensitivities': sensitivities,
        'significant_relationships': significant_relationships
    }

def generate_geo_visualizations(simulation, output_dir, integrated_data=None):
    """
    Generate geographic visualizations for simulation results.
    
    Args:
        simulation: BangladeshSimulation instance
        output_dir: Directory to save visualizations
        integrated_data: Optional pre-computed integrated data
    
    Returns:
        dict: Paths to created visualizations
    """
    from visualization.geo_visualizer import GeoVisualizer
    
    # Create output directory
    maps_dir = Path(output_dir) / 'maps'
    maps_dir.mkdir(exist_ok=True)
    
    # Initialize geo visualizer
    geo_vis = GeoVisualizer(output_dir=str(maps_dir))
    
    # Print available shapefiles
    if hasattr(geo_vis, 'bd_shapefiles') and geo_vis.bd_shapefiles:
        logger.info(f"Available Bangladesh shapefiles: {list(geo_vis.bd_shapefiles.keys())}")
    else:
        logger.warning("No Bangladesh shapefiles available.")
    
    # Paths to created visualizations
    visualization_paths = {}
    
    # Get integrated data from all models if not provided
    if integrated_data is None:
        integrated_data = process_all_models(simulation)
    
    # 1. Create GDP by division choropleth map
    # Generate sample division-level GDP data from simulation
    try:
        divisions = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh']
        # Use economic data from simulation if available
        if ('economic' in integrated_data['integrated_data'] and 
            'total_gdp' in integrated_data['integrated_data']['economic']):
            # Get final GDP value
            final_gdp = integrated_data['integrated_data']['economic']['total_gdp']['values'][-1]
            
            # Generate division-level GDP based on population weightings
            division_populations = {
                'Dhaka': 0.32,       # 32% of population
                'Chittagong': 0.20,   # 20% of population
                'Rajshahi': 0.12,     # 12% of population
                'Khulna': 0.10,       # 10% of population
                'Barisal': 0.06,      # 6% of population
                'Sylhet': 0.06,       # 6% of population
                'Rangpur': 0.09,      # 9% of population
                'Mymensingh': 0.05    # 5% of population
            }
            
            # Calculate division GDPs
            division_gdps = {div: final_gdp * weight for div, weight in division_populations.items()}
            
            # Create choropleth map - divisions level
            choropleth_path = geo_vis.create_choropleth(
                data=division_gdps,
                title='GDP by Division (Billion BDT)',
                filename='gdp_by_division',
                cmap='Greens',
                admin_level='division',
                legend_title='GDP (Billion BDT)'
            )
            
            visualization_paths['gdp_choropleth_division'] = choropleth_path
            
            # Create a country-level outline map with the divisions data 
            # This demonstrates using a different admin_level shapefile with the same data
            country_choropleth_path = geo_vis.create_choropleth(
                data=division_gdps,
                title='GDP by Division - Country Overview',
                filename='gdp_country_outline',
                cmap='Greens',
                admin_level='country',  # Use country boundary
                legend_title='GDP (Billion BDT)'
            )
            
            visualization_paths['gdp_choropleth_country'] = country_choropleth_path
    except Exception as e:
        logger.error(f"Failed to create GDP choropleth maps: {e}")
    
    # 2. Create bubble map of major cities with population size and economic activity
    try:
        # Create data for major cities with population and economic activity
        cities_data = [
            {'name': 'Dhaka', 'lat': 23.8103, 'lon': 90.4125, 'population': 21.0, 'growth': 5.2},
            {'name': 'Chittagong', 'lat': 22.3569, 'lon': 91.7832, 'population': 8.9, 'growth': 4.8},
            {'name': 'Khulna', 'lat': 22.8456, 'lon': 89.5403, 'population': 2.8, 'growth': 3.5},
            {'name': 'Rajshahi', 'lat': 24.3745, 'lon': 88.6042, 'population': 2.5, 'growth': 3.2},
            {'name': 'Sylhet', 'lat': 24.8949, 'lon': 91.8687, 'population': 2.7, 'growth': 4.1},
            {'name': 'Barisal', 'lat': 22.7010, 'lon': 90.3535, 'population': 2.1, 'growth': 2.9},
            {'name': 'Rangpur', 'lat': 25.7439, 'lon': 89.2752, 'population': 2.3, 'growth': 3.0},
            {'name': 'Mymensingh', 'lat': 24.7471, 'lon': 90.4203, 'population': 1.8, 'growth': 2.8},
            {'name': 'Comilla', 'lat': 23.4682, 'lon': 91.1702, 'population': 1.5, 'growth': 3.7},
            {'name': 'Narayanganj', 'lat': 23.6238, 'lon': 90.5000, 'population': 2.9, 'growth': 4.9}
        ]
        
        # Scale growth rates based on simulation data if available
        if ('economic' in integrated_data['integrated_data'] and 
            'gdp_growth' in integrated_data['integrated_data']['economic']):
            
            # Get average GDP growth from simulation
            growth_values = [v for v in integrated_data['integrated_data']['economic']['gdp_growth']['values'] 
                            if not pd.isna(v)]
            avg_growth = sum(growth_values) / len(growth_values) if growth_values else 3.5
            
            # Scale city growth rates around the average
            growth_factor = avg_growth / 3.5  # Normalize against baseline of 3.5%
            for city in cities_data:
                city['growth'] = city['growth'] * growth_factor
        
        # Create bubble maps with different administrative level backgrounds
        for admin_level in ['country', 'division', 'district']:
            if admin_level in geo_vis.bd_shapefiles:
                bubble_map_path = geo_vis.create_bubble_map(
                    location_data=cities_data,
                    title=f'Major Cities: Population and Economic Growth ({admin_level.capitalize()} Level)',
                    filename=f'cities_bubble_map_{admin_level}',
                    size_field='population',
                    color_field='growth',
                    label_field='name',
                    size_scale=300,
                    cmap='plasma',
                    add_labels=True,
                    admin_level=admin_level
                )
                
                visualization_paths[f'cities_bubble_map_{admin_level}'] = bubble_map_path
    except Exception as e:
        logger.error(f"Failed to create cities bubble maps: {e}")
    
    # 3. Create bivariate map showing relationship between economic and environmental factors
    try:
        # Create sample data for economic development and environmental impact
        divisions = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh']
        
        # Economic development index (higher is more developed)
        econ_development = {
            'Dhaka': 0.85,
            'Chittagong': 0.72,
            'Rajshahi': 0.56,
            'Khulna': 0.61,
            'Barisal': 0.48,
            'Sylhet': 0.67,
            'Rangpur': 0.52,
            'Mymensingh': 0.45
        }
        
        # Environmental vulnerability index (higher is more vulnerable)
        env_vulnerability = {
            'Dhaka': 0.78,
            'Chittagong': 0.65,
            'Rajshahi': 0.42,
            'Khulna': 0.82,
            'Barisal': 0.76,
            'Sylhet': 0.57,
            'Rangpur': 0.48,
            'Mymensingh': 0.53
        }
        
        # Create bivariate map with division admin level
        bivariate_map_path = geo_vis.create_bivariate_map(
            var1_data=econ_development,
            var2_data=env_vulnerability,
            title='Economic Development vs Environmental Vulnerability',
            filename='econ_env_bivariate_division',
            var1_name='Economic Development',
            var2_name='Environmental Vulnerability',
            admin_level='division',
            n_classes=3
        )
        
        visualization_paths['bivariate_map_division'] = bivariate_map_path
    except Exception as e:
        logger.error(f"Failed to create bivariate map: {e}")
    
    # 4. Create a heat map of simulated climate impact intensity
    try:
        # Create a grid covering Bangladesh
        grid_size = 50
        lat_min, lat_max = 20.5, 26.7
        lon_min, lon_max = 88.0, 92.7
        
        # Generate sample heat map data for climate impact intensity
        # Use environmental model data if available to influence the pattern
        np.random.seed(42)  # For reproducibility
        
        # Base intensity grid
        intensity_grid = np.zeros((grid_size, grid_size))
        
        # Create a general south-to-north pattern with coastal areas more vulnerable
        for i in range(grid_size):
            for j in range(grid_size):
                # Normalized coordinates
                y_norm = i / grid_size  # 0 at south, 1 at north
                x_norm = j / grid_size  # 0 at west, 1 at east
                
                # Coast proximity (simplified)
                coast_proximity = min(y_norm, 1 - y_norm) + min(x_norm, 1 - x_norm)
                
                # Heat decreases from south to north and is higher near coasts
                intensity_grid[i, j] = (1 - y_norm) * 0.7 + (1 - coast_proximity) * 0.3
        
        # Add some random variation
        intensity_grid += np.random.normal(0, 0.15, (grid_size, grid_size))
        
        # Clip values
        intensity_grid = np.clip(intensity_grid, 0, 1)
        
        # Create smoother gradient
        from scipy.ndimage import gaussian_filter
        intensity_grid = gaussian_filter(intensity_grid, sigma=1.0)
        
        # Create heat maps with different admin level boundaries
        for admin_level in ['country', 'division', 'district', 'upazila']:
            if admin_level in geo_vis.bd_shapefiles:
                heat_map_path = geo_vis.create_heat_map(
                    grid_data=intensity_grid,
                    bounds=(lon_min, lat_min, lon_max, lat_max),
                    title=f'Climate Impact Intensity ({admin_level.capitalize()} Boundaries)',
                    filename=f'climate_impact_heat_map_{admin_level}',
                    cmap='YlOrRd',
                    alpha=0.7,
                    overlay_shapefile=True,
                    admin_level=admin_level
                )
                
                visualization_paths[f'climate_heat_map_{admin_level}'] = heat_map_path
    except Exception as e:
        logger.error(f"Failed to create climate heat maps: {e}")
    
    # 5. Create time series map animation of GDP growth by division
    try:
        # Generate time series data for GDP by division
        divisions = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh']
        
        # Get years from simulation
        years = integrated_data['time_steps'][-10:]  # Last 10 years
        
        # Create division GDP time series
        division_gdp_time_series = {}
        
        # Base division GDP distribution
        base_division_distribution = {
            'Dhaka': 0.32,
            'Chittagong': 0.20,
            'Rajshahi': 0.12,
            'Khulna': 0.10,
            'Barisal': 0.06,
            'Sylhet': 0.06,
            'Rangpur': 0.09,
            'Mymensingh': 0.05
        }
        
        # Get GDP time series from simulation if available
        if ('economic' in integrated_data['integrated_data'] and 
            'total_gdp' in integrated_data['integrated_data']['economic']):
            
            # Get GDP values for the last 10 years
            gdp_values = integrated_data['integrated_data']['economic']['total_gdp']['values'][-10:]
            
            # Ensure we have enough values
            if len(gdp_values) == len(years):
                # Calculate division GDP for each year
                for division in divisions:
                    division_gdp_time_series[division] = {}
                    
                    # Initialize with base distribution
                    base_share = base_division_distribution[division]
                    
                    # Add some variation over time (growth isn't uniform)
                    for i, year in enumerate(years):
                        # Adjust share with small random variation
                        adjusted_share = base_share * (1 + np.random.normal(0, 0.02))
                        division_gdp_time_series[division][year] = gdp_values[i] * adjusted_share
            else:
                # Create fictional data if simulation data doesn't have enough years
                base_gdp = 300  # Billion BDT
                growth_rate = 0.05  # 5% annual growth
                
                for division in divisions:
                    division_gdp_time_series[division] = {}
                    base_share = base_division_distribution[division]
                    
                    for i, year in enumerate(years):
                        # Compound growth with some variation by division
                        division_growth = growth_rate * (1 + (np.random.random() - 0.5) * 0.4)
                        division_gdp = base_gdp * base_share * (1 + division_growth) ** i
                        division_gdp_time_series[division][year] = division_gdp
        else:
            # Create fictional data if no simulation data
            base_gdp = 300  # Billion BDT
            growth_rate = 0.05  # 5% annual growth
            
            for division in divisions:
                division_gdp_time_series[division] = {}
                base_share = base_division_distribution[division]
                
                for i, year in enumerate(years):
                    # Compound growth with some variation by division
                    division_growth = growth_rate * (1 + (np.random.random() - 0.5) * 0.4)
                    division_gdp = base_gdp * base_share * (1 + division_growth) ** i
                    division_gdp_time_series[division][year] = division_gdp
        
        # Create time series animation with division admin level
        animation_path = geo_vis.create_time_series_map_animation(
            time_series_data=division_gdp_time_series,
            title='GDP by Division Over Time',
            filename='gdp_by_division_animation',
            admin_level='division',  # Specifically use division level
            cmap='Greens',
            fps=1
        )
        
        visualization_paths['gdp_animation'] = animation_path
    except Exception as e:
        logger.error(f"Failed to create time series animation: {e}")
    
    # 6. Create district-level visualizations if data available
    try:
        if 'district' in geo_vis.bd_shapefiles:
            # Generate sample district-level resilience data
            # This is fictional data for demonstration
            districts = [
                'Dhaka', 'Gazipur', 'Narayanganj', 'Chittagong', 'Cox\'s Bazar', 
                'Khulna', 'Bagerhat', 'Satkhira', 'Rajshahi', 'Rangpur', 
                'Sylhet', 'Barisal', 'Mymensingh', 'Cumilla', 'Noakhali'
            ]
            
            # Generate resilience scores (0-1)
            np.random.seed(123)
            district_resilience = {district: np.random.uniform(0.3, 0.9) for district in districts}
            
            # Create district-level choropleth
            district_map_path = geo_vis.create_choropleth(
                data=district_resilience,
                title='Climate Resilience Score by District',
                filename='district_resilience',
                cmap='RdYlGn',
                admin_level='district',
                legend_title='Resilience Score'
            )
            
            visualization_paths['district_resilience'] = district_map_path
    except Exception as e:
        logger.error(f"Failed to create district-level visualization: {e}")
    
    # 7. Generate upazila-level economic activity visualization
    try:
        if 'upazila' in geo_vis.bd_shapefiles:
            # Get a sample of upazilas (this would normally come from actual data)
            # For demonstration, we'll create fictional data
            upazilas = [
                'Savar', 'Dhamrai', 'Keraniganj', 'Dohar', 'Nawabganj',
                'Gazipur Sadar', 'Kaliakair', 'Kaliganj', 'Kapasia', 'Tongi',
                'Narayanganj Sadar', 'Araihazar', 'Bandar', 'Rupganj', 'Sonargaon'
            ]
            
            # Generate economic activity index (0-100)
            np.random.seed(456)
            upazila_economy = {upazila: np.random.uniform(20, 80) for upazila in upazilas}
            
            # Create upazila-level choropleth
            upazila_map_path = geo_vis.create_choropleth(
                data=upazila_economy,
                title='Economic Activity by Upazila',
                filename='upazila_economic_activity',
                cmap='Blues',
                admin_level='upazila',
                legend_title='Economic Activity Index'
            )
            
            visualization_paths['upazila_economy'] = upazila_map_path
            
            # Create bivariate map for upazilas with additional environmental data
            # Generate environmental health scores
            upazila_env_health = {upazila: np.random.uniform(0.2, 0.8) for upazila in upazilas}
            
            bivariate_upazila_path = geo_vis.create_bivariate_map(
                var1_data={u: v/100 for u, v in upazila_economy.items()},  # Normalize to 0-1
                var2_data=upazila_env_health,
                title='Economic Activity vs Environmental Health (Upazila Level)',
                filename='upazila_econ_env_bivariate',
                var1_name='Economic Activity',
                var2_name='Environmental Health',
                admin_level='upazila',
                n_classes=3
            )
            
            visualization_paths['upazila_bivariate'] = bivariate_upazila_path
    except Exception as e:
        logger.error(f"Failed to create upazila-level visualization: {e}")
    
    return visualization_paths

if __name__ == "__main__":
    main()
