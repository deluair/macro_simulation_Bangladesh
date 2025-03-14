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
    
    # Generate HTML report
    logger.info("Generating HTML report")
    report_generator = HTMLReportGenerator(output_dir, simulation_id)
    report_path = report_generator.generate_report("Bangladesh Development Simulation Results")
    
    logger.info(f"Report generated: {report_path}")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("Simulation completed successfully")
    
    return {
        'simulation': simulation,
        'results_df': results_df,
        'summary_stats': summary_stats,
        'report_path': report_path,
        'results_dir': results_dir
    }

def visualize_results(results_df, output_dir, visualizer):
    """Generate visualizations for standard simulation results."""
    
    # Create time series data format
    time_series_data = {}
    
    # Handle multi-index DataFrame from ResultProcessor
    if isinstance(results_df.columns, pd.MultiIndex):
        # Extract data from multi-index DataFrame
        for component, variable in results_df.columns:
            var_name = f"{component}_{variable}"
            
            if var_name not in time_series_data:
                time_series_data[var_name] = {'years': [], 'values': []}
            
            # Get values for this variable
            series = results_df[(component, variable)]
            
            # Add to time series data
            for year, value in series.items():
                if not pd.isna(value):
                    time_series_data[var_name]['years'].append(year)
                    time_series_data[var_name]['values'].append(value)
    else:
        # Handle old format (if still needed)
        for _, row in results_df.iterrows():
            var = row['variable']
            year = row['year']
            value = row['value']
            
            if var not in time_series_data:
                time_series_data[var] = {'years': [], 'values': []}
            
            time_series_data[var]['years'].append(year)
            time_series_data[var]['values'].append(value)
    
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

if __name__ == "__main__":
    main()
