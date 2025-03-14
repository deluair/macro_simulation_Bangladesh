#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Result processor for the Bangladesh simulation model.
This module handles saving, processing, and analyzing simulation results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class ResultProcessor:
    """
    Utility for processing and saving simulation results.
    Handles converting results to various formats, generating summary statistics,
    and preparing data for visualization.
    """
    
    def __init__(self, output_dir='output', simulation_id=None):
        """
        Initialize the result processor with output directory.
        
        Args:
            output_dir (str): Path to the output directory
            simulation_id (str, optional): Unique identifier for this simulation run
        """
        self.output_dir = Path(output_dir)
        
        # Generate simulation ID if not provided
        if simulation_id is None:
            simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.simulation_id = simulation_id
        
        # Create simulation-specific output directory
        self.sim_output_dir = self.output_dir / self.simulation_id
        if not self.sim_output_dir.exists():
            os.makedirs(self.sim_output_dir, exist_ok=True)
            print(f"Created simulation output directory: {self.sim_output_dir}")
        
        # Initialize results tracking
        self.results_processed = False
        self.results_summary = {}
        self.time_series_data = {}
    
    def process_simulation_results(self, simulation):
        """
        Process results from a simulation object.
        
        Args:
            simulation: BangladeshSimulation object with results
            
        Returns:
            dict: Summary of processed results
        """
        # Extract results from simulation object
        results = {}
        
        # Generate time points based on simulation's time step
        # If time_points is not directly available, create it from time_step
        if hasattr(simulation, 'time_points'):
            time_points = simulation.time_points
        elif hasattr(simulation, 'time_step'):
            # Create a range from 0 to the current time_step
            time_points = list(range(simulation.time_step + 1))
        else:
            # Use the current_step attribute as a fallback
            time_points = list(range(simulation.current_step))
        
        # Extract model history data from each model in the simulation
        if hasattr(simulation, 'models'):
            for model_name, model in simulation.models.items():
                # Get the full history of each model
                model_history = model.get_history()
                
                # If history is empty, just use the current state
                if not model_history:
                    results[model_name] = [model.state]
                else:
                    results[model_name] = model_history
        else:
            # Use the history attribute directly if available
            if hasattr(simulation, 'history'):
                results = simulation.history
            else:
                # Create a minimal result structure
                results = {
                    'simulation': {
                        'steps': simulation.current_step,
                        'status': 'completed'
                    }
                }
        
        # Process these results
        return self.process_results(results, time_points)
    
    def process_results(self, results, time_points):
        """
        Process, analyze, and save simulation results.
        
        Args:
            results (dict): Dictionary of simulation results by component
            time_points (list/array): Time points for the simulation results
            
        Returns:
            dict: Summary of processed results
        """
        print(f"Processing simulation results for ID: {self.simulation_id}")
        
        # Save raw results
        self._save_raw_results(results, time_points)
        
        # Convert to time series format for analysis
        self.time_series_data = self._convert_to_time_series(results, time_points)
        
        # Generate summary statistics
        self.results_summary = self._generate_summary_statistics()
        
        # Save processed results
        self._save_processed_results()
        
        # Flag that results have been processed
        self.results_processed = True
        
        print(f"Results processed and saved to: {self.sim_output_dir}")
        return self.results_summary
    
    def _save_raw_results(self, results, time_points):
        """
        Save raw simulation results to files.
        
        Args:
            results (dict): Dictionary of simulation results by component
            time_points (list/array): Time points for the simulation results
        """
        # Save as JSON
        raw_results = {
            'simulation_id': self.simulation_id,
            'time_points': time_points.tolist() if isinstance(time_points, np.ndarray) else time_points,
            'components': {}
        }
        
        # Add results for each component
        for component, component_results in results.items():
            raw_results['components'][component] = [result for result in component_results]
        
        # Save to file
        with open(self.sim_output_dir / 'raw_results.json', 'w') as f:
            json.dump(raw_results, f, indent=2)
    
    def _convert_to_time_series(self, results, time_points):
        """
        Convert results to time series format for analysis.
        
        Args:
            results (dict): Dictionary of simulation results by component
            time_points (list/array): Time points for the simulation results
            
        Returns:
            dict: Time series data by component and variable
        """
        time_series_data = {}
        
        # Process each component's results
        for component, component_results in results.items():
            # Initialize component dictionary if not exists
            if component not in time_series_data:
                time_series_data[component] = {}
            
            # Make sure we have the right number of time points
            # If component_results has fewer items than time_points, 
            # we'll use only the available time points
            actual_time_points = time_points[:len(component_results)]
            
            # Extract all variables from the first result to setup dataframes
            if component_results:
                for variable, value in component_results[0].items():
                    # Skip 'timestamp' variable from history
                    if variable == 'timestamp':
                        continue
                        
                    # Initialize with NaN values
                    time_series_data[component][variable] = pd.Series(
                        index=time_points,
                        dtype=float if isinstance(value, (int, float)) else object
                    )
            
            # Fill in values for each time point
            for t, result in zip(actual_time_points, component_results):
                for variable, value in result.items():
                    # Skip 'timestamp' variable from history
                    if variable == 'timestamp':
                        continue
                        
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        time_series_data[component][variable][t] = value
                    else:
                        # For complex objects (dict, list), convert to JSON string
                        time_series_data[component][variable][t] = json.dumps(value)
        
        return time_series_data
    
    def _generate_summary_statistics(self):
        """
        Generate summary statistics for time series data.
        
        Returns:
            dict: Summary statistics by component and variable
        """
        summary = {}
        
        # Process each component's time series data
        for component, variables in self.time_series_data.items():
            summary[component] = {}
            
            for variable, series in variables.items():
                # Skip non-numeric variables
                if not pd.api.types.is_numeric_dtype(series):
                    continue
                
                # Remove NaN values for calculations, but keep track of original length
                original_len = len(series)
                series_clean = series.dropna()
                
                # Handle completely empty series
                if series_clean.empty:
                    summary[component][variable] = {
                        'mean': None,
                        'median': None,
                        'min': None,
                        'max': None,
                        'std': None,
                        'start': None,
                        'end': None,
                        'end_value': None,
                        'change': None,
                        'percent_change': None
                    }
                    continue
                
                # Get first and last valid values
                if not series.empty:
                    first_valid = series.first_valid_index()
                    last_valid = series.last_valid_index()
                    start_value = series.loc[first_valid] if first_valid is not None else None
                    end_value = series.loc[last_valid] if last_valid is not None else None
                else:
                    start_value = None
                    end_value = None
                
                # Calculate change and percent change safely
                try:
                    if start_value is not None and end_value is not None:
                        change = end_value - start_value
                    else:
                        change = None
                except Exception:
                    change = None
                    
                try:
                    if start_value is not None and end_value is not None and start_value != 0:
                        percent_change = (end_value / start_value - 1) * 100
                        # Check for infinity or very large values
                        if not np.isfinite(percent_change) or abs(percent_change) > 1e10:
                            percent_change = None
                    else:
                        percent_change = None
                except Exception:
                    percent_change = None
                
                # Calculate other statistics safely
                try:
                    mean_value = series_clean.mean()
                    if not np.isfinite(mean_value):
                        mean_value = None
                except Exception:
                    mean_value = None
                    
                try:
                    median_value = series_clean.median()
                    if not np.isfinite(median_value):
                        median_value = None
                except Exception:
                    median_value = None
                    
                try:
                    min_value = series_clean.min()
                    if not np.isfinite(min_value):
                        min_value = None
                except Exception:
                    min_value = None
                    
                try:
                    max_value = series_clean.max()
                    if not np.isfinite(max_value):
                        max_value = None
                except Exception:
                    max_value = None
                    
                try:
                    std_value = series_clean.std()
                    if not np.isfinite(std_value):
                        std_value = None
                except Exception:
                    std_value = None
                
                # Build the summary dict
                summary[component][variable] = {
                    'mean': mean_value,
                    'median': median_value,
                    'min': min_value,
                    'max': max_value,
                    'std': std_value,
                    'start': start_value,
                    'end': series.iloc[-1] if not series.empty else None,
                    'end_value': end_value,  # This is important for the HTML report
                    'change': change,
                    'percent_change': percent_change
                }
        
        return summary
    
    def get_results_dataframe(self):
        """
        Convert time series data to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame containing all time series data
        """
        if not self.time_series_data:
            raise ValueError("No time series data available. Process results first.")
        
        # Create a list to hold all series with multi-level column names
        all_series = []
        column_names = []
        
        # Process each component's time series data
        for component, variables in self.time_series_data.items():
            for variable, series in variables.items():
                all_series.append(series)
                column_names.append((component, variable))
        
        # Create DataFrame with multi-level columns
        if all_series:
            df = pd.concat(all_series, axis=1)
            df.columns = pd.MultiIndex.from_tuples(column_names, names=['Component', 'Variable'])
            return df
        else:
            return pd.DataFrame()
    
    def get_summary_statistics(self):
        """
        Get summary statistics for the processed results.
        
        Returns:
            dict: Summary statistics by component and variable
        """
        if not self.results_processed:
            raise ValueError("Results have not been processed yet.")
        
        return self.results_summary
    
    def _save_processed_results(self):
        """Save processed time series data and summary statistics to files."""
        
        # Create component-specific directories
        for component in self.time_series_data.keys():
            component_dir = self.sim_output_dir / component
            if not component_dir.exists():
                os.makedirs(component_dir, exist_ok=True)
        
        # Save time series data as CSV for each component
        for component, variables in self.time_series_data.items():
            # Convert variables to DataFrame
            df = pd.DataFrame(variables)
            
            # Save to CSV
            df.to_csv(self.sim_output_dir / component / 'time_series.csv')
            
            # For key numeric variables, also save individual CSV files
            for variable, series in variables.items():
                if pd.api.types.is_numeric_dtype(series):
                    series.to_csv(self.sim_output_dir / component / f'{variable}.csv', 
                                header=[variable])
        
        # Save summary statistics as JSON
        with open(self.sim_output_dir / 'summary_statistics.json', 'w') as f:
            # Use custom JSON encoder to handle non-serializable values
            def json_serializer(obj):
                """Handle non-serializable values for JSON"""
                if isinstance(obj, (pd.Timestamp, np.datetime64)):
                    return obj.isoformat()
                elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return float(obj)
                elif pd.isna(obj):
                    return None
                return str(obj)
                
            # Clean results summary to replace NaN/inf with None
            clean_summary = {}
            for component, variables in self.results_summary.items():
                clean_summary[component] = {}
                for var, stats in variables.items():
                    clean_summary[component][var] = {}
                    for stat, value in stats.items():
                        if isinstance(value, (float, np.float64, np.float32)) and (np.isnan(value) or np.isinf(value)):
                            clean_summary[component][var][stat] = None
                        else:
                            clean_summary[component][var][stat] = value
                
            json.dump(clean_summary, f, indent=2, default=json_serializer)
    
    def generate_key_metrics_report(self, output_format='markdown'):
        """
        Generate a report of key metrics from the simulation.
        
        Args:
            output_format (str): Format for the report (markdown, html, text)
            
        Returns:
            str: Report content
        """
        if not self.results_processed:
            raise ValueError("Results must be processed before generating a report")
            
        # Start building the report
        report = []
        
        if output_format == 'markdown':
            # Markdown format
            report.append(f"# Bangladesh Simulation Results\n")
            report.append(f"**Simulation ID:** {self.simulation_id}\n")
            report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # Key metrics by component
            report.append("## Key Metrics\n")
            
            # Economic metrics
            if 'economic' in self.results_summary:
                report.append("### Economic Performance\n")
                
                econ = self.results_summary['economic']
                if 'gdp' in econ:
                    report.append(f"- **GDP:** {econ['gdp']['end_value']:.2f} billion USD")
                    report.append(f"  - Growth: {econ['gdp']['percent_change']:.2f}%\n")
                
                if 'gdp_per_capita' in econ:
                    report.append(f"- **GDP per Capita:** {econ['gdp_per_capita']['end_value']:.2f} USD")
                    report.append(f"  - Growth: {econ['gdp_per_capita']['percent_change']:.2f}%\n")
                    
                if 'unemployment_rate' in econ:
                    report.append(f"- **Unemployment Rate:** {econ['unemployment_rate']['end_value']*100:.2f}%\n")
                    
                if 'inflation_rate' in econ:
                    report.append(f"- **Inflation Rate:** {econ['inflation_rate']['end_value']*100:.2f}%\n")
            
            # Demographic metrics
            if 'demographic' in self.results_summary:
                report.append("### Demographic Trends\n")
                
                demo = self.results_summary['demographic']
                if 'total_population' in demo:
                    report.append(f"- **Total Population:** {demo['total_population']['end_value']/1e6:.2f} million")
                    report.append(f"  - Growth: {demo['total_population']['percent_change']:.2f}%\n")
                
                if 'urbanization_rate' in demo:
                    report.append(f"- **Urbanization Rate:** {demo['urbanization_rate']['end_value']*100:.2f}%\n")
                    
                if 'literacy_rate' in demo:
                    report.append(f"- **Literacy Rate:** {demo['literacy_rate']['end_value']*100:.2f}%\n")
            
            # Environmental metrics
            if 'environmental' in self.results_summary:
                report.append("### Environmental Conditions\n")
                
                env = self.results_summary['environmental']
                if 'forest_cover' in env:
                    report.append(f"- **Forest Cover:** {env['forest_cover']['end_value']*100:.2f}% of land area")
                    report.append(f"  - Change: {env['forest_cover']['percent_change']:.2f}%\n")
                
                if 'air_quality_index' in env:
                    report.append(f"- **Air Quality Index:** {env['air_quality_index']['end_value']:.2f}\n")
                    
                if 'annual_emissions' in env:
                    report.append(f"- **Annual Emissions:** {env['annual_emissions']['end_value']:.2f} MT CO2e\n")
            
            # Infrastructure metrics
            if 'infrastructure' in self.results_summary:
                report.append("### Infrastructure Development\n")
                
                infra = self.results_summary['infrastructure']
                if 'electricity_coverage' in infra:
                    report.append(f"- **Electricity Coverage:** {infra['electricity_coverage']['end_value']*100:.2f}%\n")
                
                if 'renewable_energy_share' in infra:
                    report.append(f"- **Renewable Energy Share:** {infra['renewable_energy_share']['end_value']*100:.2f}%\n")
                    
                if 'road_density' in infra:
                    report.append(f"- **Road Density:** {infra['road_density']['end_value']:.2f} km/sq.km\n")
                    
                if 'internet_coverage' in infra:
                    report.append(f"- **Internet Coverage:** {infra['internet_coverage']['end_value']*100:.2f}%\n")
            
            # Governance metrics
            if 'governance' in self.results_summary:
                report.append("### Governance Indicators\n")
                
                gov = self.results_summary['governance']
                if 'institutional_effectiveness' in gov:
                    report.append(f"- **Institutional Effectiveness:** {gov['institutional_effectiveness']['end_value']:.2f}/1.0\n")
                
                if 'corruption_level' in gov:
                    report.append(f"- **Corruption Level:** {gov['corruption_level']['end_value']:.2f}/1.0\n")
        
        elif output_format == 'html':
            # HTML format implementation would go here
            pass
            
        elif output_format == 'text':
            # Plain text format implementation would go here
            pass
        
        # Join all parts of the report
        return '\n'.join(report)
    
    def create_time_series_plots(self, key_variables=None, save_format='png'):
        """
        Create time series plots for key variables.
        
        Args:
            key_variables (dict, optional): Dict mapping components to list of variables to plot
            save_format (str): Format to save plots (png, jpg, pdf, svg)
            
        Returns:
            list: Paths to saved plot files
        """
        if not self.results_processed:
            raise ValueError("Results must be processed before creating plots")
        
        # If no key variables specified, use defaults
        if key_variables is None:
            key_variables = {
                'economic': ['gdp', 'gdp_growth', 'inflation_rate', 'unemployment_rate'],
                'demographic': ['total_population', 'urbanization_rate', 'fertility_rate'],
                'environmental': ['forest_cover', 'annual_emissions', 'temperature_anomaly'],
                'infrastructure': ['electricity_coverage', 'renewable_energy_share', 'road_density', 'internet_coverage'],
                'governance': ['institutional_effectiveness', 'corruption_level', 'political_stability']
            }
        
        # Create plots directory
        plots_dir = self.sim_output_dir / 'plots'
        if not plots_dir.exists():
            os.makedirs(plots_dir, exist_ok=True)
            
        saved_plots = []
        
        # Set plotting style for better visual appeal
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create plots for each component and its key variables
        for component, variables in key_variables.items():
            if component not in self.time_series_data:
                continue
                
            # Create component-specific plot directory
            component_plots_dir = plots_dir / component
            if not component_plots_dir.exists():
                os.makedirs(component_plots_dir, exist_ok=True)
            
            # Filter to only include available numeric variables
            valid_variables = []
            for var in variables:
                if var in self.time_series_data[component]:
                    series = self.time_series_data[component][var]
                    if pd.api.types.is_numeric_dtype(series):
                        valid_variables.append(var)
            
            # Plot each variable
            for var in valid_variables:
                plt.figure(figsize=(10, 6))
                series = self.time_series_data[component][var]
                
                # Fill any NaN values to ensure continuous line
                filled_series = series.interpolate(method='linear')
                
                # Clean up the data for plotting - remove any remaining NaNs at endpoints
                if pd.isna(filled_series.iloc[0]):
                    filled_series.iloc[0] = filled_series.dropna().iloc[0]
                if pd.isna(filled_series.iloc[-1]):
                    filled_series.iloc[-1] = filled_series.dropna().iloc[-1]
                
                # Plot with improved styling
                plt.plot(filled_series.index, filled_series.values, 'b-', linewidth=3, alpha=0.7)
                
                # Add light-colored area below curve
                plt.fill_between(filled_series.index, 0, filled_series.values, alpha=0.1, color='blue')
                
                # Add points at data locations
                plt.scatter(filled_series.index, filled_series.values, color='darkblue', s=50, alpha=0.7)
                
                # Format the plot
                title = var.replace('_', ' ').title()
                plt.title(f"{component.capitalize()}: {title}", fontsize=16, pad=20)
                plt.xlabel('Year', fontsize=12, labelpad=10)
                
                # Add appropriate y-axis label
                if 'rate' in var or var in ['urbanization_rate', 'literacy_rate']:
                    plt.ylabel('Percentage (%)', fontsize=12, labelpad=10)
                    # Format y-ticks as percentages for rate variables
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
                elif var == 'gdp':
                    plt.ylabel('Billion USD', fontsize=12, labelpad=10)
                elif var == 'gdp_per_capita':
                    plt.ylabel('USD', fontsize=12, labelpad=10)
                elif var == 'total_population':
                    plt.ylabel('Population (millions)', fontsize=12, labelpad=10)
                    # Format y-ticks in millions
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000000:.1f}'))
                else:
                    plt.ylabel('Value', fontsize=12, labelpad=10)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Add annotations for start and end points
                start_value = filled_series.iloc[0]
                end_value = filled_series.iloc[-1]
                start_year = filled_series.index[0]
                end_year = filled_series.index[-1]
                
                # Format annotation text based on variable type
                if 'rate' in var or var in ['urbanization_rate', 'literacy_rate']:
                    start_text = f'{start_value*100:.1f}%'
                    end_text = f'{end_value*100:.1f}%'
                elif var == 'gdp':
                    start_text = f'${start_value:.1f}B'
                    end_text = f'${end_value:.1f}B'
                elif var == 'gdp_per_capita':
                    start_text = f'${start_value:.0f}'
                    end_text = f'${end_value:.0f}'
                elif var == 'total_population':
                    start_text = f'{start_value/1000000:.1f}M'
                    end_text = f'{end_value/1000000:.1f}M'
                else:
                    start_text = f'{start_value:.2f}'
                    end_text = f'{end_value:.2f}'
                
                # Add annotations with arrows
                plt.annotate(start_text, 
                            xy=(start_year, start_value),
                            xytext=(start_year - 1, start_value * 1.1),
                            arrowprops=dict(arrowstyle="->", color='gray'),
                            fontsize=10)
                            
                plt.annotate(end_text,
                            xy=(end_year, end_value),
                            xytext=(end_year + 1, end_value * 1.1),
                            arrowprops=dict(arrowstyle="->", color='gray'),
                            fontsize=10)
                
                # Save the plot
                plot_path = component_plots_dir / f"{var}.{save_format}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Add to list of saved plots
                saved_plots.append(str(plot_path))
                
            # Create a composite plot for the component
            if len(valid_variables) > 1:
                plt.figure(figsize=(12, 8))
                
                # Create a color palette
                colors = plt.cm.tab10(np.linspace(0, 1, len(valid_variables)))
                
                for i, var in enumerate(valid_variables):
                    series = self.time_series_data[component][var]
                    filled_series = series.interpolate(method='linear')
                    
                    # Clean up data for plotting
                    if pd.isna(filled_series.iloc[0]):
                        filled_series.iloc[0] = filled_series.dropna().iloc[0]
                    if pd.isna(filled_series.iloc[-1]):
                        filled_series.iloc[-1] = filled_series.dropna().iloc[-1]
                    
                    # Normalize values to 0-1 range for comparison
                    min_val = filled_series.min()
                    max_val = filled_series.max()
                    range_val = max_val - min_val
                    
                    if range_val > 0:
                        normalized = (filled_series - min_val) / range_val
                        plt.plot(filled_series.index, normalized, linewidth=3, 
                               label=var.replace('_', ' ').title(), alpha=0.7, color=colors[i])
                        
                        # Add points at data locations
                        plt.scatter(filled_series.index, normalized, s=50, alpha=0.7, color=colors[i])
                
                # Format the composite plot
                plt.title(f"{component.capitalize()}: Key Indicators (Normalized)", fontsize=16, pad=20)
                plt.xlabel('Year', fontsize=12, labelpad=10)
                plt.ylabel('Normalized Value (0-1 scale)', fontsize=12, labelpad=10)
                plt.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the composite plot
                composite_path = plots_dir / f"{component}_composite.{save_format}"
                plt.savefig(composite_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Add to list of saved plots
                saved_plots.append(str(composite_path))
        
        return saved_plots
