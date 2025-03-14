#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HTML Report Generator for Bangladesh Simulation.
This script generates a sample HTML report using synthetic data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import random

# Import utilities
from utils.html_report_generator import HTMLReportGenerator

def create_simulation_directory():
    """Create a simulation directory with all necessary subdirectories."""
    # Create the main output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Create a simulation ID based on timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    simulation_id = f"sim_{timestamp}"
    sim_dir = output_dir / simulation_id
    sim_dir.mkdir(exist_ok=True)
    
    # Create required subdirectories
    plots_dir = sim_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Create section subdirectories
    for section in ['economic', 'demographic', 'environmental', 'infrastructure', 'governance']:
        section_dir = plots_dir / section
        section_dir.mkdir(exist_ok=True)
    
    # Create HTML reports directory
    html_reports_dir = output_dir / 'html_reports'
    html_reports_dir.mkdir(exist_ok=True)
    
    return output_dir, simulation_id, sim_dir

def generate_synthetic_data(plots_dir):
    """Generate synthetic credible data for the simulation."""
    # Define reasonable ranges for key metrics
    reasonable_ranges = {
        'economic': {
            'gdp': (100, 500),  # Billions USD
            'gdp_growth': (0.025, 0.07),  # 2.5-7%
            'gdp_per_capita': (1800, 7000),  # USD
            'inflation_rate': (0.02, 0.06),  # 2-6%
            'unemployment_rate': (0.035, 0.09)  # 3.5-9%
        },
        'demographic': {
            'total_population': (165000000, 190000000),  # Bangladesh population range
            'urbanization_rate': (0.35, 0.6),  # 35-60%
            'fertility_rate': (1.9, 2.3),  # children per woman
            'life_expectancy': (73, 82),  # years
            'literacy_rate': (0.75, 0.92)  # 75-92%
        },
        'environmental': {
            'forest_cover': (0.08, 0.18),  # 8-18% of land
            'temperature_anomaly': (0.8, 2.5),  # degrees C
            'air_quality_index': (50, 120),  # AQI
            'water_quality_index': (55, 80),  # WQI
            'extreme_event_impact': (0.2, 0.6)  # Impact index 0-1
        },
        'infrastructure': {
            'electricity_coverage': (0.75, 0.95),  # 75-95%
            'water_supply_coverage': (0.7, 0.92),  # 70-92%
            'internet_coverage': (0.45, 0.85),  # 45-85%
            'renewable_energy_share': (0.08, 0.35),  # 8-35%
            'infrastructure_quality_index': (0.45, 0.75)  # 0-1 scale
        },
        'governance': {
            'governance_index': (0.4, 0.65),  # 0-1 scale
            'institutional_effectiveness': (0.35, 0.7),  # 0-1 scale
            'corruption_index': (0.3, 0.65),  # 0-1 scale (higher is worse)
            'regulatory_quality': (0.4, 0.7),  # 0-1 scale
            'policy_effectiveness': (0.35, 0.68)  # 0-1 scale
        }
    }
    
    # Create the summary data
    summary_data = {}
    
    # Generate time series data (2023-2050)
    years = list(range(2023, 2051))
    n_years = len(years)
    
    # Generate a trend shape function (sigmoid, exponential, or linear)
    def trend_shape(x, trend_type='sigmoid'):
        if trend_type == 'sigmoid':
            return 1 / (1 + np.exp(-10 * (x - 0.5)))
        elif trend_type == 'exponential':
            return np.exp(3 * x) / np.exp(3)
        elif trend_type == 'exponential_decay':
            return np.exp(-3 * x) / np.exp(0)
        else:  # linear
            return x
    
    # For each component, generate credible metrics
    for component, metrics in reasonable_ranges.items():
        summary_data[component] = {}
        
        for metric, (min_val, max_val) in metrics.items():
            # Decide on trend direction and type
            if metric in ['gdp', 'gdp_per_capita', 'electricity_coverage', 
                          'water_supply_coverage', 'internet_coverage',
                          'renewable_energy_share', 'governance_index',
                          'literacy_rate', 'life_expectancy']:
                # Increasing trend
                trend_direction = 'increasing'
                trend_type = random.choice(['sigmoid', 'exponential', 'linear'])
            elif metric in ['unemployment_rate', 'inflation_rate', 'corruption_index']:
                # Decreasing trend
                trend_direction = 'decreasing'
                trend_type = random.choice(['sigmoid', 'exponential_decay', 'linear'])
            else:
                # Mixed trend
                trend_direction = random.choice(['increasing', 'decreasing'])
                trend_type = random.choice(['sigmoid', 'linear'])
            
            # Generate start and end values
            range_width = max_val - min_val
            start_value = min_val + range_width * random.uniform(0.1, 0.4)
            
            if trend_direction == 'increasing':
                end_value = start_value + range_width * random.uniform(0.2, 0.5)
                end_value = min(end_value, max_val)  # Ensure within range
            else:
                end_value = start_value - range_width * random.uniform(0.1, 0.3)
                end_value = max(end_value, min_val)  # Ensure within range
            
            # Generate time series with the selected trend
            x = np.linspace(0, 1, n_years)
            noise_level = (max_val - min_val) * 0.03  # Small noise
            noise = np.random.normal(0, noise_level, n_years)
            
            if trend_direction == 'increasing':
                trend = start_value + (end_value - start_value) * trend_shape(x, trend_type) + noise
            else:
                trend = start_value - (start_value - end_value) * trend_shape(x, trend_type) + noise
            
            # Ensure all values within bounds
            trend = np.clip(trend, min_val, max_val)
            
            # Calculate summary statistics
            change = end_value - start_value
            percent_change = (change / start_value) * 100 if start_value != 0 else 100
            
            # Create the summary for this metric
            summary_data[component][metric] = {
                'start_value': start_value,
                'end_value': end_value,
                'mean': np.mean(trend),
                'std': np.std(trend),
                'min': np.min(trend),
                'max': np.max(trend),
                'change': change,
                'percent_change': percent_change
            }
            
            # Create a time series plot
            create_time_series_plot(years, trend, component, metric, plots_dir)
        
        # Create composite plot for each component
        create_composite_plot(years, reasonable_ranges[component], summary_data[component], component, plots_dir)
    
    return summary_data

def create_time_series_plot(years, values, component, metric, plots_dir):
    """Create a time series plot for a specific metric."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    plt.figure(figsize=(10, 6))
    
    # Plot the main line
    plt.plot(years, values, 'b-', linewidth=3, alpha=0.7)
    
    # Add light-colored area below curve
    plt.fill_between(years, 0, values, alpha=0.1, color='blue')
    
    # Add points at data locations
    plt.scatter(years, values, color='darkblue', s=50, alpha=0.7)
    
    # Format the plot
    title = metric.replace('_', ' ').title()
    plt.title(f"{component.capitalize()}: {title}", fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12, labelpad=10)
    
    # Add appropriate y-axis label
    if 'rate' in metric or metric in ['urbanization_rate', 'literacy_rate']:
        plt.ylabel('Percentage (%)', fontsize=12, labelpad=10)
        # Format y-ticks as percentages for rate variables
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    elif metric == 'gdp':
        plt.ylabel('Billion USD', fontsize=12, labelpad=10)
    elif metric == 'gdp_per_capita':
        plt.ylabel('USD', fontsize=12, labelpad=10)
    elif metric == 'total_population':
        plt.ylabel('Population (millions)', fontsize=12, labelpad=10)
        # Format y-ticks in millions
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000000:.1f}'))
    else:
        plt.ylabel('Value', fontsize=12, labelpad=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add annotations for start and end points
    start_value = values[0]
    end_value = values[-1]
    start_year = years[0]
    end_year = years[-1]
    
    # Format annotation text based on variable type
    if 'rate' in metric or metric in ['urbanization_rate', 'literacy_rate']:
        start_text = f'{start_value*100:.1f}%'
        end_text = f'{end_value*100:.1f}%'
    elif metric == 'gdp':
        start_text = f'${start_value:.1f}B'
        end_text = f'${end_value:.1f}B'
    elif metric == 'gdp_per_capita':
        start_text = f'${start_value:.0f}'
        end_text = f'${end_value:.0f}'
    elif metric == 'total_population':
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
    component_dir = plots_dir / component
    plot_path = component_dir / f"{metric}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_composite_plot(years, metrics, summary_data, component, plots_dir):
    """Create a composite plot showing multiple normalized metrics for a component."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Get the metrics with valid data
    valid_metrics = list(summary_data.keys())
    
    if valid_metrics:
        plt.figure(figsize=(12, 8))
        
        # Create a color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_metrics)))
        
        # For each metric, create a normalized time series
        for i, metric in enumerate(valid_metrics):
            # Get min and max values for normalization
            min_val, max_val = metrics[metric]
            
            # Generate synthetic data for this metric
            start_value = summary_data[metric]['start_value']
            end_value = summary_data[metric]['end_value']
            
            # Create a trend
            x = np.linspace(0, 1, len(years))
            
            # Add some randomness to the curve shape
            curve_factor = 0.3 + np.random.random() * 1.4  # 0.3 to 1.7
            trend = start_value + (end_value - start_value) * (x ** curve_factor)
            
            # Add some noise
            noise_level = (max_val - min_val) * 0.02
            noise = np.random.normal(0, noise_level, len(years))
            trend += noise
            
            # Normalize the trend to 0-1 range
            normalized = (trend - min_val) / (max_val - min_val)
            
            # Plot the normalized trend
            plt.plot(years, normalized, linewidth=3, 
                   label=metric.replace('_', ' ').title(), alpha=0.7, color=colors[i])
            
            # Add points
            plt.scatter(years, normalized, s=50, alpha=0.7, color=colors[i])
        
        # Format the plot
        plt.title(f"{component.capitalize()}: Key Indicators (Normalized)", fontsize=16, pad=20)
        plt.xlabel('Year', fontsize=12, labelpad=10)
        plt.ylabel('Normalized Value (0-1 scale)', fontsize=12, labelpad=10)
        plt.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the composite plot
        composite_path = plots_dir / f"{component}_composite.png"
        plt.savefig(composite_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate the HTML report."""
    print("Generating Bangladesh Simulation HTML Report...")
    
    # Create the simulation directory
    output_dir, simulation_id, sim_dir = create_simulation_directory()
    print(f"Created simulation directory: {sim_dir}")
    
    # Set plots directory for use in generate_synthetic_data
    plots_dir = sim_dir / 'plots'
    
    # Generate synthetic data
    summary_data = generate_synthetic_data(plots_dir)
    print("Generated synthetic data")
    
    # Save summary statistics to JSON
    with open(sim_dir / 'summary_statistics.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    print("Saved summary statistics")
    
    # Generate HTML report
    report_generator = HTMLReportGenerator(output_dir, simulation_id)
    report_path = report_generator.generate_report("Bangladesh Development Simulation Results")
    print(f"Generated HTML report: {report_path}")
    
    print("Done!")

if __name__ == "__main__":
    main() 