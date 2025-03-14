"""
Time series plotting module for the Bangladesh Development Simulation Model.
Contains tools for creating time series visualizations of simulation results over time.
"""

import os
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesPlotter:
    """Class for creating time series visualizations."""
    
    def __init__(self, output_dir: str = 'results/time_series'):
        """
        Initialize the time series plotter.
        
        Args:
            output_dir: Directory to save time series visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the default style for plots
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})
    
    def plot_time_series(self, 
                       data: Union[Dict[str, List[float]], pd.DataFrame], 
                       time_points: List[int],
                       title: str,
                       filename: str,
                       y_label: str = 'Value',
                       x_label: str = 'Year',
                       series_labels: Optional[List[str]] = None,
                       figsize: tuple = (12, 8)) -> str:
        """
        Create a time series plot of multiple variables over time.
        
        Args:
            data: Dictionary mapping variable names to lists of values, or DataFrame
            time_points: List of time points (e.g., years)
            title: Chart title
            filename: Filename to save the visualization
            y_label: Label for the y-axis
            x_label: Label for the x-axis
            series_labels: Optional list of labels for the series
            figsize: Figure size tuple
            
        Returns:
            Path to the saved visualization file
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data, index=time_points)
        else:
            df = data.copy()
            if not df.index.equals(pd.Index(time_points)):
                df.index = time_points
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each series
        for col in df.columns:
            ax.plot(df.index, df[col], marker='o', linewidth=2, markersize=5, 
                    label=col, alpha=0.8)
        
        # Add labels and legend
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Adjust tick labels
        plt.xticks(time_points[::5] if len(time_points) > 10 else time_points)
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_comparative_scenarios(self, 
                                scenario_data: Dict[str, Dict[str, List[float]]],
                                time_points: List[int],
                                title: str,
                                filename: str,
                                variable: str,
                                y_label: str = 'Value',
                                x_label: str = 'Year',
                                figsize: tuple = (12, 8)) -> str:
        """
        Create a comparative plot of a single variable across multiple scenarios.
        
        Args:
            scenario_data: Dictionary mapping scenario names to dictionaries mapping variables to lists of values
            time_points: List of time points (e.g., years)
            title: Chart title
            filename: Filename to save the visualization
            variable: The variable to plot across scenarios
            y_label: Label for the y-axis
            x_label: Label for the x-axis
            figsize: Figure size tuple
            
        Returns:
            Path to the saved visualization file
        """
        # Extract the variable of interest from each scenario
        data = {}
        for scenario, vars_dict in scenario_data.items():
            if variable in vars_dict:
                data[scenario] = vars_dict[variable]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each scenario
        for scenario, values in data.items():
            ax.plot(time_points, values, marker='o', linewidth=2, markersize=5, 
                   label=scenario, alpha=0.8)
        
        # Add labels and legend
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{title}: {variable}", fontsize=14)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Adjust tick labels
        plt.xticks(time_points[::5] if len(time_points) > 10 else time_points)
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, f"{filename}_{variable}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_heatmap(self,
                  data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
                  title: str,
                  filename: str,
                  cmap: str = 'viridis',
                  annot: bool = True,
                  figsize: tuple = (12, 10)) -> str:
        """
        Create a heatmap visualization of a 2D array of values.
        
        Args:
            data: Dictionary mapping row names to dictionaries mapping column names to values, or DataFrame
            title: Chart title
            filename: Filename to save the visualization
            cmap: Colormap name
            annot: Whether to annotate the heatmap with values
            figsize: Figure size tuple
            
        Returns:
            Path to the saved visualization file
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data).T
        else:
            df = data.copy()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(df, annot=annot, fmt='.2f', cmap=cmap, ax=ax, linewidths=0.5)
        
        # Add title
        plt.title(title, fontsize=14)
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
