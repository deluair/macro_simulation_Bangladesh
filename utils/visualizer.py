#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utility for the Bangladesh simulation model.
This module provides tools for generating plots, maps, and dashboards
to visualize simulation results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")


class Visualizer:
    """
    Utility for generating visualizations of simulation results.
    Provides tools for creating time series plots, comparison charts,
    heatmaps, maps, and interactive dashboards.
    """
    
    def __init__(self, output_dir='results'):
        """
        Initialize the visualizer with output directory for saving visualizations.
        
        Args:
            output_dir (str): Path to the output directory
        """
        self.output_dir = Path(output_dir)
        
        # Create visualization directory
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _setup_minimal_style(self):
        """Setup a minimal, clean plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.dpi'] = 300

    def plot_time_series(self, data, variables, title=None, ylabel=None, 
                       legend_labels=None, start_year=None, end_year=None,
                       filename=None, show_plot=False):
        """
        Create a time series plot for one or more variables.
        
        Args:
            data (pd.DataFrame): DataFrame with time series data
            variables (list): List of column names to plot
            title (str, optional): Plot title
            ylabel (str, optional): Y-axis label
            legend_labels (list, optional): Custom legend labels
            start_year (int, optional): Start year for x-axis
            end_year (int, optional): End year for x-axis
            filename (str, optional): Filename to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            plt.Figure: The generated figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each variable
        for i, var in enumerate(variables):
            if var in data.columns:
                label = legend_labels[i] if legend_labels and i < len(legend_labels) else var
                data[var].plot(ax=ax, label=label, linewidth=2)
            else:
                logger.warning(f"Variable {var} not found in data")
        
        # Set plot title and labels
        if title:
            ax.set_title(title, fontsize=14)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Year', fontsize=12)
        
        # Format x-axis as years
        if isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
        
        # Limit x-axis if specified
        if start_year and end_year:
            ax.set_xlim(pd.Timestamp(f"{start_year}"), pd.Timestamp(f"{end_year}"))
        
        # Add legend and grid
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add Bangladesh flag colors as plot theme
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
        
        # Save the plot if filename provided
        if filename:
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved time series plot to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_comparison_bar(self, data, categories, values, title=None, 
                          xlabel=None, ylabel=None, color_map='viridis',
                          filename=None, show_plot=False):
        """
        Create a bar chart comparing values across categories.
        
        Args:
            data (pd.DataFrame): DataFrame with data
            categories (str): Column name for categories
            values (str or list): Column name(s) for values to plot
            title (str, optional): Plot title
            xlabel (str, optional): X-axis label
            ylabel (str, optional): Y-axis label
            color_map (str): Matplotlib colormap name
            filename (str, optional): Filename to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            plt.Figure: The generated figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert values to list if it's a single string
        if isinstance(values, str):
            values = [values]
        
        # Create DataFrame for plotting
        plot_data = data.set_index(categories)[values].copy()
        
        # Plot the bar chart
        plot_data.plot(kind='bar', ax=ax, colormap=color_map)
        
        # Set plot title and labels
        if title:
            ax.set_title(title, fontsize=14)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add legend and grid
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot if filename provided
        if filename:
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison bar chart to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_heatmap(self, data, title=None, cmap='viridis', annot=True,
                   vmin=None, vmax=None, center=None, robust=False,
                   filename=None, show_plot=False):
        """
        Create a heatmap visualization.
        
        Args:
            data (pd.DataFrame): DataFrame with data for heatmap
            title (str, optional): Plot title
            cmap (str): Colormap for heatmap
            annot (bool): Whether to annotate cells with values
            vmin (float, optional): Minimum value for colormap
            vmax (float, optional): Maximum value for colormap
            center (float, optional): Center value for diverging colormaps
            robust (bool): Whether to use robust quantiles for color mapping
            filename (str, optional): Filename to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            plt.Figure: The generated figure
        """
        plt.figure(figsize=(12, 10))
        
        # Create the heatmap
        ax = sns.heatmap(data, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax,
                        center=center, robust=robust, linewidths=0.5, 
                        fmt='.2f' if annot else None)
        
        # Set plot title
        if title:
            plt.title(title, fontsize=14)
        
        # Adjust tick labels
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot if filename provided
        if filename:
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            fig = plt.gcf()
            plt.close(fig)
            return fig
    
    def create_dashboard(self, plots, layout=None, title=None, filename='dashboard.html'):
        """
        Create a dashboard with multiple plots.
        
        Args:
            plots (list): List of (figure, title) tuples
            layout (tuple, optional): Grid layout as (rows, cols)
            title (str, optional): Dashboard title
            filename (str): Filename to save the dashboard
            
        Returns:
            str: Path to generated dashboard file
        """
        try:
            import dash
            import dash_core_components as dcc
            import dash_html_components as html
            from dash.dependencies import Input, Output
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Dashboard creation requires Dash and Plotly. Please install with: pip install dash plotly")
            return None
        
        # Convert matplotlib figures to plotly
        plotly_figs = []
        for fig, plot_title in plots:
            import io
            import plotly.io as pio
            
            # Save matplotlib figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Create plotly figure from image
            plotly_fig = go.Figure(
                data=[go.Image(z=plt.imread(buf))],
                layout=go.Layout(title=plot_title)
            )
            plotly_figs.append(plotly_fig)
        
        # Create dashboard layout
        app = dash.Dash(__name__)
        
        # Determine layout
        if not layout:
            import math
            n_plots = len(plotly_figs)
            cols = min(3, n_plots)
            rows = math.ceil(n_plots / cols)
            layout = (rows, cols)
        
        # Create layout
        app.layout = html.Div([
            html.H1(title or "Bangladesh Simulation Dashboard"),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id=f'plot-{i}',
                        figure=fig
                    )
                ], className='col')
                for i, fig in enumerate(plotly_figs)
            ], className='row')
        ])
        
        # Save dashboard to HTML file
        save_path = self.viz_dir / filename
        
        # Run the dashboard app
        app.run_server(debug=False)
        
        return save_path
    
    def plot_map(self, data, geo_data, value_column, title=None,
               color_map='viridis', legend_title=None, tooltip_columns=None,
               filename=None, show_map=False):
        """
        Create an interactive choropleth map of Bangladesh using folium.
        
        Args:
            data (pd.DataFrame): DataFrame with region data
            geo_data (dict): GeoJSON data for Bangladesh regions
            value_column (str): Column in data containing values to map
            title (str, optional): Map title
            color_map (str): Colormap name
            legend_title (str, optional): Title for the legend
            tooltip_columns (list, optional): Columns to show in tooltip
            filename (str, optional): Filename to save the map
            show_map (bool): Whether to display the map
            
        Returns:
            folium.Map: The generated map
        """
        try:
            import folium
            from folium.features import GeoJsonTooltip
        except ImportError:
            logger.error("Map creation requires folium. Please install with: pip install folium")
            return None
        
        # Create a base map centered on Bangladesh
        m = folium.Map(
            location=[23.8, 90.4],  # Center of Bangladesh
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Add title
        if title:
            title_html = f'''
                <h3 align="center" style="font-size:16px">{title}</h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
        
        # Prepare tooltip
        if tooltip_columns:
            tooltip = GeoJsonTooltip(
                fields=tooltip_columns,
                aliases=tooltip_columns,
                localize=True,
                sticky=False,
                labels=True
            )
        else:
            tooltip = folium.features.GeoJsonTooltip(
                fields=['name', value_column],
                aliases=['Division', legend_title or value_column],
                localize=True,
                sticky=False,
                labels=True
            )
        
        # Add choropleth layer
        choropleth = folium.Choropleth(
            geo_data=geo_data,
            name='choropleth',
            data=data,
            columns=['division', value_column],
            key_on='feature.properties.division',
            fill_color=color_map,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=legend_title or value_column,
            highlight=True
        ).add_to(m)
        
        # Add tooltip to the choropleth layer
        choropleth.geojson.add_child(tooltip)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save the map if filename provided
        if filename:
            save_path = self.viz_dir / filename
            m.save(str(save_path))
            logger.info(f"Saved map to {save_path}")
        
        if show_map:
            return m
        else:
            return m
    
    def plot_comparative_scenarios(self, scenario_results, variables, scenario_names=None,
                                 title=None, ylabel=None, start_year=None, end_year=None,
                                 filename=None, show_plot=False):
        """
        Create a comparative plot of different scenarios.
        
        Args:
            scenario_results (dict): Dictionary mapping scenario names to DataFrames
            variables (str or list): Variable(s) to plot across scenarios
            scenario_names (list, optional): Custom names for scenarios
            title (str, optional): Plot title
            ylabel (str, optional): Y-axis label
            start_year (int, optional): Start year for x-axis
            end_year (int, optional): End year for x-axis
            filename (str, optional): Filename to save the plot
            show_plot (bool): Whether to display the plot
            
        Returns:
            plt.Figure: The generated figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to list if single variable
        if isinstance(variables, str):
            variables = [variables]
        
        # Create a custom color palette for scenarios
        n_scenarios = len(scenario_results)
        colors = sns.color_palette("viridis", n_scenarios)
        
        # Plot each scenario
        for i, (scenario, data) in enumerate(scenario_results.items()):
            scenario_label = scenario_names[i] if scenario_names and i < len(scenario_names) else scenario
            
            for j, var in enumerate(variables):
                if var in data.columns:
                    linestyle = '-' if j == 0 else '--' if j == 1 else ':' if j == 2 else '-.'
                    var_label = f"{scenario_label}: {var}" if len(variables) > 1 else scenario_label
                    data[var].plot(ax=ax, label=var_label, color=colors[i], linestyle=linestyle, linewidth=2)
        
        # Set plot title and labels
        if title:
            ax.set_title(title, fontsize=14)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Year', fontsize=12)
        
        # Format x-axis as years
        if start_year and end_year:
            ax.set_xlim(pd.Timestamp(f"{start_year}"), pd.Timestamp(f"{end_year}"))
        
        # Add legend and grid
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot if filename provided
        if filename:
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparative scenarios plot to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def create_impact_chart(self, impacts, systems, impact_types, title=None,
                          cmap='RdYlGn', filename=None, show_plot=False):
        """
        Create an impact visualization chart.
        
        Args:
            impacts (dict): Nested dictionary of impacts by system and impact type
            systems (list): List of systems to include
            impact_types (list): List of impact types to include
            title (str, optional): Chart title
            cmap (str): Colormap for impact visualization
            filename (str, optional): Filename to save the chart
            show_plot (bool): Whether to display the chart
            
        Returns:
            plt.Figure: The generated figure
        """
        # Create DataFrame from impacts dictionary
        data = []
        for system in systems:
            if system in impacts:
                for impact_type in impact_types:
                    if impact_type in impacts[system]:
                        data.append({
                            'System': system,
                            'Impact Type': impact_type,
                            'Value': impacts[system][impact_type]
                        })
        
        df = pd.DataFrame(data)
        
        # Create pivot table for heatmap
        pivot_df = df.pivot(index='System', columns='Impact Type', values='Value')
        
        # Create the impact chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the heatmap
        sns.heatmap(pivot_df, annot=True, cmap=cmap, center=0, 
                   linewidths=0.5, fmt='.2f', ax=ax)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        # Save the chart if filename provided
        if filename:
            save_path = self.viz_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved impact chart to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig

    # Add section-specific plot generation methods
    def create_economic_plots(self, data, output_dir):
        """
        Create plots for the economic section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing economic simulation data
            output_dir (Path): Directory to save the plots
        """
        self.logger.info(f"Generating economic plots in {output_dir}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty economic data provided")
                return False
                
            # Plot GDP Projection (if data available)
            if 'gdp_projection' in data and 'years' in data and len(data['gdp_projection']) > 0:
                plt.figure()
                plt.plot(data['years'], data['gdp_projection'], marker='o', linewidth=2, color='#1976D2')
                plt.title('GDP Projection')
                plt.xlabel('Year')
                plt.ylabel('GDP (USD Billions)')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'gdp_projection.png')
                plt.close()
            else:
                self.logger.warning("GDP projection data not available")
            
            # Other plots only if data is available
            if 'sector_contributions' in data and 'sectors' in data and len(data['sector_contributions']) > 0:
                plt.figure()
                # Use a simple color palette
                colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(data['sectors'])))
                plt.pie(data['sector_contributions'], labels=data['sectors'], 
                      autopct='%1.1f%%', startangle=90, colors=colors)
                plt.title('Economic Sector Contributions')
                plt.savefig(output_dir / 'sector_contributions.png')
                plt.close()
            else:
                self.logger.warning("Sector contributions data not available")
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating economic plots: {str(e)}")
            return False
            
    def create_environmental_plots(self, data, output_dir):
        """
        Create plots for the environmental section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing environmental simulation data
            output_dir (Path): Directory to save the plots
        """
        self.logger.info(f"Generating environmental plots in {output_dir}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty environmental data provided")
                return False
            
            # Environmental Health Index
            if 'environmental_health' in data and 'years' in data and len(data['environmental_health']) > 0:
                plt.figure()
                plt.plot(data['years'], data['environmental_health'], marker='o', linewidth=2, color='#43A047')
                plt.title('Environmental Health Index')
                plt.xlabel('Year')
                plt.ylabel('Index Value')
                plt.ylim(0, 1.0)
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'environmental_health.png')
                plt.close()
            else:
                self.logger.warning("Environmental health data not available")
                
            # Temperature Change (if available)
            if 'temperature' in data and 'years' in data and len(data['temperature']) > 0:
                plt.figure()
                plt.plot(data['years'], data['temperature'], marker='o', linewidth=2, color='#E53935')
                plt.title('Temperature Change')
                plt.xlabel('Year')
                plt.ylabel('Temperature (°C)')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'temperature_change.png')
                plt.close()
            else:
                self.logger.warning("Temperature data not available")
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating environmental plots: {str(e)}")
            return False
            
    def create_demographic_plots(self, data, output_dir):
        """
        Create plots for the demographic section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing demographic simulation data
            output_dir (Path): Directory to save the plots
        """
        self.logger.info(f"Generating demographic plots in {output_dir}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty demographic data provided")
                return False
            
            # Urbanization Rate
            if 'urbanization_rate' in data and 'years' in data and len(data['years']) > 0:
                plt.figure()
                plt.plot(data['years'], [rate * 100 for rate in data['urbanization_rate']], 
                         marker='o', linewidth=2, color='#3949AB')
                plt.title('Urbanization Rate')
                plt.xlabel('Year')
                plt.ylabel('Urbanization (%)')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'urbanization_rate.png')
                plt.close()
            else:
                self.logger.warning("Urbanization rate data not available")
                
            # Population Projection (if available)
            if 'population' in data and 'years' in data and len(data['population']) > 0:
                plt.figure()
                plt.plot(data['years'], data['population'], marker='o', linewidth=2, color='#26A69A')
                plt.title('Population Projection')
                plt.xlabel('Year')
                plt.ylabel('Population')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'population_projection.png')
                plt.close()
            else:
                self.logger.warning("Population projection data not available")
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating demographic plots: {str(e)}")
            return False
            
    def create_infrastructure_plots(self, data, output_dir):
        """
        Create plots for the infrastructure section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing infrastructure simulation data
            output_dir (Path): Directory to save the plots
        """
        self.logger.info(f"Generating infrastructure plots in {output_dir}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty infrastructure data provided")
                return False
            
            # Connectivity Index
            if 'connectivity_index' in data and 'years' in data and len(data['connectivity_index']) > 0:
                plt.figure()
                plt.plot(data['years'], data['connectivity_index'], marker='o', linewidth=2, color='#7E57C2')
                plt.title('Connectivity Index')
                plt.xlabel('Year')
                plt.ylabel('Index Value')
                plt.ylim(0, 1.0)
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'connectivity_index.png')
                plt.close()
            else:
                self.logger.warning("Connectivity index data not available")
                
            # Infrastructure Investment (if available)
            if 'infrastructure_investment' in data and 'years' in data and len(data['infrastructure_investment']) > 0:
                plt.figure()
                plt.plot(data['years'], data['infrastructure_investment'], marker='o', linewidth=2, color='#5E35B1')
                plt.title('Infrastructure Investment')
                plt.xlabel('Year')
                plt.ylabel('% of GDP')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'infrastructure_investment.png')
                plt.close()
            else:
                self.logger.warning("Infrastructure investment data not available")
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating infrastructure plots: {str(e)}")
            return False
            
    def create_governance_plots(self, data, output_dir):
        """
        Create plots for the governance section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing governance simulation data
            output_dir (Path): Directory to save the plots
        """
        self.logger.info(f"Generating governance plots in {output_dir}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty governance data provided")
                return False
                
            # Governance Effectiveness
            if 'governance_effectiveness' in data and 'years' in data and len(data['governance_effectiveness']) > 0:
                plt.figure()
                plt.plot(data['years'], data['governance_effectiveness'], marker='o', linewidth=2, color='#FB8C00')
                plt.title('Governance Effectiveness')
                plt.xlabel('Year')
                plt.ylabel('Index Value')
                plt.ylim(0, 1.0)
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'governance_effectiveness.png')
                plt.close()
            else:
                self.logger.warning("Governance effectiveness data not available")
                
            # Corruption Index (if available)
            if 'corruption_index' in data and 'years' in data and len(data['corruption_index']) > 0:
                plt.figure()
                plt.plot(data['years'], data['corruption_index'], marker='o', linewidth=2, color='#F4511E')
                plt.title('Corruption Index')
                plt.xlabel('Year')
                plt.ylabel('Index Value')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'corruption_index.png')
                plt.close()
            else:
                self.logger.warning("Corruption index data not available")
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating governance plots: {str(e)}")
            return False
            
    # Create composite plots with a minimal style
    def create_economic_composite_plot(self, data, output_path):
        """
        Create a composite plot for the economic section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing economic simulation data
            output_path (Path): Path to save the plot
        """
        self.logger.info(f"Generating economic composite plot at {output_path}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty economic data provided")
                return False
                
            # Check if we have enough data to create a useful plot
            has_gdp = 'gdp_projection' in data and 'years' in data and len(data['gdp_projection']) > 0
            
            if not has_gdp:
                self.logger.warning("Not enough economic data available for composite plot")
                return False
                
            # Create simple single plot with available data
            plt.figure(figsize=(10, 6))
            
            # GDP Growth
            if has_gdp:
                plt.plot(data['years'], data['gdp_projection'], marker='o', linewidth=2, 
                        color='#1976D2', label='GDP (USD Billions)')
                
                # Add trend line
                if len(data['years']) > 2:
                    z = np.polyfit(data['years'], data['gdp_projection'], 1)
                    p = np.poly1d(z)
                    plt.plot(data['years'], p(data['years']), ":", color='#1976D2', alpha=0.7)
            
            # Trade Balance if available
            if 'trade_balance' in data and len(data['trade_balance']) > 0:
                plt.plot(data['years'], data['trade_balance'], marker='s', linewidth=2, 
                        color='#43A047', label='Trade Balance (USD Billions)')
                
                # Add reference line at 0
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            plt.title('Economic Indicators')
            plt.xlabel('Year')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating economic composite plot: {str(e)}")
            return False

    def create_environmental_composite_plot(self, data, output_path):
        """
        Create a composite plot for the environmental section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing environmental simulation data
            output_path (Path): Path to save the plot
        """
        self.logger.info(f"Generating environmental composite plot at {output_path}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty environmental data provided")
                return False
                
            # Check if we have enough data to create a useful plot
            has_temp = 'temperature' in data and 'years' in data and len(data['temperature']) > 0
            has_env_health = 'environmental_health' in data and 'years' in data and len(data['environmental_health']) > 0
            
            if not (has_temp or has_env_health):
                self.logger.warning("Not enough environmental data available for composite plot")
                return False
                
            # Create simple single plot with available data
            plt.figure(figsize=(10, 6))
            
            # Temperature data
            if has_temp:
                plt.plot(data['years'], data['temperature'], marker='o', linewidth=2, 
                        color='#E53935', label='Temperature (°C)')
            
            # Environmental Health data
            if has_env_health:
                plt.plot(data['years'], data['environmental_health'], marker='s', linewidth=2, 
                        color='#43A047', label='Environmental Health Index')
                
            plt.title('Environmental Indicators')
            plt.xlabel('Year')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating environmental composite plot: {str(e)}")
            return False
    
    def create_demographic_composite_plot(self, data, output_path):
        """
        Create a composite plot for the demographic section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing demographic simulation data
            output_path (Path): Path to save the plot
        """
        self.logger.info(f"Generating demographic composite plot at {output_path}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty demographic data provided")
                return False
                
            # Check if we have enough data to create a useful plot
            has_population = 'population' in data and 'years' in data and len(data['population']) > 0
            has_urbanization = 'urbanization_rate' in data and 'years' in data and len(data['urbanization_rate']) > 0
            
            if not (has_population or has_urbanization):
                self.logger.warning("Not enough demographic data available for composite plot")
                return False
                
            # Create simple single plot with available data
            plt.figure(figsize=(10, 6))
            
            # Population data
            if has_population:
                population_color = '#26A69A'
                pop_line = plt.plot(data['years'], data['population'], marker='o', linewidth=2, 
                        color=population_color, label='Population')
                
                # Add trend line
                if len(data['years']) > 2:
                    z = np.polyfit(data['years'], data['population'], 1)
                    p = np.poly1d(z)
                    plt.plot(data['years'], p(data['years']), ":", color=population_color, alpha=0.7)
            
            # Urbanization data (plot on secondary y-axis if both metrics present)
            if has_urbanization:
                urban_color = '#3949AB'
                if has_population:
                    ax2 = plt.gca().twinx()
                    urban_line = ax2.plot(data['years'], [rate * 100 for rate in data['urbanization_rate']], 
                            marker='s', linewidth=2, color=urban_color, label='Urbanization (%)')
                    ax2.set_ylabel('Urbanization (%)', color=urban_color)
                    ax2.tick_params(axis='y', colors=urban_color)
                    
                    # Combine legends
                    lines = pop_line + urban_line
                    labels = [line.get_label() for line in lines]
                    plt.legend(lines, labels, loc='upper left')
                else:
                    plt.plot(data['years'], [rate * 100 for rate in data['urbanization_rate']], 
                            marker='s', linewidth=2, color=urban_color, label='Urbanization (%)')
                    plt.legend(loc='best')
            else:
                plt.legend(loc='best')
                
            plt.title('Demographic Indicators')
            plt.xlabel('Year')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating demographic composite plot: {str(e)}")
            return False
            
    def create_infrastructure_composite_plot(self, data, output_path):
        """
        Create a composite plot for the infrastructure section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing infrastructure simulation data
            output_path (Path): Path to save the plot
        """
        self.logger.info(f"Generating infrastructure composite plot at {output_path}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty infrastructure data provided")
                return False
                
            # Check if we have enough data to create a useful plot
            has_connectivity = 'connectivity_index' in data and 'years' in data and len(data.get('connectivity_index', [])) > 0
            has_investment = 'infrastructure_investment' in data and 'years' in data and len(data.get('infrastructure_investment', [])) > 0
            
            if not (has_connectivity or has_investment):
                self.logger.warning("Not enough infrastructure data available for composite plot")
                return False
                
            # Create simple single plot with available data
            plt.figure(figsize=(10, 6))
            
            # Connectivity data
            if has_connectivity:
                plt.plot(data['years'], data['connectivity_index'], marker='o', linewidth=2, 
                        color='#7E57C2', label='Connectivity Index')
            
            # Investment data
            if has_investment:
                plt.plot(data['years'], data['infrastructure_investment'], marker='s', linewidth=2, 
                        color='#5E35B1', label='Investment (% of GDP)')
                
            plt.title('Infrastructure Indicators')
            plt.xlabel('Year')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating infrastructure composite plot: {str(e)}")
            return False
            
    def create_governance_composite_plot(self, data, output_path):
        """
        Create a composite plot for the governance section using actual simulation data.
        
        Args:
            data (dict): Dictionary containing governance simulation data
            output_path (Path): Path to save the plot
        """
        self.logger.info(f"Generating governance composite plot at {output_path}")
        self._setup_minimal_style()
        
        try:
            if not data or not isinstance(data, dict):
                self.logger.error("Invalid or empty governance data provided")
                return False
                
            # Check if we have enough data to create a useful plot
            has_effectiveness = 'governance_effectiveness' in data and 'years' in data and len(data.get('governance_effectiveness', [])) > 0
            has_corruption = 'corruption_index' in data and 'years' in data and len(data.get('corruption_index', [])) > 0
            
            if not (has_effectiveness or has_corruption):
                self.logger.warning("Not enough governance data available for composite plot")
                return False
                
            # Create simple single plot with available data
            plt.figure(figsize=(10, 6))
            
            # Governance Effectiveness
            if has_effectiveness:
                plt.plot(data['years'], data['governance_effectiveness'], marker='o', linewidth=2, 
                        color='#FB8C00', label='Governance Effectiveness')
            
            # Corruption Index (might need inverted scale if lower is better)
            if has_corruption:
                plt.plot(data['years'], data['corruption_index'], marker='s', linewidth=2, 
                        color='#F4511E', label='Corruption Index')
                
            plt.title('Governance Indicators')
            plt.xlabel('Year')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating governance composite plot: {str(e)}")
            return False
