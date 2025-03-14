"""
Visualization module for the Bangladesh Development Simulation Model.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

class Plotter:
    """Class for generating visualizations of simulation results."""
    
    def __init__(self, output_dir: str = 'results/plots'):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Set figure size and DPI
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 300
    
    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """
        Save a plot to file.
        
        Args:
            fig: The figure to save
            filename: Name of the file
        """
        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logger.info(f"Plot saved to {output_path}")
    
    def plot_development_trajectory(self, history: List[Dict[str, Any]]) -> None:
        """
        Plot the development trajectory over time.
        
        Args:
            history: List of simulation states
        """
        df = pd.DataFrame(history)
        
        fig, ax = plt.subplots()
        
        # Plot indices
        indices = ['development_index', 'sustainability_index', 
                  'resilience_index', 'wellbeing_index']
        
        for idx in indices:
            ax.plot(df.index, df[idx], label=idx.replace('_', ' ').title())
        
        ax.set_title('Development Trajectory')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Index Value')
        ax.legend()
        ax.grid(True)
        
        self._save_plot(fig, 'development_trajectory.png')
    
    def plot_sector_performance(self, history: List[Dict[str, Any]]) -> None:
        """
        Plot sector performance over time.
        
        Args:
            history: List of simulation states
        """
        df = pd.DataFrame(history)
        
        fig, ax = plt.subplots()
        
        # Plot sector contributions
        sectors = ['agriculture', 'industry', 'services']
        for sector in sectors:
            ax.plot(df.index, df[f'{sector}_gdp'], label=sector.title())
        
        ax.set_title('Sector Performance')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('GDP Contribution')
        ax.legend()
        ax.grid(True)
        
        self._save_plot(fig, 'sector_performance.png')
    
    def plot_environmental_impact(self, history: List[Dict[str, Any]]) -> None:
        """
        Plot environmental impact indicators.
        
        Args:
            history: List of simulation states
        """
        df = pd.DataFrame(history)
        
        fig = plt.figure()
        gs = GridSpec(2, 2, figure=fig)
        
        # Climate change
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df.index, df['temperature'], label='Temperature')
        ax1.set_title('Temperature Change')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('°C')
        ax1.legend()
        ax1.grid(True)
        
        # Precipitation
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df.index, df['precipitation'], label='Precipitation')
        ax2.set_title('Precipitation Change')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('%')
        ax2.legend()
        ax2.grid(True)
        
        # Sea level
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df.index, df['sea_level'], label='Sea Level')
        ax3.set_title('Sea Level Rise')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('m')
        ax3.legend()
        ax3.grid(True)
        
        # Environmental health
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df.index, df['flood_risk'], label='Flood Risk')
        ax4.plot(df.index, df['water_stress'], label='Water Stress')
        ax4.set_title('Environmental Health')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Index')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        self._save_plot(fig, 'environmental_impact.png')
    
    def plot_demographic_indicators(self, history: List[Dict[str, Any]]) -> None:
        """
        Plot demographic indicators over time.
        
        Args:
            history: List of simulation states
        """
        df = pd.DataFrame(history)
        
        fig = plt.figure()
        gs = GridSpec(2, 2, figure=fig)
        
        # Population
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df.index, df['population'], label='Population')
        ax1.set_title('Population Growth')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True)
        
        # HDI
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df.index, df['hdi'], label='HDI')
        ax2.set_title('Human Development Index')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Index')
        ax2.legend()
        ax2.grid(True)
        
        # Poverty
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df.index, df['poverty_rate'], label='Poverty Rate')
        ax3.set_title('Poverty Rate')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('%')
        ax3.legend()
        ax3.grid(True)
        
        # Inequality
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df.index, df['inequality'], label='Inequality')
        ax4.set_title('Inequality Index')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Index')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        self._save_plot(fig, 'demographic_indicators.png')
    
    def plot_infrastructure_development(self, history: List[Dict[str, Any]]) -> None:
        """
        Plot infrastructure development indicators.
        
        Args:
            history: List of simulation states
        """
        df = pd.DataFrame(history)
        
        fig = plt.figure()
        gs = GridSpec(2, 2, figure=fig)
        
        # Physical infrastructure
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df.index, df['road_length'], label='Road Length')
        ax1.plot(df.index, df['electricity_access'], label='Electricity Access')
        ax1.set_title('Physical Infrastructure')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Index')
        ax1.legend()
        ax1.grid(True)
        
        # Quality indicators
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df.index, df['infrastructure_quality'], label='Quality')
        ax2.plot(df.index, df['connectivity'], label='Connectivity')
        ax2.set_title('Infrastructure Quality')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Index')
        ax2.legend()
        ax2.grid(True)
        
        # Resilience
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(df.index, df['infrastructure_resilience'], label='Resilience')
        ax3.set_title('Infrastructure Resilience')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Index')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        self._save_plot(fig, 'infrastructure_development.png')
    
    def plot_governance_indicators(self, history: List[Dict[str, Any]]) -> None:
        """
        Plot governance indicators over time.
        
        Args:
            history: List of simulation states
        """
        df = pd.DataFrame(history)
        
        fig = plt.figure()
        gs = GridSpec(2, 2, figure=fig)
        
        # Institutional quality
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df.index, df['bureaucracy_efficiency'], label='Bureaucracy')
        ax1.plot(df.index, df['transparency'], label='Transparency')
        ax1.set_title('Institutional Quality')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Index')
        ax1.legend()
        ax1.grid(True)
        
        # Corruption
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df.index, df['corruption_index'], label='Corruption')
        ax2.set_title('Corruption Index')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Index')
        ax2.legend()
        ax2.grid(True)
        
        # Policy performance
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(df.index, df['policy_effectiveness'], label='Effectiveness')
        ax3.plot(df.index, df['policy_implementation'], label='Implementation')
        ax3.plot(df.index, df['policy_coordination'], label='Coordination')
        ax3.set_title('Policy Performance')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Index')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        self._save_plot(fig, 'governance_indicators.png')
    
    def plot_correlation_matrix(self, history: List[Dict[str, Any]]) -> None:
        """
        Plot correlation matrix of key indicators.
        
        Args:
            history: List of simulation states
        """
        df = pd.DataFrame(history)
        
        # Select key indicators
        indicators = [
            'development_index', 'sustainability_index', 'resilience_index',
            'wellbeing_index', 'gdp_growth', 'employment_rate', 'hdi',
            'poverty_rate', 'infrastructure_quality', 'corruption_index'
        ]
        
        # Calculate correlation matrix
        corr_matrix = df[indicators].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, ax=ax)
        
        ax.set_title('Correlation Matrix of Key Indicators')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        self._save_plot(fig, 'correlation_matrix.png') 