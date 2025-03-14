#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Climate system model for Bangladesh simulation.
This module handles temperature, precipitation patterns, and climate change effects.
"""

import numpy as np
import pandas as pd
from scipy.stats import gamma


class ClimateSystem:
    """
    Climate system model representing Bangladesh's temperature and precipitation patterns
    with climate change effects.
    """
    
    def __init__(self, config, environmental_data):
        """
        Initialize the climate system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the climate system
            environmental_data (dict): Initial environmental data
        """
        self.config = config
        
        # Temperature parameters
        self.temperature_anomaly = environmental_data.get('temperature_anomaly', 0.8)  # °C above pre-industrial
        self.temperature_variability = environmental_data.get('temperature_variability', 0.3)  # Standard deviation
        self.warming_trend = config.get('warming_trend', 0.03)  # °C per year
        
        # Precipitation parameters
        self.annual_rainfall = environmental_data.get('annual_rainfall', 2500.0)  # mm/year
        self.rainfall_variability = environmental_data.get('rainfall_variability', 0.15)  # Coefficient of variation
        self.rainfall_trend = config.get('rainfall_trend', 0.005)  # % change per year
        
        # Monsoon patterns
        self.monsoon_intensity = environmental_data.get('monsoon_intensity', 1.0)  # normalized
        self.monsoon_onset = environmental_data.get('monsoon_onset', 152)  # day of year (June 1st)
        self.monsoon_duration = environmental_data.get('monsoon_duration', 120)  # days
        self.monsoon_timing_shift = environmental_data.get('monsoon_timing_shift', 0.0)  # days
        
        # Seasonal parameters (% of annual rainfall in each season)
        self.seasonal_rainfall_distribution = environmental_data.get('seasonal_rainfall', {
            'winter': 0.04,  # Dec-Feb
            'pre_monsoon': 0.18,  # Mar-May
            'monsoon': 0.68,  # Jun-Sep
            'post_monsoon': 0.10  # Oct-Nov
        })
        
        # Spatial variation (relative to national average)
        self.spatial_rainfall_distribution = environmental_data.get('spatial_rainfall', {
            'northwest': 0.7,  # Dryer
            'northeast': 1.2,  # Wetter (Sylhet)
            'central': 0.9,
            'southwest': 1.1,  # Khulna
            'southeast': 1.4,  # Chattogram, very wet
            'coastal': 1.2
        })
        
        # Extreme indicators
        self.rainfall_adequacy = 1.0  # 1.0 is adequate
        self.temperature_stress = 0.0  # 0.0 is no stress
        self.current_year = config.get('start_year', 2023)
        
        print("Climate system initialized")
        
    def generate_monthly_temperatures(self, year):
        """
        Generate monthly temperatures for the given year.
        
        Args:
            year (int): Simulation year
            
        Returns:
            dict: Monthly temperatures
        """
        # Base seasonal pattern for Bangladesh (°C)
        baseline_temperatures = {
            'january': 18.0,
            'february': 21.0,
            'march': 26.0,
            'april': 29.0,
            'may': 30.0,
            'june': 29.5,
            'july': 29.0,
            'august': 29.0,
            'september': 29.0,
            'october': 28.0,
            'november': 24.0,
            'december': 20.0
        }
        
        # Apply warming trend and anomaly
        warming = self.temperature_anomaly
        
        # Add seasonal variability
        monthly_temps = {}
        for month, baseline in baseline_temperatures.items():
            # Apply warming
            temp = baseline + warming
            
            # Add random variability (more in winter, less in summer)
            if month in ['december', 'january', 'february']:
                variability = np.random.normal(0, self.temperature_variability * 1.2)
            else:
                variability = np.random.normal(0, self.temperature_variability)
                
            monthly_temps[month] = temp + variability
            
        return monthly_temps
    
    def generate_monthly_rainfall(self, year):
        """
        Generate monthly rainfall for the given year.
        
        Args:
            year (int): Simulation year
            
        Returns:
            dict: Monthly rainfall
        """
        # Base seasonal pattern for Bangladesh (as fraction of annual rainfall)
        baseline_fractions = {
            'january': 0.01,
            'february': 0.01,
            'march': 0.03,
            'april': 0.07,
            'may': 0.08,
            'june': 0.18,
            'july': 0.24,
            'august': 0.18,
            'september': 0.08,
            'october': 0.07,
            'november': 0.03,
            'december': 0.02
        }
        
        # Calculate current annual rainfall
        current_annual_rainfall = self.annual_rainfall
        
        # Apply monsoon intensity effect
        monsoon_months = ['june', 'july', 'august', 'september']
        for month in monsoon_months:
            baseline_fractions[month] *= self.monsoon_intensity
            
        # Normalize to ensure fractions sum to 1
        total_fraction = sum(baseline_fractions.values())
        normalized_fractions = {m: f/total_fraction for m, f in baseline_fractions.items()}
        
        # Apply variability to each month
        monthly_rainfall = {}
        for month, fraction in normalized_fractions.items():
            expected_rainfall = fraction * current_annual_rainfall
            
            # More variability in pre-monsoon and monsoon months
            if month in ['april', 'may'] + monsoon_months:
                cv = self.rainfall_variability * 1.2
            else:
                cv = self.rainfall_variability
                
            # Generate rainfall with gamma distribution for more realistic distribution
            shape = 1 / (cv ** 2)
            scale = expected_rainfall / shape
            monthly_rainfall[month] = gamma.rvs(shape, scale=scale)
            
        return monthly_rainfall
    
    def calculate_rainfall_adequacy(self, monthly_rainfall):
        """
        Calculate rainfall adequacy for agriculture.
        
        Args:
            monthly_rainfall (dict): Monthly rainfall amounts
            
        Returns:
            float: Rainfall adequacy index (1.0 is adequate)
        """
        # Define optimal rainfall by month for Bangladesh agriculture
        optimal_rainfall = {
            'january': 10,
            'february': 20,
            'march': 50,
            'april': 100,
            'may': 200,
            'june': 300,
            'july': 330,
            'august': 300,
            'september': 250,
            'october': 150,
            'november': 50,
            'december': 20
        }
        
        # Calculate adequacy for each month
        monthly_adequacy = {}
        for month, optimal in optimal_rainfall.items():
            actual = monthly_rainfall.get(month, 0)
            
            # Too little rain is worse than too much (up to a point)
            if actual < optimal:
                adequacy = actual / optimal
            else:
                # Too much rain reduces adequacy, but less severely
                excess = (actual - optimal) / optimal
                adequacy = 1.0 - 0.3 * min(excess, 2.0)
                
            monthly_adequacy[month] = max(adequacy, 0.0)
            
        # Weight by importance of month for agriculture
        weights = {
            'january': 0.03,
            'february': 0.04,
            'march': 0.06,
            'april': 0.10,
            'may': 0.12,
            'june': 0.15,
            'july': 0.15,
            'august': 0.15,
            'september': 0.10,
            'october': 0.06,
            'november': 0.03,
            'december': 0.01
        }
        
        # Calculate weighted average
        weighted_adequacy = sum(monthly_adequacy[m] * weights[m] for m in monthly_adequacy) / sum(weights.values())
        
        return weighted_adequacy
    
    def calculate_temperature_stress(self, monthly_temperatures):
        """
        Calculate temperature stress for agriculture and human activities.
        
        Args:
            monthly_temperatures (dict): Monthly temperatures
            
        Returns:
            float: Temperature stress index (0.0 is no stress, 1.0 is extreme stress)
        """
        # Define temperature thresholds for stress
        optimal_max_temp = {
            'january': 26,
            'february': 28,
            'march': 32,
            'april': 34,
            'may': 35,
            'june': 35,
            'july': 34,
            'august': 34,
            'september': 34,
            'october': 33,
            'november': 30,
            'december': 28
        }
        
        # Calculate stress for each month
        monthly_stress = {}
        for month, optimal in optimal_max_temp.items():
            actual = monthly_temperatures.get(month, 20)
            
            # Temperature stress increases as temperature exceeds optimal
            if actual <= optimal:
                stress = 0.0
            else:
                # Stress increases progressively with temperature
                excess = actual - optimal
                stress = min(0.2 * excess, 1.0)
                
            monthly_stress[month] = stress
            
        # Weight by importance of month for agriculture and human activities
        # Summer months get higher weights due to higher population vulnerability
        weights = {
            'january': 0.04,
            'february': 0.04,
            'march': 0.06,
            'april': 0.10,
            'may': 0.12,
            'june': 0.15,
            'july': 0.15,
            'august': 0.15,
            'september': 0.10,
            'october': 0.06,
            'november': 0.03,
            'december': 0.00
        }
        
        # Calculate weighted average
        weighted_stress = sum(monthly_stress[m] * weights[m] for m in monthly_stress) / sum(weights.values())
        
        return weighted_stress
            
    def calculate_monsoon_dynamics(self, year, regional_climate):
        """
        Calculate monsoon dynamics for the year.
        
        Args:
            year (int): Simulation year
            regional_climate (dict): Regional climate parameters
            
        Returns:
            dict: Monsoon dynamics
        """
        # Update monsoon parameters based on climate change
        baseline_onset = 152  # June 1st
        baseline_duration = 120  # days
        
        # Extract regional climate parameters
        monsoon_intensity = regional_climate.get('monsoon_intensity', self.monsoon_intensity)
        monsoon_timing_shift = regional_climate.get('monsoon_timing_shift', self.monsoon_timing_shift)
        
        # Calculate onset with climate change effect and natural variability
        onset_variability = np.random.normal(0, 5)  # Natural variability in onset
        current_onset = baseline_onset + monsoon_timing_shift + onset_variability
        
        # Calculate duration with climate change effect and natural variability
        # Warming generally leads to longer monsoon with more intense rainfall
        duration_change = 0.1 * (monsoon_intensity - 1.0) * baseline_duration
        duration_variability = np.random.normal(0, 7)  # Natural variability in duration
        current_duration = baseline_duration + duration_change + duration_variability
        
        # Calculate rainfall distribution during monsoon
        # Warming tends to intensify rainfall in peak monsoon months
        peak_intensity = monsoon_intensity * (1 + 0.2 * (regional_climate.get('regional_temperature_anomaly', 0) - 0.8))
        
        # Return monsoon parameters
        monsoon_params = {
            'onset_day': current_onset,
            'duration_days': current_duration,
            'intensity': monsoon_intensity,
            'peak_intensity': peak_intensity,
            'timing_shift': monsoon_timing_shift
        }
        
        return monsoon_params
    
    def update_parameters(self, global_climate, regional_climate):
        """
        Update internal climate parameters based on global and regional climate inputs.
        
        Args:
            global_climate (dict): Global climate parameters
            regional_climate (dict): Regional climate parameters
            
        Returns:
            dict: Updated parameters
        """
        # Update temperature anomaly
        self.temperature_anomaly = regional_climate.get('regional_temperature_anomaly', self.temperature_anomaly)
        
        # Update annual rainfall
        self.annual_rainfall = regional_climate.get('annual_rainfall', self.annual_rainfall)
        
        # Update monsoon parameters
        self.monsoon_intensity = regional_climate.get('monsoon_intensity', self.monsoon_intensity)
        self.monsoon_timing_shift = regional_climate.get('monsoon_timing_shift', self.monsoon_timing_shift)
        
        # Return updated parameters
        updated_params = {
            'temperature_anomaly': self.temperature_anomaly,
            'annual_rainfall': self.annual_rainfall,
            'monsoon_intensity': self.monsoon_intensity,
            'monsoon_timing_shift': self.monsoon_timing_shift
        }
        
        return updated_params
    
    def step(self, year, global_climate, regional_climate):
        """
        Advance the climate system by one time step.
        
        Args:
            year (int): Current simulation year
            global_climate (dict): Global climate parameters
            regional_climate (dict): Regional climate parameters
            
        Returns:
            dict: Climate system results after the step
        """
        # Update climate parameters
        updated_params = self.update_parameters(global_climate, regional_climate)
        
        # Generate monthly temperatures and rainfall
        monthly_temperatures = self.generate_monthly_temperatures(year)
        monthly_rainfall = self.generate_monthly_rainfall(year)
        
        # Calculate rainfall adequacy
        self.rainfall_adequacy = self.calculate_rainfall_adequacy(monthly_rainfall)
        
        # Calculate temperature stress
        self.temperature_stress = self.calculate_temperature_stress(monthly_temperatures)
        
        # Calculate monsoon dynamics
        monsoon_dynamics = self.calculate_monsoon_dynamics(year, regional_climate)
        
        # Update current year
        self.current_year = year
        
        # Compile results
        results = {
            'year': year,
            'temperature_anomaly': self.temperature_anomaly,
            'annual_rainfall': self.annual_rainfall,
            'monthly_temperatures': monthly_temperatures,
            'monthly_rainfall': monthly_rainfall,
            'average_temperature': sum(monthly_temperatures.values()) / 12,
            'total_rainfall': sum(monthly_rainfall.values()),
            'rainfall_adequacy': self.rainfall_adequacy,
            'temperature_stress': self.temperature_stress,
            'monsoon': monsoon_dynamics
        }
        
        return results
