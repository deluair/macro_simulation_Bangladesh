#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Disaster system model for Bangladesh simulation.
This module handles cyclones, floods, droughts, and other natural hazards.
"""

import numpy as np
import pandas as pd
from scipy.stats import weibull_min, poisson, gamma


class DisasterSystem:
    """
    Disaster system model representing Bangladesh's natural hazards and disaster risk.
    """
    
    def __init__(self, config, environmental_data):
        """
        Initialize the disaster system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the disaster system
            environmental_data (dict): Initial environmental data
        """
        self.config = config
        
        # Cyclone parameters
        self.cyclone_annual_frequency = environmental_data.get('cyclone_frequency', 1.4)  # Average number per year
        self.cyclone_intensity_scale = environmental_data.get('cyclone_intensity', 1.0)  # Scale factor (1.0 is baseline)
        self.cyclone_landfall_prob = environmental_data.get('cyclone_landfall_prob', 0.7)  # Probability of landfall
        
        # Flood parameters
        self.flood_annual_probability = environmental_data.get('flood_probability', 0.8)  # Annual probability of major flood
        self.flood_intensity_scale = environmental_data.get('flood_intensity', 1.0)  # Scale factor (1.0 is baseline)
        
        # Drought parameters
        self.drought_annual_probability = environmental_data.get('drought_probability', 0.25)  # Annual probability
        self.drought_intensity_scale = environmental_data.get('drought_intensity', 1.0)  # Scale factor (1.0 is baseline)
        
        # Other disasters
        self.riverbank_erosion_rate = environmental_data.get('erosion_rate', 1.0)  # Scale factor
        self.landslide_probability = environmental_data.get('landslide_probability', 0.2)  # Annual probability in hill areas
        
        # Disaster impacts tracking
        self.current_impacts = {
            'cyclone': 0.0,
            'flood': 0.0,
            'drought': 0.0,
            'erosion': 0.0,
            'landslide': 0.0
        }
        self.historical_events = []
        
        print("Disaster system initialized")
    
    def generate_cyclones(self, year, temperature_anomaly, sea_level_rise):
        """
        Generate tropical cyclones for the year based on climate conditions.
        
        Args:
            year (int): Current simulation year
            temperature_anomaly (float): Temperature anomaly in degrees C
            sea_level_rise (float): Sea level rise in meters
            
        Returns:
            list: Cyclone events for the year
        """
        # Climate change effects on cyclone frequency and intensity
        # Temperature effect: Higher temperatures increase frequency and intensity
        temp_effect_frequency = 0.15 * (temperature_anomaly - 0.8)  # Normalized to 0.8°C reference
        temp_effect_intensity = 0.2 * (temperature_anomaly - 0.8)
        
        # Sea level effect: Higher sea levels increase surge height and impacts
        sea_level_effect = 0.1 * (sea_level_rise / 0.3)  # Normalized to 0.3m reference
        
        # Update cyclone parameters with climate effects
        current_frequency = self.cyclone_annual_frequency * (1 + temp_effect_frequency)
        current_intensity = self.cyclone_intensity_scale * (1 + temp_effect_intensity)
        
        # Generate number of cyclones for the year using Poisson distribution
        num_cyclones = poisson.rvs(current_frequency)
        
        # Generate cyclones and their characteristics
        cyclones = []
        for i in range(num_cyclones):
            # Determine timing (month)
            # Cyclones typically occur in pre-monsoon (Apr-May) and post-monsoon (Oct-Nov) seasons
            season = np.random.choice(['pre-monsoon', 'post-monsoon'], p=[0.4, 0.6])
            if season == 'pre-monsoon':
                month = np.random.choice(['april', 'may', 'june'], p=[0.3, 0.5, 0.2])
            else:
                month = np.random.choice(['october', 'november', 'december'], p=[0.5, 0.4, 0.1])
            
            # Determine landfall (whether it hits Bangladesh)
            landfall = np.random.random() < self.cyclone_landfall_prob
            
            if landfall:
                # Determine location of landfall
                location = np.random.choice(['southeast', 'southwest', 'south-central', 'northeast', 'northwest'], 
                                           p=[0.4, 0.25, 0.2, 0.1, 0.05])
                
                # Determine intensity category (Saffir-Simpson scale, 1-5)
                # Using Weibull distribution for realistic intensity distribution
                # Shape parameter 2.0 gives a reasonable distribution of cyclone intensities
                raw_intensity = weibull_min.rvs(2.0, loc=1, scale=1.5 * current_intensity)
                category = min(5, max(1, int(raw_intensity)))
                
                # Calculate wind speed based on category
                base_wind = 65 + 20 * (category - 1)  # Approximate wind speed in knots
                wind_speed = base_wind * (1 + 0.1 * np.random.normal())  # Add some variability
                
                # Calculate storm surge height
                base_surge = 1.0 + 0.8 * (category - 1)  # Approximate surge in meters
                surge_height = base_surge * (1 + sea_level_effect + 0.2 * np.random.normal())
                
                # Calculate rainfall intensity
                rainfall_intensity = 200 + 100 * category + 50 * np.random.normal()  # mm/day
                
                # Calculate overall impact index (0-1 scale)
                impact = min(1.0, 0.05 * category + 0.1 * (surge_height / 3) + 0.02 * (rainfall_intensity / 300))
                
                # Create cyclone event
                cyclone = {
                    'type': 'cyclone',
                    'year': year,
                    'month': month,
                    'category': category,
                    'wind_speed': wind_speed,
                    'surge_height': surge_height,
                    'rainfall_intensity': rainfall_intensity,
                    'location': location,
                    'impact': impact
                }
                
                cyclones.append(cyclone)
        
        # Update current impact
        if cyclones:
            self.current_impacts['cyclone'] = max(event['impact'] for event in cyclones)
        else:
            self.current_impacts['cyclone'] = 0.0
            
        return cyclones
    
    def generate_floods(self, year, temperature_anomaly, monsoon_intensity, water_stress):
        """
        Generate flood events for the year based on climate conditions.
        
        Args:
            year (int): Current simulation year
            temperature_anomaly (float): Temperature anomaly in degrees C
            monsoon_intensity (float): Monsoon intensity factor
            water_stress (float): Water stress index
            
        Returns:
            list: Flood events for the year
        """
        # Climate change effects on flood frequency and intensity
        # Temperature effect: Higher temperatures intensify hydrological cycle
        temp_effect = 0.1 * (temperature_anomaly - 0.8)  # Normalized to 0.8°C reference
        
        # Monsoon effect: Stronger monsoons increase flood probability and intensity
        monsoon_effect = 0.3 * (monsoon_intensity - 1.0)  # Normalized to 1.0 reference
        
        # Update flood parameters with climate effects
        current_probability = min(1.0, self.flood_annual_probability * (1 + temp_effect + monsoon_effect))
        current_intensity = self.flood_intensity_scale * (1 + temp_effect + monsoon_effect)
        
        # Determine if a major flood occurs this year
        major_flood_occurs = np.random.random() < current_probability
        
        floods = []
        if major_flood_occurs:
            # Determine timing (month)
            # Floods typically occur during monsoon season (Jun-Sep)
            month = np.random.choice(['june', 'july', 'august', 'september'], p=[0.1, 0.3, 0.4, 0.2])
            
            # Determine flood type
            flood_type = np.random.choice(['flash', 'riverine', 'coastal'], p=[0.2, 0.7, 0.1])
            
            # Determine affected regions
            if flood_type == 'flash':
                # Flash floods most common in northeast and hilly areas
                regions = np.random.choice(['northeast', 'southeast', 'north-central'], size=1 + np.random.binomial(2, 0.5))
            elif flood_type == 'riverine':
                # Riverine floods affect river basins, especially central and northern regions
                regions = np.random.choice(['north-central', 'central', 'northwest', 'northeast', 'southwest'], 
                                          size=2 + np.random.binomial(3, 0.6))
            else:  # coastal
                # Coastal floods mainly affect southern coastal areas
                regions = np.random.choice(['southwest', 'south-central', 'southeast'], size=1 + np.random.binomial(2, 0.7))
            
            # Determine intensity and characteristics
            # Using gamma distribution for more realistic flood intensity
            raw_intensity = gamma.rvs(2.0, scale=0.2 * current_intensity)
            intensity = min(1.0, raw_intensity)
            
            # Calculate flood extent (% of land area)
            if flood_type == 'flash':
                extent = 0.05 + 0.1 * intensity
            elif flood_type == 'riverine':
                extent = 0.15 + 0.3 * intensity
            else:  # coastal
                extent = 0.05 + 0.15 * intensity
            
            # Calculate flood duration (days)
            if flood_type == 'flash':
                duration = 3 + 7 * intensity
            elif flood_type == 'riverine':
                duration = 10 + 40 * intensity
            else:  # coastal
                duration = 5 + 15 * intensity
            
            # Calculate flood depth (meters)
            depth = 0.5 + 2.0 * intensity + 0.5 * np.random.normal()
            depth = max(0.2, depth)
            
            # Calculate overall impact index (0-1 scale)
            impact = min(1.0, 0.3 * intensity + 0.2 * (extent / 0.3) + 0.1 * (duration / 30))
            
            # Create flood event
            flood = {
                'type': 'flood',
                'subtype': flood_type,
                'year': year,
                'month': month,
                'intensity': intensity,
                'extent': extent,
                'duration': duration,
                'depth': depth,
                'regions': list(regions),
                'impact': impact
            }
            
            floods.append(flood)
            
            # Update current impact
            self.current_impacts['flood'] = impact
        else:
            # Minor flood events may still occur
            # These have smaller impacts but happen more frequently
            num_minor_floods = np.random.binomial(3, 0.4)
            
            for i in range(num_minor_floods):
                month = np.random.choice(['june', 'july', 'august', 'september'], p=[0.2, 0.3, 0.3, 0.2])
                flood_type = np.random.choice(['flash', 'riverine', 'coastal'], p=[0.4, 0.5, 0.1])
                
                # Characteristics of minor floods
                intensity = 0.1 + 0.2 * np.random.random()
                extent = 0.02 + 0.05 * intensity
                duration = 2 + 5 * intensity
                depth = 0.2 + 0.5 * intensity
                
                impact = min(0.3, 0.1 * intensity + 0.05 * (extent / 0.05) + 0.02 * (duration / 5))
                
                if flood_type == 'flash':
                    regions = np.random.choice(['northeast', 'southeast', 'north-central'], size=1)
                elif flood_type == 'riverine':
                    regions = np.random.choice(['north-central', 'central', 'northwest', 'northeast'], size=1)
                else:  # coastal
                    regions = np.random.choice(['southwest', 'south-central', 'southeast'], size=1)
                
                minor_flood = {
                    'type': 'flood',
                    'subtype': flood_type,
                    'year': year,
                    'month': month,
                    'intensity': intensity,
                    'extent': extent,
                    'duration': duration,
                    'depth': depth,
                    'regions': list(regions),
                    'impact': impact
                }
                
                floods.append(minor_flood)
            
            # Update current impact (max of minor floods)
            if floods:
                self.current_impacts['flood'] = max(event['impact'] for event in floods)
            else:
                self.current_impacts['flood'] = 0.0
                
        return floods
    
    def generate_droughts(self, year, temperature_anomaly, water_stress):
        """
        Generate drought events for the year based on climate conditions.
        
        Args:
            year (int): Current simulation year
            temperature_anomaly (float): Temperature anomaly in degrees C
            water_stress (float): Water stress index
            
        Returns:
            list: Drought events for the year
        """
        # Climate change effects on drought frequency and intensity
        # Temperature effect: Higher temperatures increase evaporation and drought risk
        temp_effect = 0.2 * (temperature_anomaly - 0.8)  # Normalized to 0.8°C reference
        
        # Water stress effect: Higher baseline stress increases drought vulnerability
        stress_effect = 0.3 * (water_stress - 0.4)  # Normalized to 0.4 reference
        
        # Update drought parameters with climate effects
        current_probability = min(1.0, self.drought_annual_probability * (1 + temp_effect + stress_effect))
        current_intensity = self.drought_intensity_scale * (1 + temp_effect + stress_effect)
        
        # Determine if a drought occurs this year
        drought_occurs = np.random.random() < current_probability
        
        droughts = []
        if drought_occurs:
            # Determine timing (season)
            # Droughts most common in dry season (Nov-Apr)
            season = np.random.choice(['winter', 'pre-monsoon', 'monsoon'], p=[0.5, 0.4, 0.1])
            
            if season == 'winter':
                months = ['november', 'december', 'january', 'february']
            elif season == 'pre-monsoon':
                months = ['march', 'april', 'may']
            else:
                months = ['june', 'july', 'august', 'september']
            
            # Determine affected regions
            # Northwestern and central regions are more drought-prone
            num_regions = 1 + np.random.binomial(3, 0.6)
            regions = np.random.choice(['northwest', 'north-central', 'central', 'southwest', 'northeast'], 
                                      size=num_regions, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            
            # Determine intensity and characteristics
            # Using gamma distribution for realistic drought intensity
            raw_intensity = gamma.rvs(2.0, scale=0.2 * current_intensity)
            intensity = min(1.0, raw_intensity)
            
            # Calculate drought duration (months)
            duration = 2 + 6 * intensity
            
            # Calculate rainfall deficit (%)
            rainfall_deficit = 30 + 40 * intensity
            
            # Calculate severity index (0-1 scale)
            severity = 0.3 + 0.7 * intensity
            
            # Calculate agricultural impact (% yield reduction)
            ag_impact = 10 + 50 * intensity
            
            # Calculate overall impact index (0-1 scale)
            impact = min(1.0, 0.4 * intensity + 0.2 * (duration / 6) + 0.1 * (rainfall_deficit / 50))
            
            # Create drought event
            drought = {
                'type': 'drought',
                'year': year,
                'season': season,
                'months': months,
                'intensity': intensity,
                'duration': duration,
                'rainfall_deficit': rainfall_deficit,
                'severity': severity,
                'agricultural_impact': ag_impact,
                'regions': list(regions),
                'impact': impact
            }
            
            droughts.append(drought)
            
            # Update current impact
            self.current_impacts['drought'] = impact
        else:
            self.current_impacts['drought'] = 0.0
            
        return droughts
    
    def generate_other_disasters(self, year, monsoon_intensity, temperature_anomaly):
        """
        Generate other disaster events like landslides and river erosion.
        
        Args:
            year (int): Current simulation year
            monsoon_intensity (float): Monsoon intensity factor
            temperature_anomaly (float): Temperature anomaly in degrees C
            
        Returns:
            list: Other disaster events for the year
        """
        disasters = []
        
        # Climate effects
        monsoon_effect = 0.3 * (monsoon_intensity - 1.0)
        temp_effect = 0.1 * (temperature_anomaly - 0.8)
        
        # 1. Riverbank erosion
        # Erosion is closely linked to river discharge and flood intensity
        erosion_probability = 0.9  # High probability every year
        erosion_intensity = self.riverbank_erosion_rate * (1 + monsoon_effect)
        
        if np.random.random() < erosion_probability:
            # Riverbank erosion occurs
            raw_intensity = gamma.rvs(2.0, scale=0.15 * erosion_intensity)
            intensity = min(1.0, raw_intensity)
            
            # Determine affected rivers
            rivers = np.random.choice(['brahmaputra', 'ganges', 'meghna', 'padma', 'other'], 
                                     size=1 + np.random.binomial(2, 0.6))
            
            # Calculate land loss (square km)
            land_loss = 10 + 40 * intensity
            
            # Calculate people displaced (thousands)
            people_displaced = 5 + 25 * intensity
            
            # Calculate overall impact
            impact = min(0.8, 0.2 * intensity + 0.1 * (land_loss / 30) + 0.1 * (people_displaced / 20))
            
            erosion = {
                'type': 'erosion',
                'year': year,
                'intensity': intensity,
                'rivers': list(rivers),
                'land_loss': land_loss,
                'people_displaced': people_displaced,
                'impact': impact
            }
            
            disasters.append(erosion)
            self.current_impacts['erosion'] = impact
        else:
            self.current_impacts['erosion'] = 0.0
        
        # 2. Landslides
        # Landslides occur mainly in hilly regions (Chittagong, Sylhet) and are triggered by heavy rainfall
        landslide_probability = self.landslide_probability * (1 + monsoon_effect)
        
        if np.random.random() < landslide_probability:
            # Landslide occurs
            month = np.random.choice(['june', 'july', 'august', 'september'], p=[0.2, 0.3, 0.3, 0.2])
            
            # Determine intensity
            raw_intensity = gamma.rvs(1.5, scale=0.2)
            intensity = min(1.0, raw_intensity)
            
            # Determine location
            location = np.random.choice(['chittagong_hills', 'sylhet_hills', 'other_hills'], 
                                       p=[0.6, 0.3, 0.1])
            
            # Calculate casualties and damages
            casualties = np.random.poisson(5 * intensity)
            homes_damaged = np.random.poisson(50 * intensity)
            
            # Calculate overall impact
            impact = min(0.8, 0.2 * intensity + 0.03 * casualties + 0.01 * (homes_damaged / 10))
            
            landslide = {
                'type': 'landslide',
                'year': year,
                'month': month,
                'intensity': intensity,
                'location': location,
                'casualties': casualties,
                'homes_damaged': homes_damaged,
                'impact': impact
            }
            
            disasters.append(landslide)
            self.current_impacts['landslide'] = impact
        else:
            self.current_impacts['landslide'] = 0.0
            
        return disasters
    
    def generate_events(self, year, temperature_anomaly, sea_level_rise, monsoon_intensity, water_stress):
        """
        Generate all disaster events for the year.
        
        Args:
            year (int): Current simulation year
            temperature_anomaly (float): Temperature anomaly in degrees C
            sea_level_rise (float): Sea level rise in meters
            monsoon_intensity (float): Monsoon intensity factor
            water_stress (float): Water stress index
            
        Returns:
            list: All disaster events for the year
        """
        # Generate different types of disasters
        cyclones = self.generate_cyclones(year, temperature_anomaly, sea_level_rise)
        floods = self.generate_floods(year, temperature_anomaly, monsoon_intensity, water_stress)
        droughts = self.generate_droughts(year, temperature_anomaly, water_stress)
        others = self.generate_other_disasters(year, monsoon_intensity, temperature_anomaly)
        
        # Combine all events
        all_events = cyclones + floods + droughts + others
        
        # Add to historical record
        self.historical_events.extend(all_events)
        
        return all_events
    
    def calculate_overall_impact(self):
        """
        Calculate overall disaster impact for the current year.
        
        Returns:
            float: Overall disaster impact (0-1 scale)
        """
        # Weights for different disaster types
        weights = {
            'cyclone': 0.3,
            'flood': 0.3,
            'drought': 0.2,
            'erosion': 0.1,
            'landslide': 0.1
        }
        
        # Calculate weighted impact
        overall_impact = sum(weights[disaster] * impact for disaster, impact in self.current_impacts.items())
        
        return overall_impact
    
    def get_historical_summary(self, years=5):
        """
        Get summary of historical disaster events for recent years.
        
        Args:
            years (int): Number of years to summarize
            
        Returns:
            dict: Summary of historical disasters
        """
        if not self.historical_events:
            return {'summary': 'No historical data available'}
        
        # Filter recent events
        current_year = max(event['year'] for event in self.historical_events)
        recent_events = [e for e in self.historical_events if e['year'] > current_year - years]
        
        # Count by type
        counts = {}
        for event in recent_events:
            event_type = event['type']
            counts[event_type] = counts.get(event_type, 0) + 1
        
        # Calculate average impacts by type
        impacts = {}
        for event_type in counts.keys():
            type_events = [e for e in recent_events if e['type'] == event_type]
            average_impact = sum(e['impact'] for e in type_events) / len(type_events)
            impacts[event_type] = average_impact
        
        # Find most severe event
        if recent_events:
            most_severe = max(recent_events, key=lambda e: e['impact'])
        else:
            most_severe = None
        
        # Generate summary
        summary = {
            'period': f"{current_year - years + 1}-{current_year}",
            'event_counts': counts,
            'average_impacts': impacts,
            'most_severe_event': most_severe,
            'total_events': len(recent_events)
        }
        
        return summary
