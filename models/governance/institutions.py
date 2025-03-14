#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Governance system model for Bangladesh simulation.
This module implements institutional effectiveness, corruption, NGO intervention,
gender disparity metrics, and political stability factors.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, beta


class GovernanceSystem:
    """
    Governance system model representing Bangladesh's institutional environment
    and social governance factors.
    """
    
    def __init__(self, config, data_loader):
        """
        Initialize the governance system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the governance system
            data_loader (DataLoader): Data loading utility for governance data
        """
        self.config = config
        self.data_loader = data_loader
        
        # Load initial governance data
        self.governance_data = data_loader.load_governance_data()
        
        # Set up time-related variables
        self.current_year = config.get('start_year', 2023)
        self.time_step = config.get('time_step', 1.0)
        self.base_year = config.get('base_year', 2000)
        
        # Institutional effectiveness parameters
        self.institutional_effectiveness = self.governance_data.get('institutional_effectiveness', 0.45)  # 0-1 scale
        self.regulatory_quality = self.governance_data.get('regulatory_quality', 0.4)  # 0-1 scale
        self.rule_of_law = self.governance_data.get('rule_of_law', 0.35)  # 0-1 scale
        self.government_effectiveness = self.governance_data.get('government_effectiveness', 0.4)  # 0-1 scale
        
        # Corruption parameters
        self.corruption_index = self.governance_data.get('corruption_index', 0.65)  # 0-1 scale (higher means more corrupt)
        self.corruption_by_sector = self.governance_data.get('corruption_by_sector', {
            'public_administration': 0.7,
            'judiciary': 0.6,
            'law_enforcement': 0.65,
            'education': 0.5,
            'healthcare': 0.55,
            'utilities': 0.6,
            'private_sector': 0.5
        })
        
        # NGO and civil society parameters
        self.ngo_density = self.governance_data.get('ngo_density', 15.0)  # NGOs per 100,000 people
        self.ngo_effectiveness = self.governance_data.get('ngo_effectiveness', 0.6)  # 0-1 scale
        self.civil_society_strength = self.governance_data.get('civil_society_strength', 0.5)  # 0-1 scale
        
        # Gender disparity parameters
        self.gender_inequality_index = self.governance_data.get('gender_inequality_index', 0.55)  # 0-1 scale
        self.female_labor_participation = self.governance_data.get('female_labor_participation', 0.36)  # % of women in labor force
        self.female_education_ratio = self.governance_data.get('female_education_ratio', 0.85)  # Female/male education ratio
        self.women_in_parliament = self.governance_data.get('women_in_parliament', 0.21)  # % of parliament seats held by women
        
        # Political stability parameters
        self.political_stability = self.governance_data.get('political_stability', 0.45)  # 0-1 scale
        self.democracy_index = self.governance_data.get('democracy_index', 0.5)  # 0-1 scale
        self.voice_accountability = self.governance_data.get('voice_accountability', 0.4)  # 0-1 scale
        self.political_violence = self.governance_data.get('political_violence', 0.3)  # 0-1 scale
        
        # Investment parameters
        self.infrastructure_investment = self.governance_data.get('infrastructure_investment', 0.03)
        self.education_investment = self.governance_data.get('education_investment', 0.025)
        self.health_investment = self.governance_data.get('health_investment', 0.02)
        
        # Education system parameters
        self.education_effectiveness = self.governance_data.get('education_effectiveness', 0.5)  # 0-1 scale
        self.teacher_student_ratio = self.governance_data.get('teacher_student_ratio', 40.0)  # students per teacher
        
        # Healthcare system parameters
        self.healthcare_accessibility = self.governance_data.get('healthcare_accessibility', 0.6)  # 0-1 scale
        self.physicians_density = self.governance_data.get('physicians_density', 0.6)  # per 1000 people
        
        # Social safety net parameters
        self.social_safety_coverage = self.governance_data.get('social_safety_coverage', 0.25)  # % of population covered
        self.welfare_spending = self.governance_data.get('welfare_spending', 0.01)  # % of GDP
        
        # Regional governance variation
        self.regions = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh']
        self.regional_governance = {}
        for region in self.regions:
            self.regional_governance[region] = {
                'institutional_effectiveness': self.institutional_effectiveness,
                'corruption_index': self.corruption_index,
                'ngo_density': self.ngo_density
            }
            
            # Add random variation by region (±15%)
            for key in self.regional_governance[region]:
                variation = 1.0 + np.random.uniform(-0.15, 0.15)
                if key == 'corruption_index':
                    # Dhaka tends to have higher corruption
                    if region == 'Dhaka':
                        variation += 0.1
                    # Remote areas might have less oversight
                    if region in ['Sylhet', 'Rangpur']:
                        variation += 0.05
                self.regional_governance[region][key] *= variation
                
                # Ensure values stay in reasonable range
                if key != 'ngo_density':  # ngo_density isn't bounded by 0-1
                    self.regional_governance[region][key] = max(0.1, min(0.9, self.regional_governance[region][key]))
        
        print("Governance system initialized")
    
    def update_institutional_effectiveness(self, year, economic_system):
        """
        Update institutional effectiveness based on economic development,
        policy changes, and governance factors.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            
        Returns:
            dict: Updated institutional effectiveness parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'institutional_effectiveness': self.institutional_effectiveness}
            
        # Economic influences on institutional development
        gdp_per_capita = economic_system.gdp_per_capita if hasattr(economic_system, 'gdp_per_capita') else 2500
        gdp_growth = economic_system.gdp_growth if hasattr(economic_system, 'gdp_growth') else 0.05
        
        # Economic development effect
        # Higher income tends to correlate with better institutions
        economic_factor = 0.02 * np.log(gdp_per_capita / 2500)
        
        # Diminishing returns on economic development for institutional improvement
        if gdp_per_capita > 5000:
            economic_factor *= 0.7
            
        # Growth effect - sustained growth can help institutional development
        growth_factor = 0.01 * (gdp_growth - 0.04)  # baseline of 4% growth
        
        # Institutional momentum (existing institutions tend to persist)
        # Low corruption helps institutional development 
        corruption_factor = -0.03 * (self.corruption_index - 0.5)
        
        # Civil society pressure for better institutions
        civil_society_factor = 0.02 * (self.civil_society_strength - 0.5)
        
        # Random shock factor (political changes, scandals, reforms)
        # Using beta distribution for asymmetric shocks
        random_factor = 0.02 * (beta.rvs(2, 5) - 0.25)
        
        # Combined annual change
        annual_change = (
            economic_factor +
            growth_factor +
            corruption_factor + 
            civil_society_factor +
            random_factor
        )
        
        # Apply change over time period
        self.institutional_effectiveness = min(0.95, max(0.1, 
            self.institutional_effectiveness + annual_change * time_delta))
            
        # Update related metrics
        self.government_effectiveness = min(0.95, max(0.1,
            0.7 * self.institutional_effectiveness + 0.3 * self.government_effectiveness))
            
        self.regulatory_quality = min(0.95, max(0.1,
            0.6 * self.institutional_effectiveness + 0.4 * self.regulatory_quality))
            
        self.rule_of_law = min(0.95, max(0.1,
            0.5 * self.institutional_effectiveness + 0.5 * self.rule_of_law))
            
        self.current_year = year
        
        return {
            'institutional_effectiveness': self.institutional_effectiveness,
            'government_effectiveness': self.government_effectiveness,
            'regulatory_quality': self.regulatory_quality,
            'rule_of_law': self.rule_of_law,
            'annual_change': annual_change
        }
    
    def update_corruption(self, year, economic_system, demographic_system):
        """
        Update corruption levels based on economic development, 
        institutional effectiveness, and other factors.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            demographic_system (DemographicSystem): Demographic system state
            
        Returns:
            dict: Updated corruption parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'corruption_index': self.corruption_index}
            
        # Institutional factors
        # Better institutions reduce corruption
        institutional_factor = -0.03 * (self.institutional_effectiveness - 0.45)
        
        # Economic factors
        gdp_per_capita = economic_system.gdp_per_capita if hasattr(economic_system, 'gdp_per_capita') else 2500
        economic_inequality = economic_system.gini_coefficient if hasattr(economic_system, 'gini_coefficient') else 0.4
        
        # Higher income can reduce corruption over time
        economic_factor = -0.01 * np.log(gdp_per_capita / 2500)
        
        # Higher inequality tends to correlate with higher corruption
        inequality_factor = 0.02 * (economic_inequality - 0.4)
        
        # Demographic factors - education and urbanization
        education_levels = demographic_system.education_distribution if hasattr(demographic_system, 'education_distribution') else {
            'tertiary': 0.1
        }
        
        # Higher education tends to reduce corruption
        education_factor = -0.03 * (education_levels.get('tertiary', 0.1) - 0.1)
        
        # Civil society pressure
        civil_society_factor = -0.02 * (self.civil_society_strength - 0.5)
        
        # Random shock factor (scandals, reforms, anti-corruption drives)
        random_factor = 0.03 * (np.random.beta(2, 2) - 0.5)
        
        # Combined annual change
        annual_change = (
            institutional_factor +
            economic_factor +
            inequality_factor + 
            education_factor +
            civil_society_factor +
            random_factor
        )
        
        # Apply change over time period
        self.corruption_index = min(0.95, max(0.05, 
            self.corruption_index + annual_change * time_delta))
            
        # Update sectoral corruption
        for sector in self.corruption_by_sector:
            # Each sector responds differently to overall corruption changes
            if sector == 'public_administration':
                # Public administration corruption closely follows overall corruption
                sector_change = annual_change * 1.2
            elif sector in ['judiciary', 'law_enforcement']:
                # These sectors tend to be more resistant to change
                sector_change = annual_change * 0.7
            elif sector == 'education':
                # Education corruption more responsive to education factors
                sector_change = annual_change * 0.8 - 0.01 * education_levels.get('tertiary', 0.1)
            elif sector == 'private_sector':
                # Private sector corruption more responsive to economic factors
                sector_change = annual_change * 0.9 - 0.01 * np.log(gdp_per_capita / 2500)
            else:
                # Default change rate
                sector_change = annual_change
                
            self.corruption_by_sector[sector] = min(0.95, max(0.05,
                self.corruption_by_sector[sector] + sector_change * time_delta))
        
        return {
            'corruption_index': self.corruption_index,
            'corruption_by_sector': self.corruption_by_sector,
            'annual_change': annual_change
        }
    
    def update_ngo_activity(self, year, demographic_system, environmental_system):
        """
        Update NGO activity and effectiveness based on demographic and environmental factors.
        
        Args:
            year (int): Current simulation year
            demographic_system (DemographicSystem): Demographic system state
            environmental_system (EnvironmentalSystem): Environmental system state
            
        Returns:
            dict: Updated NGO parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'ngo_density': self.ngo_density}
            
        # Get environmental stress factors
        disaster_impact = environmental_system.calculate_overall_impact() if hasattr(environmental_system, 'calculate_overall_impact') else 0.1
        
        # NGO response to environmental stress
        # More disasters tend to attract more NGOs
        environment_factor = 0.5 * disaster_impact
        
        # Demographic factors
        population_density = demographic_system.urbanization_rate if hasattr(demographic_system, 'urbanization_rate') else 0.35
        poverty_rate = 0.3  # Could be obtained from economic system in future
        
        # Higher population density makes it easier for NGOs to operate
        density_factor = 0.2 * (population_density - 0.35)
        
        # Higher poverty rates attract more NGO activity
        poverty_factor = 0.3 * (poverty_rate - 0.3)
        
        # Political openness affects NGO operations
        political_factor = 0.3 * (self.voice_accountability - 0.4)
        
        # Random variation (funding changes, international focus)
        random_factor = 0.2 * (np.random.normal(0, 1))
        
        # Combined annual change in NGO density
        annual_change = (
            environment_factor +
            density_factor +
            poverty_factor + 
            political_factor +
            random_factor
        )
        
        # Apply change to NGO density
        self.ngo_density = max(5.0, self.ngo_density + annual_change * time_delta)
        
        # Update NGO effectiveness
        # Effectiveness influenced by institutional environment and density
        # Too many NGOs can lead to coordination problems
        density_effect = -0.01 * max(0, (self.ngo_density - 20) / 10)  # Diminishing returns after 20 NGOs per 100k
        
        institutional_effect = 0.02 * (self.institutional_effectiveness - 0.45)
        
        self.ngo_effectiveness = min(0.9, max(0.3,
            self.ngo_effectiveness + (density_effect + institutional_effect) * time_delta))
            
        # Update civil society strength
        education_effect = 0.02 * (demographic_system.literacy_rate - 0.74) if hasattr(demographic_system, 'literacy_rate') else 0
        
        self.civil_society_strength = min(0.9, max(0.2,
            0.7 * self.civil_society_strength + 0.2 * self.ngo_effectiveness + 0.1 + education_effect))
        
        return {
            'ngo_density': self.ngo_density,
            'ngo_effectiveness': self.ngo_effectiveness,
            'civil_society_strength': self.civil_society_strength
        }
    
    def update_gender_disparity(self, year, economic_system, demographic_system):
        """
        Update gender disparity metrics based on economic, demographic,
        and political factors.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            demographic_system (DemographicSystem): Demographic system state
            
        Returns:
            dict: Updated gender disparity parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'gender_inequality_index': self.gender_inequality_index}
            
        # Economic development factors
        gdp_per_capita = economic_system.gdp_per_capita if hasattr(economic_system, 'gdp_per_capita') else 2500
        
        # Higher income tends to correlate with lower gender inequality
        economic_factor = -0.01 * np.log(gdp_per_capita / 2500)
        
        # Education factors
        female_education = demographic_system.female_education_ratio if hasattr(demographic_system, 'female_education_ratio') else 0.85
        
        # Higher female education reduces gender inequality
        education_factor = -0.03 * (female_education - 0.85)
        
        # Policy and institutional factors
        # Better institutions tend to promote gender equality
        institutional_factor = -0.02 * (self.institutional_effectiveness - 0.45)
        
        # NGOs often focus on gender issues
        ngo_factor = -0.01 * (self.ngo_effectiveness - 0.6)
        
        # Political representation effect
        # More women in parliament tends to improve gender equality policies
        political_factor = -0.02 * (self.women_in_parliament - 0.21)
        
        # Cultural and religious factors (simplified)
        cultural_inertia = 0.01  # Resistance to change in gender norms
        
        # Random variation (international initiatives, campaigns)
        random_factor = 0.01 * (np.random.normal(0, 1))
        
        # Combined annual change in gender inequality
        annual_change = (
            economic_factor +
            education_factor +
            institutional_factor + 
            ngo_factor +
            political_factor +
            cultural_inertia +
            random_factor
        )
        
        # Apply change to gender inequality index
        self.gender_inequality_index = min(0.9, max(0.1, 
            self.gender_inequality_index + annual_change * time_delta))
            
        # Update female labor participation 
        # Affected by gender inequality and economic opportunities
        labor_demand = economic_system.female_labor_demand if hasattr(economic_system, 'female_labor_demand') else 0.4
        
        labor_change = (
            -0.02 * (self.gender_inequality_index - 0.55) +  # Lower inequality increases participation
            0.01 * (labor_demand - 0.4)  # Higher demand increases participation
        )
        
        self.female_labor_participation = min(0.8, max(0.3,
            self.female_labor_participation + labor_change * time_delta))
            
        # Update female education ratio
        # Generally improves over time with reduced inequality
        education_change = -0.01 * (self.gender_inequality_index - 0.55)
        
        self.female_education_ratio = min(1.1, max(0.7,  # Can exceed 1.0 as happens in some countries
            self.female_education_ratio + education_change * time_delta))
            
        # Update women in parliament
        # Tends to change more slowly due to electoral cycles
        if year % 5 == 0:  # Election years (simplified)
            election_factor = 0.03 * (np.random.beta(2, 2) - 0.3) - 0.02 * (self.gender_inequality_index - 0.55)
            self.women_in_parliament = min(0.5, max(0.1,
                self.women_in_parliament + election_factor))
        
        return {
            'gender_inequality_index': self.gender_inequality_index,
            'female_labor_participation': self.female_labor_participation,
            'female_education_ratio': self.female_education_ratio,
            'women_in_parliament': self.women_in_parliament
        }
    
    def update_political_stability(self, year, economic_system, demographic_system, environmental_system):
        """
        Update political stability metrics based on economic, demographic,
        environmental, and governance factors.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            demographic_system (DemographicSystem): Demographic system state
            environmental_system (EnvironmentalSystem): Environmental system state
            
        Returns:
            dict: Updated political stability parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'political_stability': self.political_stability}
            
        # Economic factors affecting stability
        gdp_per_capita = economic_system.gdp_per_capita if hasattr(economic_system, 'gdp_per_capita') else 2500
        gdp_growth = economic_system.gdp_growth if hasattr(economic_system, 'gdp_growth') else 0.05
        unemployment = economic_system.unemployment_rate if hasattr(economic_system, 'unemployment_rate') else 0.12
        gini = economic_system.gini_coefficient if hasattr(economic_system, 'gini_coefficient') else 0.4
        
        # Economic performance impact on stability
        # Strong economic performance increases stability
        growth_factor = 0.04 * (gdp_growth - 0.04)  # 4% growth as baseline
        
        # High unemployment decreases stability
        unemployment_factor = -0.03 * (unemployment - 0.12)
        
        # High inequality decreases stability
        inequality_factor = -0.02 * (gini - 0.4)
        
        # Environmental stress
        disaster_impact = environmental_system.calculate_overall_impact() if hasattr(environmental_system, 'calculate_overall_impact') else 0.1
        
        # Severe environmental stress can decrease stability
        environment_factor = -0.05 * disaster_impact
        
        # Institutional factors
        # Better institutions generally promote stability
        institutional_factor = 0.04 * (self.institutional_effectiveness - 0.45)
        
        # Higher corruption decreases stability 
        corruption_factor = -0.03 * (self.corruption_index - 0.65)
        
        # Demographic pressures
        youth_bulge = 0.3  # Could be calculated from demographic_system
        
        # Higher youth unemployment often correlates with instability
        youth_factor = -0.02 * youth_bulge
        
        # Random shocks (political events, international relations)
        # Using beta distribution to allow for occasional larger negative shocks
        is_election_year = (year % 5 == 0)
        election_volatility = 0.04 if is_election_year else 0.0
        
        # Elections can cause temporary destabilization
        random_factor = (np.random.beta(2, 5) - 0.3) * (0.03 + election_volatility)
        
        # Combined annual change in political stability
        annual_change = (
            growth_factor +
            unemployment_factor +
            inequality_factor + 
            environment_factor +
            institutional_factor +
            corruption_factor +
            youth_factor +
            random_factor
        )
        
        # Apply change to political stability index
        self.political_stability = min(0.95, max(0.1, 
            self.political_stability + annual_change * time_delta))
            
        # Update related metrics
        
        # Democracy index updates
        # Tends to follow political stability but with additional factors
        democracy_change = (
            0.3 * annual_change +  # General stability factors
            0.02 * (self.civil_society_strength - 0.5) +  # Civil society effect
            0.01 * (np.log(gdp_per_capita / 2500))  # Economic development effect
        )
        
        self.democracy_index = min(0.9, max(0.2,
            self.democracy_index + democracy_change * time_delta))
            
        # Voice and accountability updates
        voice_change = (
            0.2 * annual_change +  # General stability factors
            0.03 * (self.civil_society_strength - 0.5) +  # Civil society has larger effect
            -0.04 * (self.corruption_index - 0.65)  # Corruption has larger negative effect
        )
        
        self.voice_accountability = min(0.9, max(0.2,
            self.voice_accountability + voice_change * time_delta))
            
        # Political violence updates
        # Opposite direction from stability
        violence_change = (
            -0.3 * annual_change +  # Opposite of stability trends
            0.05 * (np.random.beta(2, 5) - 0.3)  # Random factor
        )
        
        self.political_violence = min(0.8, max(0.1,
            self.political_violence + violence_change * time_delta))
        
        return {
            'political_stability': self.political_stability,
            'democracy_index': self.democracy_index,
            'voice_accountability': self.voice_accountability,
            'political_violence': self.political_violence,
            'is_election_year': is_election_year
        }
    
    def update_education_system(self, year, economic_system, demographic_system):
        """
        Update education system parameters based on economic factors,
        demographic changes, and governance decisions.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            demographic_system (DemographicSystem): Demographic system state
            
        Returns:
            dict: Updated education system parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'education_effectiveness': self.education_effectiveness}
            
        # Economic factors
        gdp = economic_system.gdp if hasattr(economic_system, 'gdp') else 300e9  # $300B nominal
        gdp_growth = economic_system.gdp_growth if hasattr(economic_system, 'gdp_growth') else 0.05
        
        # Investment typically increases with GDP and based on policy priorities
        # Using sigmoid function to model saturation effects in investment
        base_investment_pct = self.education_investment
        
        # Economic growth typically allows more investment
        growth_effect = 0.001 * (gdp_growth - 0.04)
        
        # Institutional effect - better governance leads to higher educational priority
        governance_effect = 0.002 * (self.institutional_effectiveness - 0.45)
        
        # Update education investment as percentage of GDP
        self.education_investment = min(0.05, max(0.015,  # Between 1.5% to 5% of GDP
            base_investment_pct + (growth_effect + governance_effect) * time_delta))
            
        # Calculate absolute investment
        education_budget = self.education_investment * gdp
        
        # Demographic pressure - student population
        student_population = demographic_system.calculate_school_age_population() if hasattr(demographic_system, 'calculate_school_age_population') else 50e6
        
        # Per-student spending affects quality
        per_student_spending = education_budget / student_population
        
        # Teacher hiring based on spending
        # Simplistic model where hiring increases with spending
        hiring_factor = 0.1 * (self.education_investment - 0.02)
        
        # Lower teacher-student ratio improves quality
        self.teacher_student_ratio = max(30, min(50,  # Between 30 and 50 students per teacher
            self.teacher_student_ratio * (1 - hiring_factor * time_delta)))
            
        # Education effectiveness updates
        # Based on investment, teacher ratio, and governance factors
        investment_effect = 0.05 * (self.education_investment - 0.02)
        teacher_effect = -0.02 * (self.teacher_student_ratio - 40) / 10
        corruption_effect = -0.03 * (self.corruption_by_sector.get('education', 0.5) - 0.5)
        
        effectiveness_change = investment_effect + teacher_effect + corruption_effect
        
        self.education_effectiveness = min(0.9, max(0.3,
            self.education_effectiveness + effectiveness_change * time_delta))
        
        return {
            'education_investment': self.education_investment,
            'education_budget': education_budget,
            'teacher_student_ratio': self.teacher_student_ratio,
            'education_effectiveness': self.education_effectiveness
        }
    
    def update_healthcare_system(self, year, economic_system, demographic_system, environmental_system):
        """
        Update healthcare system parameters based on economic, demographic,
        and environmental factors.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            demographic_system (DemographicSystem): Demographic system state
            environmental_system (EnvironmentalSystem): Environmental system state
            
        Returns:
            dict: Updated healthcare system parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'healthcare_accessibility': self.healthcare_accessibility}
            
        # Economic factors
        gdp = economic_system.gdp if hasattr(economic_system, 'gdp') else 300e9  # $300B nominal
        gdp_growth = economic_system.gdp_growth if hasattr(economic_system, 'gdp_growth') else 0.05
        
        # Similar to education, investment increases with GDP and policy priorities
        base_investment_pct = self.health_investment
        
        # Economic growth effect
        growth_effect = 0.001 * (gdp_growth - 0.04)
        
        # Demographic pressures - aging population increases healthcare needs
        aging_effect = 0.001  # Simplified effect
        
        # Governance effect
        governance_effect = 0.001 * (self.institutional_effectiveness - 0.45)
        
        # Update healthcare investment as percentage of GDP
        self.health_investment = min(0.06, max(0.02,  # Between 2% to 6% of GDP
            base_investment_pct + (growth_effect + aging_effect + governance_effect) * time_delta))
            
        # Calculate absolute investment
        healthcare_budget = self.health_investment * gdp
        
        # Population factors
        total_population = demographic_system.total_population if hasattr(demographic_system, 'total_population') else 170e6
        
        # Per-capita spending affects quality
        per_capita_spending = healthcare_budget / total_population
        
        # Physician density updates
        # More investment leads to more physicians
        hiring_factor = 0.1 * (self.health_investment - 0.02)
        
        self.physicians_density = min(2.0, max(0.5,  # Between 0.5 and 2.0 per 1000 people
            self.physicians_density * (1 + hiring_factor * time_delta)))
            
        # Healthcare accessibility updates
        # Based on investment, physician density, corruption
        investment_effect = 0.05 * (self.health_investment - 0.02)
        physician_effect = 0.03 * (self.physicians_density - 0.6)
        corruption_effect = -0.04 * (self.corruption_by_sector.get('healthcare', 0.55) - 0.55)
        
        # Environmental factors like pollution affect health outcomes
        pollution_level = environmental_system.pollution_level if hasattr(environmental_system, 'pollution_level') else 0.3
        pollution_effect = -0.02 * (pollution_level - 0.3)
        
        accessibility_change = investment_effect + physician_effect + corruption_effect + pollution_effect
        
        self.healthcare_accessibility = min(0.9, max(0.3,
            self.healthcare_accessibility + accessibility_change * time_delta))
        
        return {
            'health_investment': self.health_investment,
            'healthcare_budget': healthcare_budget,
            'physicians_density': self.physicians_density,
            'healthcare_accessibility': self.healthcare_accessibility
        }
    
    def update_social_safety_net(self, year, economic_system, demographic_system):
        """
        Update social safety net parameters based on economic and demographic factors.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            demographic_system (DemographicSystem): Demographic system state
            
        Returns:
            dict: Updated social safety net parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'social_safety_coverage': self.social_safety_coverage}
            
        # Economic factors
        gdp = economic_system.gdp if hasattr(economic_system, 'gdp') else 300e9
        gdp_growth = economic_system.gdp_growth if hasattr(economic_system, 'gdp_growth') else 0.05
        poverty_rate = economic_system.poverty_rate if hasattr(economic_system, 'poverty_rate') else 0.2
        
        # Economic growth effect
        growth_effect = 0.001 * (gdp_growth - 0.04)
        
        # Poverty pressure increases social safety net needs
        poverty_effect = 0.002 * (poverty_rate - 0.2)
        
        # Governance effect
        governance_effect = 0.001 * (self.institutional_effectiveness - 0.45)
        
        # Update social welfare spending as percentage of GDP
        self.welfare_spending = min(0.03, max(0.005,  # Between 0.5% to 3% of GDP
            self.welfare_spending + (growth_effect + poverty_effect + governance_effect) * time_delta))
            
        # Calculate absolute spending
        welfare_budget = self.welfare_spending * gdp
        
        # Coverage updates based on spending and governance efficiency
        spending_effect = 0.1 * (self.welfare_spending - 0.01)
        efficiency_effect = 0.05 * (self.institutional_effectiveness - 0.45)
        corruption_effect = -0.1 * (self.corruption_index - 0.65)
        
        coverage_change = spending_effect + efficiency_effect + corruption_effect
        
        self.social_safety_coverage = min(0.6, max(0.1,  # Between 10% and 60% of population
            self.social_safety_coverage + coverage_change * time_delta))
        
        return {
            'welfare_spending': self.welfare_spending,
            'welfare_budget': welfare_budget,
            'social_safety_coverage': self.social_safety_coverage
        }
    
    def update_regional_governance(self, year):
        """
        Update governance indicators for different regions.
        
        Args:
            year (int): Current simulation year
            
        Returns:
            dict: Updated regional governance parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'regional_governance': self.regional_governance}
        
        # National-level changes
        institutional_change = self.institutional_effectiveness - self.regional_governance['Dhaka']['institutional_effectiveness']
        corruption_change = self.corruption_index - self.regional_governance['Dhaka']['corruption_index']
        
        # Update each region
        for region in self.regions:
            # Regions tend to converge toward national average, but at different rates
            if region == 'Dhaka':
                # Capital tends to lead changes
                convergence_rate = 0.3
            elif region in ['Chittagong', 'Khulna']:
                # Major cities follow more quickly
                convergence_rate = 0.2
            else:
                # Rural areas change more slowly
                convergence_rate = 0.1
                
            # Update regional institutional effectiveness
            self.regional_governance[region]['institutional_effectiveness'] += institutional_change * convergence_rate * time_delta
            
            # Update regional corruption
            self.regional_governance[region]['corruption_index'] += corruption_change * convergence_rate * time_delta
            
            # Update regional NGO density
            # Tends to be influenced by poverty, disasters, and institutional factors
            # For now, simple update based on national trends
            ngo_change = (self.ngo_density - self.regional_governance[region]['ngo_density']) * 0.1 * time_delta
            self.regional_governance[region]['ngo_density'] += ngo_change
        
        return {'regional_governance': self.regional_governance}
    
    def get_education_impact_on_demographics(self):
        """
        Calculate education system impact on demographic factors.
        
        Returns:
            dict: Education system impact factors
        """
        # Better education reduces fertility rates and improves development
        fertility_impact = -0.2 * (self.education_effectiveness - 0.5)
        
        # Education-driven improvement in human capital
        skill_improvement = 0.3 * self.education_effectiveness
        
        # Female education impact
        gender_education_impact = 0.4 * self.female_education_ratio * self.education_effectiveness
        
        return {
            'fertility_impact': fertility_impact,
            'skill_improvement': skill_improvement,
            'gender_education_impact': gender_education_impact
        }
    
    def get_healthcare_impact_on_demographics(self):
        """
        Calculate healthcare system impact on demographic factors.
        
        Returns:
            dict: Healthcare system impact factors
        """
        # Better healthcare reduces mortality rates
        mortality_impact = -0.3 * (self.healthcare_accessibility - 0.6)
        
        # Healthcare impact on life expectancy
        life_expectancy_impact = 0.5 * self.healthcare_accessibility
        
        # Impact on productive workforce (fewer sick days)
        productivity_impact = 0.2 * self.healthcare_accessibility
        
        return {
            'mortality_impact': mortality_impact,
            'life_expectancy_impact': life_expectancy_impact,
            'productivity_impact': productivity_impact
        }
    
    def get_governance_impact_on_economy(self):
        """
        Calculate governance impact on economic factors.
        
        Returns:
            dict: Governance impact factors
        """
        # Institutional quality impact on investment climate
        investment_climate = 0.5 * self.institutional_effectiveness - 0.3 * self.corruption_index
        
        # Regulatory quality impact on business environment
        business_environment = 0.6 * self.regulatory_quality - 0.2 * self.corruption_index
        
        # Rule of law impact on contract enforcement and property rights
        property_rights = 0.7 * self.rule_of_law
        
        # Political stability impact on economic confidence
        economic_confidence = 0.4 * self.political_stability
        
        # Corruption impact on economic efficiency
        efficiency_loss = 0.5 * self.corruption_index
        
        return {
            'investment_climate': investment_climate,
            'business_environment': business_environment,
            'property_rights': property_rights,
            'economic_confidence': economic_confidence,
            'efficiency_loss': efficiency_loss
        }
    
    def step(self, year, economic_system, demographic_system, environmental_system, infrastructure_system=None):
        """
        Advance the governance system by one time step.
        
        Args:
            year (int): Current simulation year
            economic_system (EconomicSystem): Economic system state
            demographic_system (DemographicSystem): Demographic system state
            environmental_system (EnvironmentalSystem): Environmental system state
            infrastructure_system (InfrastructureSystem, optional): Infrastructure system state
            
        Returns:
            dict: Updated governance system state and impact factors
        """
        print(f"Advancing governance system to year {year}")
        
        # Update institutional effectiveness
        institutional_results = self.update_institutional_effectiveness(year, economic_system)
        
        # Update corruption
        corruption_results = self.update_corruption(year, economic_system, demographic_system)
        
        # Update NGO activity
        ngo_results = self.update_ngo_activity(year, demographic_system, environmental_system)
        
        # Update gender disparity
        gender_results = self.update_gender_disparity(year, economic_system, demographic_system)
        
        # Update political stability
        stability_results = self.update_political_stability(year, economic_system, demographic_system, environmental_system)
        
        # Update education system
        education_results = self.update_education_system(year, economic_system, demographic_system)
        
        # Update healthcare system
        healthcare_results = self.update_healthcare_system(year, economic_system, demographic_system, environmental_system)
        
        # Update social safety net
        safety_net_results = self.update_social_safety_net(year, economic_system, demographic_system)
        
        # Update regional governance
        regional_results = self.update_regional_governance(year)
        
        # Calculate impact factors for other systems
        education_impact = self.get_education_impact_on_demographics()
        healthcare_impact = self.get_healthcare_impact_on_demographics()
        economic_impact = self.get_governance_impact_on_economy()
        
        # Compile results
        results = {
            'year': year,
            'institutional_effectiveness': self.institutional_effectiveness,
            'government_effectiveness': self.government_effectiveness,
            'regulatory_quality': self.regulatory_quality,
            'rule_of_law': self.rule_of_law,
            'corruption_index': self.corruption_index,
            'corruption_by_sector': self.corruption_by_sector,
            'ngo_density': self.ngo_density,
            'ngo_effectiveness': self.ngo_effectiveness,
            'civil_society_strength': self.civil_society_strength,
            'gender_inequality_index': self.gender_inequality_index,
            'female_labor_participation': self.female_labor_participation,
            'female_education_ratio': self.female_education_ratio,
            'women_in_parliament': self.women_in_parliament,
            'political_stability': self.political_stability,
            'democracy_index': self.democracy_index,
            'voice_accountability': self.voice_accountability,
            'political_violence': self.political_violence,
            'education_investment': self.education_investment,
            'education_effectiveness': self.education_effectiveness,
            'teacher_student_ratio': self.teacher_student_ratio,
            'health_investment': self.health_investment,
            'healthcare_accessibility': self.healthcare_accessibility,
            'physicians_density': self.physicians_density,
            'social_safety_coverage': self.social_safety_coverage,
            'welfare_spending': self.welfare_spending,
            'regional_governance': self.regional_governance,
            'education_impact': education_impact,
            'healthcare_impact': healthcare_impact,
            'economic_impact': economic_impact
        }
        
        return results
