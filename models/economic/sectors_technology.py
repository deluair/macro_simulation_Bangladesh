#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technology and Informal Economy sector models for Bangladesh simulation.
These sectors complement the garment and agricultural sectors in the economic system.
"""

import numpy as np
import pandas as pd
from models.economic.sectors import BaseSector


class TechnologySector(BaseSector):
    """
    Model of Bangladesh's emerging technology and service sector.
    """
    
    def __init__(self, config, economic_data):
        """
        Initialize the technology sector.
        
        Args:
            config (dict): Technology sector configuration
            economic_data (dict): Economic data including technology sector data
        """
        super().__init__('technology', config, economic_data)
        
        # Initialize sector-specific attributes
        self.output = economic_data.get('technology_output', 10.0)  # Billion USD
        self.employment = economic_data.get('technology_employment', 500000)  # 0.5 million workers
        self.productivity = economic_data.get('technology_productivity', 1.5)  # Higher than average
        self.export_share = economic_data.get('technology_export_share', 0.25)  # 25% exported
        self.formal_share = economic_data.get('technology_formal_share', 0.85)  # 85% formal
        
        # Technology subsectors
        self.subsectors = {
            'software_services': {
                'output': economic_data.get('software_output', 4.0),
                'growth_rate': economic_data.get('software_growth', 0.15),
                'export_orientation': economic_data.get('software_export_share', 0.40),
                'employment': economic_data.get('software_employment', 200000)
            },
            'bpo': {  # Business Process Outsourcing
                'output': economic_data.get('bpo_output', 2.0),
                'growth_rate': economic_data.get('bpo_growth', 0.18),
                'export_orientation': economic_data.get('bpo_export_share', 0.60),
                'employment': economic_data.get('bpo_employment', 120000)
            },
            'digital_commerce': {
                'output': economic_data.get('ecommerce_output', 1.5),
                'growth_rate': economic_data.get('ecommerce_growth', 0.25),
                'export_orientation': economic_data.get('ecommerce_export_share', 0.05),
                'employment': economic_data.get('ecommerce_employment', 80000)
            },
            'fintech': {
                'output': economic_data.get('fintech_output', 1.0),
                'growth_rate': economic_data.get('fintech_growth', 0.20),
                'export_orientation': economic_data.get('fintech_export_share', 0.10),
                'employment': economic_data.get('fintech_employment', 40000)
            },
            'other_tech': {
                'output': economic_data.get('other_tech_output', 1.5),
                'growth_rate': economic_data.get('other_tech_growth', 0.12),
                'export_orientation': economic_data.get('other_tech_export_share', 0.15),
                'employment': economic_data.get('other_tech_employment', 60000)
            }
        }
        
        # Sector-specific attributes
        self.digital_literacy = economic_data.get('digital_literacy', 0.40)  # 40% of workforce
        self.internet_penetration = economic_data.get('internet_penetration', 0.50)  # 50% of population
        self.startup_ecosystem = economic_data.get('startup_ecosystem', 0.30)  # Ecosystem maturity (0-1)
        self.venture_capital = economic_data.get('venture_capital', 0.15)  # Venture capital availability (0-1)
        self.tech_talent_pool = economic_data.get('tech_talent_pool', 0.25)  # Talent availability (0-1)
        
        print("Technology sector initialized")
    
    def step(self, exchange_rate=None, inflation_rate=None, labor_supply=None, 
             infrastructure=None, environmental=None, financial_markets=None, governance=None):
        """
        Advance the technology sector by one time step.
        
        Args:
            exchange_rate (float): Current exchange rate
            inflation_rate (float): Current inflation rate
            labor_supply (dict): Labor market conditions
            infrastructure (dict): Infrastructure quality metrics
            environmental (dict): Environmental conditions and impacts
            financial_markets (dict): Financial market conditions
            governance (dict): Governance factors
            
        Returns:
            dict: Technology sector results after the step
        """
        # Digital infrastructure effect (critical for tech sector)
        infrastructure_effect = 0.0
        if infrastructure:
            internet_effect = 0.3 * (infrastructure.get('telecom_coverage', 0.5) - 0.5)
            electricity_effect = 0.2 * (infrastructure.get('electricity_reliability', 0.5) - 0.5)
            infrastructure_effect = internet_effect + electricity_effect
            
            # Update internet penetration based on infrastructure
            internet_growth = 0.05 + 0.2 * (infrastructure.get('telecom_investment', 0) - 0.05)
            self.internet_penetration = min(self.internet_penetration * (1 + internet_growth), 0.95)
        
        # Human capital effect
        human_capital_effect = 0.0
        if labor_supply:
            education_effect = 0.4 * (labor_supply.get('tertiary_education', 0.15) - 0.15)
            english_proficiency = 0.3 * (labor_supply.get('english_proficiency', 0.4) - 0.4)
            youth_share = 0.2 * (labor_supply.get('youth_share', 0.3) - 0.3)
            human_capital_effect = education_effect + english_proficiency + youth_share
            
            # Update digital literacy
            digital_literacy_growth = 0.03 + 0.1 * (labor_supply.get('tertiary_education', 0.15) - 0.15)
            self.digital_literacy = min(self.digital_literacy * (1 + digital_literacy_growth), 0.8)
            
            # Update tech talent pool
            talent_pool_growth = 0.05 + 0.2 * (labor_supply.get('stem_graduates', 0.1) - 0.1)
            self.tech_talent_pool = min(self.tech_talent_pool * (1 + talent_pool_growth), 0.7)
            
            # Update employment based on output growth
            employment_growth = 0.8 * self.growth_rate  # Tech creates fewer jobs per output unit
            self.employment *= (1 + employment_growth)
        
        # Financial ecosystem effect
        finance_effect = 0.0
        if financial_markets:
            formal_credit = financial_markets.get('access_rates', {}).get('urban_formal_access', 0.6)
            finance_effect = 0.15 * (formal_credit - 0.6)
            
            # Update venture capital availability
            vc_growth = 0.03 + 0.1 * (financial_markets.get('financial_development', 0.5) - 0.5)
            self.venture_capital = min(self.venture_capital * (1 + vc_growth), 0.6)
        
        # Governance and policy effect
        governance_effect = 0.0
        if governance:
            # Check if governance is a dictionary or an object
            if isinstance(governance, dict):
                digital_policy = 0.25 * (governance.get('digital_initiatives', 0.5) - 0.5)
                regulatory_burden = -0.2 * (governance.get('regulatory_burden', 0.5) - 0.5)
                
                # Update startup ecosystem based on policy
                ecosystem_growth = 0.02 + 0.15 * (governance.get('business_environment', 0.4) - 0.4)
            else:
                # Handle governance system object
                digital_policy = 0.25 * (getattr(governance, 'digital_initiatives', 0.5) - 0.5)
                regulatory_burden = -0.2 * (getattr(governance, 'regulatory_burden', 0.5) - 0.5)
                
                # Update startup ecosystem based on policy
                ecosystem_growth = 0.02 + 0.15 * (getattr(governance, 'business_environment', 0.4) - 0.4)
            
            governance_effect = digital_policy + regulatory_burden
            self.startup_ecosystem = min(self.startup_ecosystem * (1 + ecosystem_growth), 0.7)
        
        # Global technology trends effect
        global_tech_trend = 0.08  # Strong global growth in technology sector
        global_volatility = np.random.normal(0, 0.04)  # Tech sector volatility
        global_effect = global_tech_trend + global_volatility
        
        # Update productivity
        education_level = labor_supply.get('tertiary_education', 0.15) if labor_supply else 0.15
        self.update_productivity(education_level, infrastructure, technology_spillover=0.03)
        
        # Update innovation at higher rate for tech sector
        r_d_investment = 0.03  # Tech invests more in R&D
        openness = 0.7  # Tech is more globally integrated
        self.update_innovation(education_level, r_d_investment, openness)
        
        # Combine all effects to determine output growth
        self.growth_rate = (
            0.06 +  # Base growth trend (higher for tech)
            infrastructure_effect +
            human_capital_effect +
            finance_effect +
            governance_effect +
            global_effect +
            0.3 * (self.productivity - 1.5)  # Productivity effect
        )
        
        # Update output
        self.output *= (1 + self.growth_rate)
        
        # Update subsectors
        total_employment = 0
        for subsector, data in self.subsectors.items():
            # Different growth rates for different subsectors
            if subsector == 'fintech':
                subsector_growth = self.growth_rate * 1.2  # Faster than average
            elif subsector == 'digital_commerce':
                subsector_growth = self.growth_rate * 1.3  # Much faster than average
            elif subsector == 'bpo':
                subsector_growth = self.growth_rate * 0.9  # Slower than average
            else:
                subsector_growth = self.growth_rate
                
            # Update subsector data
            data['growth_rate'] = subsector_growth
            data['output'] *= (1 + subsector_growth)
            data['employment'] *= (1 + 0.8 * subsector_growth)  # Employment grows slower than output
            total_employment += data['employment']
            
            # Update export orientation
            export_change = 0.01 * (self.productivity - 1.5)
            data['export_orientation'] = min(data['export_orientation'] + export_change, 0.8)
        
        # Recalculate total output and employment from subsectors
        self.output = sum(data['output'] for data in self.subsectors.values())
        self.employment = total_employment
        
        # Calculate resource use (tech uses less physical resources)
        resource_use = self.calculate_resource_use(self.growth_rate)
        
        # Compile results
        results = {
            'output': self.output,
            'growth_rate': self.growth_rate,
            'employment': self.employment,
            'productivity': self.productivity,
            'export_share': self.export_share,
            'subsectors': self.subsectors,
            'digital_literacy': self.digital_literacy,
            'internet_penetration': self.internet_penetration,
            'startup_ecosystem': self.startup_ecosystem,
            'venture_capital': self.venture_capital,
            'tech_talent_pool': self.tech_talent_pool,
            'innovation_rate': self.innovation_rate,
            'technology_level': self.technology_level,
            'resource_use': resource_use,
            'effects': {
                'infrastructure': infrastructure_effect,
                'human_capital': human_capital_effect,
                'finance': finance_effect,
                'governance': governance_effect,
                'global': global_effect
            }
        }
        
        return results


class InformalEconomySector(BaseSector):
    """
    Model of Bangladesh's large informal economy sector.
    """
    
    def __init__(self, config, economic_data):
        """
        Initialize the informal economy sector.
        
        Args:
            config (dict): Informal sector configuration
            economic_data (dict): Economic data including informal sector data
        """
        super().__init__('informal', config, economic_data)
        
        # Initialize sector-specific attributes
        self.output = economic_data.get('informal_output', 120.0)  # Billion USD
        self.employment = economic_data.get('informal_employment', 35000000)  # 35 million workers
        self.productivity = economic_data.get('informal_productivity', 0.7)  # Lower than formal sectors
        self.export_share = economic_data.get('informal_export_share', 0.05)  # 5% exported
        self.formal_share = 0.0  # By definition
        
        # Informal subsectors
        self.subsectors = {
            'retail_trade': {
                'output': economic_data.get('informal_retail_output', 30.0),
                'employment': economic_data.get('informal_retail_employment', 8000000),
                'urban_share': economic_data.get('informal_retail_urban_share', 0.65)
            },
            'construction': {
                'output': economic_data.get('informal_construction_output', 25.0),
                'employment': economic_data.get('informal_construction_employment', 5000000),
                'urban_share': economic_data.get('informal_construction_urban_share', 0.7)
            },
            'transport': {
                'output': economic_data.get('informal_transport_output', 20.0),
                'employment': economic_data.get('informal_transport_employment', 6000000),
                'urban_share': economic_data.get('informal_transport_urban_share', 0.6)
            },
            'small_manufacturing': {
                'output': economic_data.get('informal_manufacturing_output', 15.0),
                'employment': economic_data.get('informal_manufacturing_employment', 7000000),
                'urban_share': economic_data.get('informal_manufacturing_urban_share', 0.55)
            },
            'household_services': {
                'output': economic_data.get('informal_services_output', 30.0),
                'employment': economic_data.get('informal_services_employment', 9000000),
                'urban_share': economic_data.get('informal_services_urban_share', 0.6)
            }
        }
        
        # Sector-specific attributes
        self.formalization_rate = economic_data.get('formalization_rate', 0.02)  # 2% per year
        self.urban_share = economic_data.get('informal_urban_share', 0.60)  # 60% in urban areas
        self.female_participation = economic_data.get('informal_female_share', 0.35)  # 35% female workers
        self.youth_share = economic_data.get('informal_youth_share', 0.40)  # 40% youth workers
        self.average_wage_ratio = economic_data.get('informal_wage_ratio', 0.6)  # 60% of formal wages
        
        print("Informal economy sector initialized")
    
    def step(self, exchange_rate=None, inflation_rate=None, labor_supply=None, 
             infrastructure=None, environmental=None, financial_markets=None, governance=None):
        """
        Advance the informal economy sector by one time step.
        
        Args:
            exchange_rate (float): Current exchange rate
            inflation_rate (float): Current inflation rate
            labor_supply (dict): Labor market conditions
            infrastructure (dict): Infrastructure quality metrics
            environmental (dict): Environmental conditions and impacts
            financial_markets (dict): Financial market conditions
            governance (dict): Governance factors
            
        Returns:
            dict: Informal economy sector results after the step
        """
        # Formal economy spillover effect
        formal_economy_growth = 0.06  # Assumed overall formal economy growth
        if hasattr(financial_markets, 'gdp_growth') and financial_markets.gdp_growth is not None:
            formal_economy_growth = financial_markets.gdp_growth
        
        formal_spillover = 0.5 * formal_economy_growth
        
        # Urban development effect
        urban_effect = 0.0
        if labor_supply:
            urbanization_rate = labor_supply.get('urbanization_rate', 0.03)
            urban_effect = 0.3 * urbanization_rate
            
            # Update urban share
            self.urban_share = min(self.urban_share + 0.5 * urbanization_rate, 0.8)
        
        # Formalization pressure
        formalization_pressure = 0.0
        if governance:
            # Regulatory enforcement increases formalization
            enforcement_effect = 0.05 * (governance.get('regulatory_enforcement', 0.5) - 0.5)
            tax_incentives = -0.03 * (governance.get('tax_burden', 0.5) - 0.5)
            formalization_pressure = enforcement_effect + tax_incentives
            
            # Update formalization rate
            self.formalization_rate = max(0.01, min(self.formalization_rate + 0.2 * formalization_pressure, 0.08))
        
        # Access to finance effect (lack of access keeps businesses informal)
        finance_effect = 0.0
        if financial_markets:
            microfinance_access = financial_markets.get('access_rates', {}).get('microfinance_coverage', 0.35)
            informal_credit_cost = financial_markets.get('interest_rates', {}).get('informal_lending_rate', 0.3)
            
            # Higher microfinance access increases formalization
            finance_effect = 0.1 * (microfinance_access - 0.35) - 0.05 * (informal_credit_cost - 0.3)
        
        # Environmental stress particularly affects vulnerable informal sector
        environmental_effect = 0.0
        if environmental:
            # Informal workers often most exposed to environmental hazards
            flood_effect = -0.15 * environmental.get('flood_impacts', 0)
            urban_heat = -0.1 * environmental.get('urban_heat_island', 0)
            environmental_effect = flood_effect + urban_heat
        
        # Inflation affects informal sector differently (often hit harder)
        inflation_effect = 0.0
        if inflation_rate:
            inflation_effect = -0.2 * (inflation_rate - 0.055)
        
        # Infrastructure effect
        infrastructure_effect = 0.0
        if infrastructure:
            transport_effect = 0.1 * (infrastructure.get('transport_efficiency', 0.5) - 0.5)
            urban_services = 0.1 * (infrastructure.get('urban_services', 0.5) - 0.5)
            infrastructure_effect = transport_effect + urban_services
        
        # Labor market dynamics
        labor_effect = 0.0
        if labor_supply:
            # More unemployed people enter informal sector
            unemployment_effect = 0.2 * (labor_supply.get('unemployment_rate', 0.05) - 0.05)
            youth_bulge = 0.1 * (labor_supply.get('youth_unemployment', 0.1) - 0.1)
            labor_effect = unemployment_effect + youth_bulge
            
            # Update employment (informal employment rises when formal job creation is insufficient)
            formal_job_creation = labor_supply.get('formal_job_creation', 500000)
            labor_force_growth = labor_supply.get('labor_force_growth', 0.015)
            
            # If formal job creation doesn't absorb labor force growth, informal employment increases
            labor_force = labor_supply.get('total_labor_force', 70000000)
            excess_labor = labor_force * labor_force_growth - formal_job_creation
            
            # Adjust employment based on sector growth and excess labor absorption
            employment_growth = self.growth_rate - self.formalization_rate + max(0, excess_labor / self.employment)
            self.employment *= (1 + employment_growth)
        
        # Update productivity (typically slower in informal sector)
        self.update_productivity(labor_skills=0.4, infrastructure=infrastructure)
        
        # Update innovation (much lower in informal sector)
        self.update_innovation(education_level=0.3, r_d_investment=0.005, openness=0.3)
        
        # Combine all effects to determine output growth
        self.growth_rate = (
            0.03 +  # Base growth trend
            formal_spillover +
            urban_effect +
            finance_effect +
            environmental_effect +
            infrastructure_effect +
            labor_effect +
            inflation_effect -
            self.formalization_rate +  # Loss due to formalization
            0.1 * (self.productivity - 0.7)  # Productivity effect
        )
        
        # Update output
        self.output *= (1 + self.growth_rate)
        
        # Update average wage ratio (very slow improvement)
        self.average_wage_ratio += 0.005 * self.productivity
        self.average_wage_ratio = min(max(self.average_wage_ratio, 0.4), 0.8)
        
        # Update female participation (gradually increasing)
        female_participation_change = 0.01 * (1 - self.female_participation)
        self.female_participation += female_participation_change
        
        # Update subsectors
        for subsector, data in self.subsectors.items():
            # Different growth rates for different subsectors
            if subsector == 'retail_trade':
                subsector_growth = self.growth_rate + 0.01  # Slightly faster than average
            elif subsector == 'construction':
                subsector_growth = self.growth_rate + 0.02 * urban_effect  # Affected by urbanization
            elif subsector == 'transport':
                subsector_growth = self.growth_rate + 0.02 * infrastructure_effect  # Affected by infrastructure
            else:
                subsector_growth = self.growth_rate
                
            # Update subsector data
            data['output'] *= (1 + subsector_growth)
            data['employment'] *= (1 + subsector_growth - 0.5 * self.formalization_rate)
            
            # Update urban share
            data['urban_share'] = min(data['urban_share'] + 0.3 * (self.urban_share - data['urban_share']), 0.9)
        
        # Recalculate total output and employment from subsectors
        self.output = sum(data['output'] for data in self.subsectors.values())
        self.employment = sum(data['employment'] for data in self.subsectors.values())
        
        # Calculate resource use
        resource_use = self.calculate_resource_use(self.growth_rate)
        
        # Compile results
        results = {
            'output': self.output,
            'growth_rate': self.growth_rate,
            'employment': self.employment,
            'productivity': self.productivity,
            'formalization_rate': self.formalization_rate,
            'subsectors': self.subsectors,
            'urban_share': self.urban_share,
            'female_participation': self.female_participation,
            'youth_share': self.youth_share,
            'average_wage_ratio': self.average_wage_ratio,
            'resource_use': resource_use,
            'effects': {
                'formal_spillover': formal_spillover,
                'urban': urban_effect,
                'formalization_pressure': formalization_pressure,
                'finance': finance_effect,
                'environmental': environmental_effect,
                'inflation': inflation_effect,
                'infrastructure': infrastructure_effect,
                'labor': labor_effect
            }
        }
        
        return results
