#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Economic sectors model for Bangladesh simulation.
This module implements specific economic sectors including garment manufacturing,
agriculture, technology, and informal economy.
"""

import numpy as np
import pandas as pd


class BaseSector:
    """Base class for all economic sectors."""
    
    def __init__(self, name, config, economic_data):
        """
        Initialize the base sector.
        
        Args:
            name (str): Sector name
            config (dict): Sector-specific configuration
            economic_data (dict): Economic data for initialization
        """
        self.name = name
        self.config = config
        self.economic_data = economic_data
        
        # Core sector attributes
        self.output = 0.0
        self.employment = 0
        self.productivity = 1.0
        self.growth_rate = 0.0
        self.formal_share = 0.0
        
        # Innovation and technology attributes
        self.technology_level = 0.5
        self.innovation_rate = 0.01
        
        # Resource use
        self.capital_stock = 0.0
        self.resource_intensity = 1.0
        self.energy_use = 0.0
        self.water_use = 0.0
        self.land_use = 0.0
        
    def step(self, **kwargs):
        """
        Base step method to be overridden by subclasses.
        
        Args:
            **kwargs: Various inputs from other systems
            
        Returns:
            dict: Sector results after the step
        """
        raise NotImplementedError("Subclasses must implement step method")
    
    def update_productivity(self, labor_skills=0.5, infrastructure=None, technology_spillover=0.0):
        """
        Update sector productivity based on inputs.
        
        Args:
            labor_skills (float): Labor force skill level (0-1)
            infrastructure (dict): Infrastructure quality metrics
            technology_spillover (float): Technology spillover from other sectors
            
        Returns:
            float: Updated productivity level
        """
        # Base productivity growth
        base_growth = self.config.get('base_productivity_growth', 0.02)
        
        # Skills effect
        skills_effect = 0.5 * (labor_skills - 0.5)
        
        # Infrastructure effect
        infrastructure_effect = 0.0
        if infrastructure:
            electricity_effect = 0.2 * (infrastructure.get('electricity_reliability', 0.5) - 0.5)
            transport_effect = 0.15 * (infrastructure.get('transport_efficiency', 0.5) - 0.5)
            telecom_effect = 0.1 * (infrastructure.get('telecom_coverage', 0.5) - 0.5)
            infrastructure_effect = electricity_effect + transport_effect + telecom_effect
        
        # Technology effect
        tech_effect = 0.3 * self.innovation_rate + 0.2 * technology_spillover
        
        # Random component (shocks, unmeasured factors)
        random_effect = np.random.normal(0, 0.01)
        
        # Calculate productivity growth
        productivity_growth = base_growth + skills_effect + infrastructure_effect + tech_effect + random_effect
        
        # Update productivity with diminishing returns as productivity level increases
        diminishing_factor = max(0.5, 1 - 0.3 * self.productivity)
        self.productivity *= (1 + productivity_growth * diminishing_factor)
        
        # Bound productivity to reasonable values
        self.productivity = max(min(self.productivity, 3.0), 0.3)
        
        return self.productivity
    
    def update_innovation(self, education_level=0.5, r_d_investment=0.01, openness=0.5):
        """
        Update the sector's innovation rate.
        
        Args:
            education_level (float): Education level of workforce (0-1)
            r_d_investment (float): R&D investment as share of output
            openness (float): Economic openness/trade integration (0-1)
            
        Returns:
            float: Updated innovation rate
        """
        # Base innovation trend
        base_innovation = self.config.get('base_innovation_rate', 0.01)
        
        # Education effect
        education_effect = 0.4 * (education_level - 0.5)
        
        # R&D investment effect
        rd_effect = 5.0 * (r_d_investment - 0.01)
        
        # Openness effect (technology transfer)
        openness_effect = 0.2 * (openness - 0.5)
        
        # Random component (breakthrough, ideas)
        random_effect = np.random.normal(0, 0.005)
        
        # Calculate new innovation rate
        innovation_growth = base_innovation + education_effect + rd_effect + openness_effect + random_effect
        
        # Update innovation rate
        self.innovation_rate = max(min(self.innovation_rate + innovation_growth, 0.10), 0.0)
        
        # Update technology level based on innovation
        self.technology_level = min(self.technology_level + self.innovation_rate, 1.0)
        
        return self.innovation_rate
    
    def calculate_resource_use(self, output_growth):
        """
        Calculate resource use based on output growth and efficiency.
        
        Args:
            output_growth (float): Growth in sector output
            
        Returns:
            dict: Resource use statistics
        """
        # Efficiency improvement due to technology
        efficiency_improvement = 0.3 * self.innovation_rate
        
        # Resource intensity decreases with efficiency improvements
        self.resource_intensity *= (1 - efficiency_improvement)
        
        # Calculate resource use
        self.energy_use = self.output * self.resource_intensity * self.config.get('energy_intensity', 1.0)
        self.water_use = self.output * self.resource_intensity * self.config.get('water_intensity', 1.0)
        self.land_use = self.output * self.resource_intensity * self.config.get('land_intensity', 1.0)
        
        resource_use = {
            'energy_use': self.energy_use,
            'water_use': self.water_use,
            'land_use': self.land_use,
            'resource_intensity': self.resource_intensity
        }
        
        return resource_use


class GarmentSector(BaseSector):
    """
    Model of Bangladesh's garment manufacturing sector.
    """
    
    def __init__(self, config, economic_data):
        """
        Initialize the garment sector.
        
        Args:
            config (dict): Garment sector configuration
            economic_data (dict): Economic data including garment sector data
        """
        super().__init__('garment', config, economic_data)
        
        # Initialize sector-specific attributes
        self.output = economic_data.get('garment_output', 30.0)  # Billion USD
        self.employment = economic_data.get('garment_employment', 4500000)  # 4.5 million workers
        self.productivity = economic_data.get('garment_productivity', 1.0)
        self.export_share = economic_data.get('garment_export_share', 0.85)  # 85% of output is exported
        self.formal_share = economic_data.get('garment_formal_share', 0.65)  # 65% is formal
        
        # Garment-specific attributes
        self.factories = economic_data.get('garment_factories', 4500)
        self.compliance_level = economic_data.get('garment_compliance', 0.6)  # Safety/labor compliance
        self.value_addition = economic_data.get('garment_value_addition', 0.4)  # Value addition ratio
        self.female_employment_share = economic_data.get('garment_female_share', 0.65)  # 65% female workers
        
        # Supply chain attributes
        self.raw_material_dependence = economic_data.get('garment_import_dependence', 0.8)  # 80% imported inputs
        self.backward_linkage = economic_data.get('garment_backward_linkage', 0.2)  # Domestic input sourcing
        
        print("Garment sector initialized")
    
    def step(self, exchange_rate=None, inflation_rate=None, labor_supply=None, 
             infrastructure=None, environmental=None, financial_markets=None, governance=None):
        """
        Advance the garment sector by one time step.
        
        Args:
            exchange_rate (float): Current exchange rate
            inflation_rate (float): Current inflation rate
            labor_supply (dict): Labor market conditions
            infrastructure (dict): Infrastructure quality metrics
            environmental (dict): Environmental conditions and impacts
            financial_markets (dict): Financial market conditions
            governance (dict): Governance factors
            
        Returns:
            dict: Garment sector results after the step
        """
        # Exchange rate effect on export competitiveness
        # Depreciation (higher BDT per USD) improves export competitiveness
        exchange_rate_effect = 0.0
        if exchange_rate:
            exchange_rate_effect = 0.3 * (exchange_rate / self.economic_data.get('previous_exchange_rate', exchange_rate) - 1)
        
        # Global demand effect (fluctuations in international markets)
        global_demand_trend = 0.02  # Long-term growth in global apparel demand
        demand_volatility = np.random.normal(0, 0.03)  # Short-term fluctuations
        global_demand_effect = global_demand_trend + demand_volatility
        
        # Labor market effects
        labor_effect = 0.0
        wage_growth = 0.0
        if labor_supply is not None:
            # Check if labor_supply is a dictionary or a numeric value
            if isinstance(labor_supply, dict):
                # Original behavior for dictionary input
                wage_growth = labor_supply.get('wage_growth', 0.05)
                labor_availability = labor_supply.get('garment_labor_availability', 0.7)
                skill_level = labor_supply.get('garment_skill_level', 0.5)
            else:
                # Handle numeric input (total labor force)
                # Default values based on total labor availability
                wage_growth = 0.05  # Default annual wage growth
                # Calculate availability as a function of total labor supply
                # Higher labor supply means higher labor availability
                labor_availability = min(0.9, max(0.5, 0.7 + (labor_supply / 100_000_000 - 1) * 0.1))
                skill_level = 0.5  # Default skill level
            
            # Labor effect combines availability and skills
            labor_effect = 0.2 * (labor_availability - 0.7) + 0.3 * (skill_level - 0.5)
            
            # Update employment based on output growth and productivity
            employment_growth = self.growth_rate - 0.5 * (self.productivity - 1)
            self.employment *= (1 + employment_growth)
        
        # Infrastructure effects
        infrastructure_effect = 0.0
        if infrastructure:
            electricity_effect = 0.25 * (infrastructure.get('electricity_reliability', 0.5) - 0.5)
            transport_effect = 0.15 * (infrastructure.get('transport_efficiency', 0.5) - 0.5)
            infrastructure_effect = electricity_effect + transport_effect
        
        # Environmental impacts
        environmental_effect = 0.0
        if environmental:
            flood_impact = -0.1 * environmental.get('flood_impacts', 0)
            cyclone_impact = -0.2 * environmental.get('cyclone_damage', 0)
            environmental_effect = flood_impact + cyclone_impact
        
        # Access to finance
        finance_effect = 0.0
        if financial_markets:
            credit_availability = financial_markets.get('credit_volumes', {}).get('formal_credit_volume', 100) / 100
            interest_rate = financial_markets.get('interest_rates', {}).get('formal_lending_rate', 0.09)
            finance_effect = 0.1 * (credit_availability - 1) - 0.2 * (interest_rate - 0.09)
        
        # Governance and compliance effects
        governance_effect = 0.0
        if governance:
            # Check if governance is a dictionary or an object
            if isinstance(governance, dict):
                corruption_impact = -0.15 * governance.get('corruption_index', 0.5)
                labor_regulations = -0.05 * governance.get('regulatory_burden', 0.5)
            else:
                # If governance is an object, try to get attributes or use defaults
                corruption_impact = -0.15 * getattr(governance, 'corruption_index', 0.5)
                labor_regulations = -0.05 * getattr(governance, 'regulatory_burden', 0.5)
            
            governance_effect = corruption_impact + labor_regulations
            
            # Update compliance level based on governance
            if isinstance(governance, dict):
                compliance_improvement = 0.05 * (governance.get('institutional_efficiency', 0.5) - 0.5)
            else:
                compliance_improvement = 0.05 * (getattr(governance, 'institutional_efficiency', 0.5) - 0.5)
            
            self.compliance_level = min(max(self.compliance_level + compliance_improvement, 0.3), 0.9)
        
        # Update productivity
        labor_skills = 0.5  # Default value
        if labor_supply:
            if isinstance(labor_supply, dict):
                labor_skills = labor_supply.get('garment_skill_level', 0.5)
            else:
                # If labor_supply is a float, use the default skill level
                labor_skills = 0.5
                
        self.update_productivity(labor_skills, infrastructure)
        
        # Update innovation
        education_level = 0.5  # Default value
        if labor_supply:
            if isinstance(labor_supply, dict):
                education_level = labor_supply.get('education_level', 0.5)
            else:
                # If labor_supply is a float, use the default education level
                education_level = 0.5
                
        self.update_innovation(education_level)
        
        # Combine all effects to determine output growth
        self.growth_rate = (
            0.04 +  # Base growth trend
            exchange_rate_effect +
            global_demand_effect +
            labor_effect +
            infrastructure_effect +
            environmental_effect +
            finance_effect +
            governance_effect +
            0.2 * (self.productivity - 1)  # Productivity effect
        )
        
        # Update output
        self.output *= (1 + self.growth_rate)
        
        # Update value addition (gradual improvement from innovation)
        self.value_addition += 0.2 * self.innovation_rate
        self.value_addition = min(max(self.value_addition, 0.3), 0.7)
        
        # Update backward linkage (domestic input sourcing)
        backward_linkage_growth = 0.01 + 0.1 * self.innovation_rate
        self.backward_linkage = min(self.backward_linkage * (1 + backward_linkage_growth), 0.5)
        
        # Calculate resource use
        resource_use = self.calculate_resource_use(self.growth_rate)
        
        # Compile results
        results = {
            'output': self.output,
            'growth_rate': self.growth_rate,
            'employment': self.employment,
            'productivity': self.productivity,
            'export_share': self.export_share,
            'value_addition': self.value_addition,
            'backward_linkage': self.backward_linkage,
            'compliance_level': self.compliance_level,
            'factories': self.factories,
            'female_employment_share': self.female_employment_share,
            'resource_use': resource_use,
            'effects': {
                'exchange_rate': exchange_rate_effect,
                'global_demand': global_demand_effect,
                'labor': labor_effect,
                'infrastructure': infrastructure_effect,
                'environmental': environmental_effect,
                'finance': finance_effect,
                'governance': governance_effect
            }
        }
        
        return results


class AgriculturalSector(BaseSector):
    """
    Model of Bangladesh's agricultural sector including rice, jute, and tea production.
    """
    
    def __init__(self, config, economic_data):
        """
        Initialize the agricultural sector.
        
        Args:
            config (dict): Agricultural sector configuration
            economic_data (dict): Economic data including agricultural sector data
        """
        super().__init__('agricultural', config, economic_data)
        
        # Initialize sector-specific attributes
        self.output = economic_data.get('agricultural_output', 35.0)  # Billion USD
        self.employment = economic_data.get('agricultural_employment', 25000000)  # 25 million workers
        self.productivity = economic_data.get('agricultural_productivity', 1.0)
        self.export_share = economic_data.get('agricultural_export_share', 0.10)  # 10% exported
        self.formal_share = economic_data.get('agricultural_formal_share', 0.20)  # 20% formal
        
        # Agricultural subsectors
        self.subsectors = {
            'rice': {
                'output': economic_data.get('rice_output', 20.0),
                'land_area': economic_data.get('rice_land_area', 11.0),  # Million hectares
                'yield': economic_data.get('rice_yield', 4.0),  # Tons per hectare
                'export_share': economic_data.get('rice_export_share', 0.05)
            },
            'jute': {
                'output': economic_data.get('jute_output', 3.0),
                'land_area': economic_data.get('jute_land_area', 0.8),  # Million hectares
                'yield': economic_data.get('jute_yield', 2.5),  # Tons per hectare
                'export_share': economic_data.get('jute_export_share', 0.60)
            },
            'tea': {
                'output': economic_data.get('tea_output', 1.0),
                'land_area': economic_data.get('tea_land_area', 0.06),  # Million hectares
                'yield': economic_data.get('tea_yield', 1.5),  # Tons per hectare
                'export_share': economic_data.get('tea_export_share', 0.70)
            },
            'vegetables': {
                'output': economic_data.get('vegetables_output', 5.0),
                'land_area': economic_data.get('vegetables_land_area', 1.0),  # Million hectares
                'yield': economic_data.get('vegetables_yield', 16.0),  # Tons per hectare
                'export_share': economic_data.get('vegetables_export_share', 0.15)
            },
            'fisheries': {
                'output': economic_data.get('fisheries_output', 6.0),
                'area': economic_data.get('fisheries_area', 2.8),  # Million hectares of water bodies
                'yield': economic_data.get('fisheries_yield', 4.2),  # Tons per hectare
                'export_share': economic_data.get('fisheries_export_share', 0.20)
            }
        }
        
        # Agricultural inputs and technology
        self.irrigation_coverage = economic_data.get('irrigation_coverage', 0.55)  # 55% of cultivable land
        self.fertilizer_use = economic_data.get('fertilizer_use', 1.0)  # Normalized index
        self.mechanization_level = economic_data.get('mechanization_level', 0.3)  # 30% mechanized
        self.seed_quality = economic_data.get('seed_quality', 0.6)  # 0-1 index
        
        # Environmental sensitivity
        self.climate_sensitivity = economic_data.get('agricultural_climate_sensitivity', 0.8)  # High sensitivity
        self.water_dependency = economic_data.get('agricultural_water_dependency', 0.7)  # High dependency
        
        print("Agricultural sector initialized")
    
    def step(self, exchange_rate=None, inflation_rate=None, labor_supply=None, 
             infrastructure=None, environmental=None, financial_markets=None, governance=None):
        """
        Advance the agricultural sector by one time step.
        
        Args:
            exchange_rate (float): Current exchange rate
            inflation_rate (float): Current inflation rate
            labor_supply (dict): Labor market conditions
            infrastructure (dict): Infrastructure quality metrics
            environmental (dict): Environmental conditions and impacts
            financial_markets (dict): Financial market conditions
            governance (dict): Governance factors
            
        Returns:
            dict: Agricultural sector results after the step
        """
        # Weather and climate effects (critical for agriculture)
        weather_effect = 0.0
        if environmental:
            if isinstance(environmental, dict):
                rainfall_effect = 0.3 * (environmental.get('rainfall_adequacy', 0.5) - 0.5)
                temperature_effect = -0.2 * abs(environmental.get('temperature_anomaly', 0))
                flood_effect = -0.3 * environmental.get('flood_impacts', 0)
                salinity_effect = -0.15 * environmental.get('salinity_intrusion', 0)
                adaptation_effect = 0.05 * environmental.get('adaptation_investment', 0)
            else:
                # If environmental is an object, try to get attributes or use defaults
                rainfall_effect = 0.3 * (getattr(environmental, 'rainfall_adequacy', 0.5) - 0.5)
                temperature_effect = -0.2 * abs(getattr(environmental, 'temperature_anomaly', 0))
                flood_effect = -0.3 * getattr(environmental, 'flood_impacts', 0)
                salinity_effect = -0.15 * getattr(environmental, 'salinity_intrusion', 0)
                adaptation_effect = 0.05 * getattr(environmental, 'adaptation_investment', 0)
                
            weather_effect = rainfall_effect + temperature_effect + flood_effect + salinity_effect
            
            # Update climate sensitivity based on adaptation measures
            self.climate_sensitivity = max(self.climate_sensitivity - adaptation_effect, 0.4)
        
        # Labor market effects
        labor_effect = 0.0
        if labor_supply is not None:
            # Check if labor_supply is a dictionary or a numeric value
            if isinstance(labor_supply, dict):
                # Original behavior for dictionary input
                labor_availability = labor_supply.get('agricultural_labor_availability', 0.7)
                skill_level = labor_supply.get('agricultural_skill_level', 0.4)
            else:
                # Handle numeric input (total labor force)
                # Default values based on total labor availability
                # Calculate availability as a function of total labor supply
                # Higher labor supply means higher labor availability
                labor_availability = min(0.9, max(0.5, 0.7 + (labor_supply / 100_000_000 - 1) * 0.1))
                skill_level = 0.4  # Default skill level
            
            # Labor effect combines availability and skills
            labor_effect = 0.1 * (labor_availability - 0.7) + 0.3 * (skill_level - 0.5)
            
            # Update employment based on output growth and mechanization
            employment_growth = self.growth_rate - 0.3 * (self.mechanization_level - 0.3)
            self.employment *= (1 + employment_growth)
        
        # Infrastructure effects
        infrastructure_effect = 0.0
        if infrastructure:
            if isinstance(infrastructure, dict):
                irrigation_effect = 0.25 * (infrastructure.get('irrigation_coverage', 0.5) - 0.5)
                transport_effect = 0.15 * (infrastructure.get('rural_road_density', 0.5) - 0.5)
                storage_effect = 0.10 * (infrastructure.get('storage_capacity', 0.5) - 0.5)
                irrigation_investment = infrastructure.get('irrigation_investment', 0)
            else:
                # If infrastructure is an object, try to get attributes or use defaults
                irrigation_effect = 0.25 * (getattr(infrastructure, 'irrigation_coverage', 0.5) - 0.5)
                transport_effect = 0.15 * (getattr(infrastructure, 'rural_road_density', 0.5) - 0.5)
                storage_effect = 0.10 * (getattr(infrastructure, 'storage_capacity', 0.5) - 0.5)
                irrigation_investment = getattr(infrastructure, 'irrigation_investment', 0)
                
            infrastructure_effect = irrigation_effect + transport_effect + storage_effect
            
            # Update irrigation coverage
            self.irrigation_coverage = min(self.irrigation_coverage * (1 + 0.1 * irrigation_investment), 0.8)
        
        # Access to finance
        finance_effect = 0.0
        if financial_markets:
            if isinstance(financial_markets, dict):
                rural_credit = financial_markets.get('access_rates', {}).get('rural_formal_access', 0.25)
                microfinance = financial_markets.get('credit_volumes', {}).get('microfinance_credit_volume', 30)
            else:
                # If financial_markets is an object, try to get attributes or use defaults
                access_rates = getattr(financial_markets, 'access_rates', {})
                credit_volumes = getattr(financial_markets, 'credit_volumes', {})
                
                if isinstance(access_rates, dict):
                    rural_credit = access_rates.get('rural_formal_access', 0.25)
                else:
                    rural_credit = getattr(access_rates, 'rural_formal_access', 0.25)
                    
                if isinstance(credit_volumes, dict):
                    microfinance = credit_volumes.get('microfinance_credit_volume', 30)
                else:
                    microfinance = getattr(credit_volumes, 'microfinance_credit_volume', 30)
                    
            finance_effect = 0.15 * (rural_credit - 0.25) + 0.1 * (microfinance / 30 - 1)
        
        # Governance and policy effects
        governance_effect = 0.0
        if governance:
            # Check if governance is a dictionary or an object
            if isinstance(governance, dict):
                subsidy_effect = 0.15 * governance.get('agricultural_subsidies', 0.5)
                extension_effect = 0.10 * governance.get('extension_services', 0.5)
            else:
                # If governance is an object, try to get attributes or use defaults
                subsidy_effect = 0.15 * getattr(governance, 'agricultural_subsidies', 0.5)
                extension_effect = 0.10 * getattr(governance, 'extension_services', 0.5)
            
            governance_effect = subsidy_effect + extension_effect
        
        # Input prices effect
        input_price_effect = 0.0
        if inflation_rate:
            # Agricultural inputs often rise faster than general inflation
            input_price_growth = 1.2 * inflation_rate
            input_price_effect = -0.2 * input_price_growth
        
        # Update productivity
        labor_skills = 0.4  # Default value
        if labor_supply:
            if isinstance(labor_supply, dict):
                labor_skills = labor_supply.get('agricultural_skill_level', 0.4)
            else:
                # If labor_supply is a float, use the default skill level
                labor_skills = 0.4
                
        self.update_productivity(labor_skills, infrastructure)
        
        # Update technology adoption
        self.update_innovation(education_level=0.4, r_d_investment=0.01)
        
        # Update mechanization level
        self.mechanization_level += 0.01 + 0.1 * self.innovation_rate
        self.mechanization_level = min(self.mechanization_level, 0.7)
        
        # Update fertilizer use based on availability and price
        fertilizer_adjustment = 0.01
        if finance_effect > 0:
            fertilizer_adjustment += 0.02
        self.fertilizer_use *= (1 + fertilizer_adjustment)
        self.fertilizer_use = min(max(self.fertilizer_use, 0.5), 1.5)
        
        # Update seed quality with innovation
        self.seed_quality += 0.2 * self.innovation_rate
        self.seed_quality = min(max(self.seed_quality, 0.4), 0.9)
        
        # Combine all effects to determine output growth
        self.growth_rate = (
            0.02 +  # Base growth trend
            self.climate_sensitivity * weather_effect +
            labor_effect +
            infrastructure_effect +
            finance_effect +
            governance_effect +
            input_price_effect +
            0.15 * (self.productivity - 1)  # Productivity effect
        )
        
        # Update output
        self.output *= (1 + self.growth_rate)
        
        # Update subsectors
        for subsector, data in self.subsectors.items():
            # Different growth rates for different subsectors
            if subsector == 'rice':
                subsector_growth = self.growth_rate - 0.01  # Slightly slower than average
            elif subsector == 'jute':
                subsector_growth = self.growth_rate + 0.01  # Slightly faster than average
            elif subsector == 'fisheries':
                subsector_growth = self.growth_rate + 0.02  # Faster growth
            else:
                subsector_growth = self.growth_rate
                
            # Update output
            data['output'] *= (1 + subsector_growth)
            
            # Update yields with technology improvement
            yield_improvement = 0.01 + 0.1 * self.innovation_rate
            data['yield'] *= (1 + yield_improvement)
            
            # Update land area based on yield and output
            data['land_area'] = data['output'] / data['yield'] if 'land_area' in data else data['area']
        
        # Calculate resource use
        resource_use = self.calculate_resource_use(self.growth_rate)
        
        # Recalculate total output from subsectors
        self.output = sum(data['output'] for data in self.subsectors.values())
        
        # Compile results
        results = {
            'output': self.output,
            'growth_rate': self.growth_rate,
            'employment': self.employment,
            'productivity': self.productivity,
            'subsectors': self.subsectors,
            'irrigation_coverage': self.irrigation_coverage,
            'fertilizer_use': self.fertilizer_use,
            'mechanization_level': self.mechanization_level,
            'seed_quality': self.seed_quality,
            'climate_sensitivity': self.climate_sensitivity,
            'resource_use': resource_use,
            'effects': {
                'weather': weather_effect,
                'labor': labor_effect,
                'infrastructure': infrastructure_effect,
                'finance': finance_effect,
                'governance': governance_effect,
                'input_prices': input_price_effect
            }
        }
        
        return results


class TechnologySector(BaseSector):
    """
    Model of Bangladesh's technology and digital sector.
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
        self.output = economic_data.get('tech_output', 8.0)  # Billion USD
        self.employment = economic_data.get('tech_employment', 1000000)  # 1 million workers
        self.productivity = economic_data.get('tech_productivity', 1.5)  # Higher than average
        self.export_share = economic_data.get('tech_export_share', 0.30)  # 30% exported (IT services)
        self.formal_share = economic_data.get('tech_formal_share', 0.80)  # 80% formal
        
        # Tech subsectors
        self.subsectors = {
            'software': {
                'output': economic_data.get('software_output', 3.5),  # Billion USD
                'employment': economic_data.get('software_employment', 400000),
                'growth_rate': economic_data.get('software_growth', 0.12)  # 12% growth rate
            },
            'hardware': {
                'output': economic_data.get('hardware_output', 2.0),  # Billion USD
                'employment': economic_data.get('hardware_employment', 300000),
                'growth_rate': economic_data.get('hardware_growth', 0.08)  # 8% growth rate
            },
            'telecom': {
                'output': economic_data.get('telecom_output', 2.5),  # Billion USD
                'employment': economic_data.get('telecom_employment', 300000),
                'growth_rate': economic_data.get('telecom_growth', 0.10)  # 10% growth rate
            }
        }
        
        # Digital infrastructure dependencies
        self.internet_dependency = economic_data.get('tech_internet_dependency', 0.9)  # High dependency
        self.electricity_dependency = economic_data.get('tech_electricity_dependency', 0.95)  # Very high dependency
        
        # Innovation metrics
        self.innovation_rate = economic_data.get('tech_innovation_rate', 0.08)  # 8% innovation rate
        self.r_d_investment = economic_data.get('tech_r_d_investment', 0.15)  # 15% of output to R&D
        self.startups = economic_data.get('tech_startups', 500)  # Number of tech startups
        self.patents = economic_data.get('tech_patents', 100)  # Annual patents
        
        # Human capital
        self.skilled_labor_share = economic_data.get('tech_skilled_labor', 0.70)  # 70% skilled
        self.education_level = economic_data.get('tech_education_level', 0.75)  # 0-1 index
        self.digital_literacy = economic_data.get('digital_literacy', 0.50)  # 0-1 index
        
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
        # Track previous values for growth calculation
        previous_output = self.output
        
        # 1. Infrastructure constraints
        infrastructure_constraints = self._assess_infrastructure_constraints(infrastructure)
        
        # 2. Update productivity based on inputs and constraints
        education_level = 0.5  # Default value
        if labor_supply:
            if isinstance(labor_supply, dict):
                education_level = labor_supply.get('education_index', 0.5)
            else:
                # If labor_supply is a float, use the default education level
                education_level = 0.5
                
        skills_level = 0.6  # Default value
        if labor_supply:
            if isinstance(labor_supply, dict):
                skills_level = labor_supply.get('tech_skills', 0.6)
            else:
                # If labor_supply is a float, use the default skill level
                skills_level = 0.6
                
        telecom_quality = 0.5  # Default value
        if infrastructure:
            if isinstance(infrastructure, dict):
                telecom_quality = infrastructure.get('telecom_index', 0.5)
            else:
                # If infrastructure is a float, use the default telecom quality
                telecom_quality = 0.5
                
        electricity_reliability = 0.5  # Default value
        if infrastructure:
            if isinstance(infrastructure, dict):
                electricity_reliability = infrastructure.get('electricity_reliability', 0.5)
            else:
                # If infrastructure is a float, use the default electricity reliability
                electricity_reliability = 0.5
                
        # Calculate effective productivity with infrastructure constraints
        self.update_productivity(
            labor_skills=skills_level,
            infrastructure={
                'telecom_coverage': telecom_quality,
                'electricity_reliability': electricity_reliability
            },
            technology_spillover=0.05  # Assumption: constant tech spillover
        )
        
        # 3. Innovation dynamics
        r_d_investment = 0.01  # Default value
        if financial_markets:
            if isinstance(financial_markets, dict):
                r_d_investment = financial_markets.get('venture_capital', 0.01)
            else:
                # If financial_markets is a float, use the default r_d_investment
                r_d_investment = 0.01
        
        trade_openness = 0.6  # Base assumption, could come from data
        
        self.update_innovation(
            education_level=education_level,
            r_d_investment=r_d_investment,
            openness=trade_openness
        )
        
        # 4. Update startups and patents
        startup_growth = self.r_d_investment * 2.0 + self.innovation_rate * 5.0
        self.startups = int(self.startups * (1 + startup_growth))
        
        patent_growth = self.innovation_rate * 3.0
        self.patents = int(self.patents * (1 + patent_growth))
        
        # 5. Update subsector outputs
        for subsector in self.subsectors:
            # Base growth
            base_growth = self.subsectors[subsector]['growth_rate']
            
            # Modify with innovation effect
            innovation_effect = self.innovation_rate * 0.5
            
            # Infrastructure constraint effect
            constraint_effect = -0.05 * infrastructure_constraints
            
            # Governance effect on digital economy
            governance_effect = 0.0
            if governance:
                if isinstance(governance, dict):
                    policy_effectiveness = governance.get('policy_effectiveness', 0.5)
                    digital_policy = governance.get('digital_policy', 0.5)
                else:
                    policy_effectiveness = getattr(governance, 'policy_effectiveness', 0.5)
                    digital_policy = getattr(governance, 'digital_policy', 0.5)
                governance_effect = (policy_effectiveness + digital_policy - 1.0) * 0.1
            
            # Final growth rate with random component
            growth_rate = base_growth + innovation_effect + constraint_effect + governance_effect + np.random.normal(0, 0.02)
            
            # Update subsector
            self.subsectors[subsector]['output'] *= (1 + growth_rate)
            self.subsectors[subsector]['employment'] *= (1 + growth_rate * 0.7)  # Employment grows slower due to productivity
        
        # 6. Calculate total sector output and employment
        self.output = sum(subsector['output'] for subsector in self.subsectors.values())
        self.employment = sum(int(subsector['employment']) for subsector in self.subsectors.values())
        
        # 7. Update export share based on global connectivity and competitiveness
        global_connectivity = telecom_quality * 0.7 + self.innovation_rate * 0.3
        self.export_share = min(0.6, self.export_share + (global_connectivity - 0.5) * 0.02)
        
        # 8. Calculate sector growth rate
        self.growth_rate = (self.output / previous_output) - 1.0 if previous_output > 0 else 0.0
        
        # 9. Calculate resource use
        resource_use = self.calculate_resource_use(self.growth_rate)
        
        # 10. Calculate sector impacts on digital literacy and broader economy
        digital_literacy_impact = self.output * 0.01 / 100  # Normalized contribution to digital literacy
        
        # Compile and return results
        return {
            'output': self.output,
            'employment': self.employment,
            'productivity': self.productivity,
            'growth_rate': self.growth_rate,
            'export_share': self.export_share,
            'exports': self.output * self.export_share,
            'subsectors': self.subsectors,
            'innovation_rate': self.innovation_rate,
            'r_d_investment': self.r_d_investment,
            'startups': self.startups,
            'patents': self.patents,
            'resource_use': resource_use,
            'digital_literacy_impact': digital_literacy_impact,
            'infrastructure_constraints': infrastructure_constraints,
            'formal_share': self.formal_share
        }
    
    def _assess_infrastructure_constraints(self, infrastructure):
        """
        Calculate the infrastructure constraints on tech sector growth.
        
        Args:
            infrastructure (dict): Infrastructure quality metrics
            
        Returns:
            float: Constraint factor (0-1, where 0 is no constraint)
        """
        if not infrastructure:
            return 0.3  # Default moderate constraint
        
        # Extract relevant infrastructure metrics
        internet_coverage = infrastructure.get('internet_coverage', 0.5)
        broadband_penetration = infrastructure.get('broadband_penetration', 0.3)
        electricity_reliability = infrastructure.get('electricity_reliability', 0.5)
        digital_services = infrastructure.get('digital_services', 0.4)
        
        # Calculate constraints from each infrastructure type
        internet_constraint = self.internet_dependency * (1 - internet_coverage)
        electricity_constraint = self.electricity_dependency * (1 - electricity_reliability)
        broadband_constraint = 0.8 * (1 - broadband_penetration)  # 80% dependency on broadband
        digital_services_constraint = 0.6 * (1 - digital_services)  # 60% dependency on digital services
        
        # Combined constraint (weighted average)
        combined_constraint = (internet_constraint * 0.3 + 
                              electricity_constraint * 0.3 + 
                              broadband_constraint * 0.25 + 
                              digital_services_constraint * 0.15)
        
        return combined_constraint
        
    def analyze_digital_divide(self, regional_data=None):
        """
        Analyze the digital divide across regions.
        
        Args:
            regional_data (dict): Regional infrastructure and demographic data
            
        Returns:
            dict: Digital divide analysis results
        """
        if not regional_data:
            return {'digital_divide_severity': 0.5}  # Default
        
        # Calculate variance in digital access across regions
        internet_accesses = [region.get('internet_coverage', 0) for region in regional_data.values()]
        if not internet_accesses:
            return {'digital_divide_severity': 0.5}  # Default
        
        # Use coefficient of variation as digital divide measure
        mean_access = sum(internet_accesses) / len(internet_accesses)
        variance = sum((x - mean_access) ** 2 for x in internet_accesses) / len(internet_accesses)
        std_dev = variance ** 0.5
        digital_divide = std_dev / mean_access if mean_access > 0 else 1.0
        
        return {
            'digital_divide_severity': min(1.0, digital_divide),
            'regional_variations': {
                'min_access': min(internet_accesses),
                'max_access': max(internet_accesses),
                'mean_access': mean_access
            }
        }

class InformalEconomySector(BaseSector):
    """
    Model of Bangladesh's informal economy including bazaar trade, 
    domestic services, small-scale manufacturing, and other unregistered activities.
    """
    
    def __init__(self, config, economic_data):
        """
        Initialize the informal economy sector.
        
        Args:
            config (dict): Informal economy configuration
            economic_data (dict): Economic data including informal sector data
        """
        super().__init__('informal', config, economic_data)
        
        # Initialize sector-specific attributes
        self.output = economic_data.get('informal_output', 50.0)  # Billion USD
        self.employment = economic_data.get('informal_employment', 35000000)  # 35 million workers
        self.productivity = economic_data.get('informal_productivity', 0.7)  # Lower than formal sector
        self.export_share = economic_data.get('informal_export_share', 0.05)  # 5% exported
        self.formal_share = 0.0  # By definition
        
        # Informal subsectors
        self.subsectors = {
            'retail_trade': {
                'output': economic_data.get('informal_retail_output', 18.0),  # Billion USD
                'employment': economic_data.get('informal_retail_employment', 12000000),
                'growth_rate': economic_data.get('informal_retail_growth', 0.04)  # 4% growth rate
            },
            'manufacturing': {
                'output': economic_data.get('informal_manufacturing_output', 10.0),  # Billion USD
                'employment': economic_data.get('informal_manufacturing_employment', 8000000),
                'growth_rate': economic_data.get('informal_manufacturing_growth', 0.03)  # 3% growth rate
            },
            'services': {
                'output': economic_data.get('informal_services_output', 15.0),  # Billion USD
                'employment': economic_data.get('informal_services_employment', 10000000),
                'growth_rate': economic_data.get('informal_services_growth', 0.04)  # 4% growth rate
            },
            'agriculture': {
                'output': economic_data.get('informal_agriculture_output', 7.0),  # Billion USD
                'employment': economic_data.get('informal_agriculture_employment', 5000000),
                'growth_rate': economic_data.get('informal_agriculture_growth', 0.02)  # 2% growth rate
            }
        }
        
        # Informal sector characteristics
        self.tax_evasion_rate = economic_data.get('informal_tax_evasion', 0.95)  # 95% tax evasion
        self.urban_share = economic_data.get('informal_urban_share', 0.65)  # 65% in urban areas
        self.female_participation = economic_data.get('informal_female_participation', 0.45)  # 45% female participation
        self.youth_participation = economic_data.get('informal_youth_participation', 0.30)  # 30% youth participation
        self.labor_vulnerability = economic_data.get('informal_labor_vulnerability', 0.85)  # High vulnerability
        
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
        # Track previous values for growth calculation
        previous_output = self.output
        
        # 1. Governance and regulatory effects
        regulatory_burden, enforcement_level = self._assess_regulatory_environment(governance)
        
        # 2. Update productivity (typically slower in informal sector)
        if labor_supply is not None:
            # Check if labor_supply is a dictionary or a numeric value
            if isinstance(labor_supply, dict):
                # Original behavior for dictionary input
                labor_skills = labor_supply.get('average_skills', 0.4)
            else:
                # Handle numeric input (total labor force)
                labor_skills = 0.4  # Default lower skill level
            
            self.update_productivity(
                labor_skills=labor_skills,
                infrastructure=infrastructure,
                technology_spillover=0.02  # Lower technology spillover
            )
        
        # 3. Calculate formal-to-informal and informal-to-formal transition rates
        formalization_rate, informalization_rate = self._calculate_transition_rates(
            governance, financial_markets, labor_supply
        )
        
        # 4. Update subsector outputs
        for subsector in self.subsectors:
            # Base growth with subsector-specific rate
            base_growth = self.subsectors[subsector]['growth_rate']
            
            # Governance effect (higher enforcement reduces growth)
            governance_effect = -0.02 * enforcement_level
            
            # Unemployment and poverty effect (economic distress increases informal activity)
            economic_distress = 0.0
            if labor_supply:
                if isinstance(labor_supply, dict):
                    unemployment = labor_supply.get('unemployment_rate', 0.05)
                    wage_differential = labor_supply.get('formal_informal_wage_ratio', 1.8)
                else:
                    # Default values if labor_supply is not a dictionary
                    unemployment = 0.05
                    wage_differential = 1.8
                economic_distress = max(0, (unemployment - 0.05) * 0.5)
            
            # Economic cycle effect (informal sector often countercyclical)
            formal_growth = 0.04  # Assume moderate formal sector growth
            countercyclical_effect = max(-0.02, min(0.02, (0.04 - formal_growth) * 0.5))
            
            # Infrastructure effect (less significant than in formal sectors)
            infrastructure_effect = 0.0
            if infrastructure:
                transport_quality = infrastructure.get('transport_index', 0.5)
                infrastructure_effect = (transport_quality - 0.5) * 0.05
            
            # Random component
            random_effect = np.random.normal(0, 0.02)
            
            # Calculate growth rate
            growth_rate = (base_growth + governance_effect + economic_distress + 
                          countercyclical_effect + infrastructure_effect + random_effect)
            
            # Adjust for formalization/informalization (net flow)
            net_transition = informalization_rate - formalization_rate
            growth_rate += net_transition
            
            # Update subsector values
            self.subsectors[subsector]['output'] *= (1 + growth_rate)
            self.subsectors[subsector]['employment'] *= (1 + growth_rate)
        
        # 5. Update sector totals
        self.output = sum(subsector['output'] for subsector in self.subsectors.values())
        self.employment = sum(int(subsector['employment']) for subsector in self.subsectors.values())
        
        # 6. Calculate sector growth rate
        self.growth_rate = (self.output / previous_output) - 1.0 if previous_output > 0 else 0.0
        
        # 7. Calculate resource use (often less efficient)
        resource_use = self.calculate_resource_use(self.growth_rate)
        
        # 8. Calculate tax leakage
        tax_leakage = self.output * self.tax_evasion_rate * 0.15  # Assuming 15% effective tax rate
        
        # 9. Vulnerability metrics
        vulnerability_metrics = self._calculate_vulnerability_metrics(
            inflation_rate, environmental, labor_supply
        )
        
        # Compile and return results
        return {
            'output': self.output,
            'employment': self.employment,
            'productivity': self.productivity,
            'growth_rate': self.growth_rate,
            'subsectors': self.subsectors,
            'resource_use': resource_use,
            'tax_leakage': tax_leakage,
            'formalization_rate': formalization_rate,
            'urban_share': self.urban_share,
            'female_participation': self.female_participation,
            'youth_participation': self.youth_participation,
            'vulnerability_metrics': vulnerability_metrics,
            'labor_vulnerability': self.labor_vulnerability
        }
    
    def _assess_regulatory_environment(self, governance):
        """
        Assess the regulatory environment affecting the informal sector.
        
        Args:
            governance (dict): Governance metrics
            
        Returns:
            tuple: (regulatory_burden, enforcement_level)
        """
        if not governance:
            return 0.6, 0.4  # Default moderate burden, limited enforcement
        
        # Regulatory burden (higher values discourage formalization)
        if isinstance(governance, dict):
            reg_complexity = governance.get('regulatory_complexity', 0.6)
            cost_of_doing_business = governance.get('cost_of_doing_business', 0.6)
        else:
            reg_complexity = getattr(governance, 'regulatory_complexity', 0.6)
            cost_of_doing_business = getattr(governance, 'cost_of_doing_business', 0.6)
        regulatory_burden = (reg_complexity * 0.6) + (cost_of_doing_business * 0.4)
        
        # Enforcement level (higher values encourage formalization)
        if isinstance(governance, dict):
            rule_of_law = governance.get('rule_of_law', 0.4)
            enforcement_capacity = governance.get('enforcement_capacity', 0.4)
        else:
            rule_of_law = getattr(governance, 'rule_of_law', 0.4)
            enforcement_capacity = getattr(governance, 'enforcement_capacity', 0.4)
        enforcement_level = (rule_of_law * 0.5) + (enforcement_capacity * 0.5)
        
        return regulatory_burden, enforcement_level
    
    def _calculate_transition_rates(self, governance, financial_markets, labor_supply):
        """
        Calculate rates of transition between formal and informal sectors.
        
        Args:
            governance (dict): Governance metrics
            financial_markets (dict): Financial market conditions
            labor_supply (dict): Labor market conditions
            
        Returns:
            tuple: (formalization_rate, informalization_rate)
        """
        # Base rates
        base_formalization = 0.01  # 1% of informal becomes formal by default
        base_informalization = 0.005  # 0.5% of formal becomes informal by default
        
        # Governance effects
        regulatory_burden, enforcement_level = self._assess_regulatory_environment(governance)
        governance_effect_form = -0.02 * regulatory_burden + 0.03 * enforcement_level
        
        # Financial inclusion effect
        financial_effect = 0.0
        if financial_markets:
            financial_inclusion = financial_markets.get('financial_inclusion', 0.3)
            credit_access = financial_markets.get('sme_credit_access', 0.3)
            financial_effect = (financial_inclusion + credit_access) * 0.01
        
        # Economic opportunity effect
        opportunity_effect = 0.0
        if labor_supply:
            if isinstance(labor_supply, dict):
                unemployment = labor_supply.get('unemployment_rate', 0.05)
                wage_differential = labor_supply.get('formal_informal_wage_ratio', 1.8)
            else:
                # Default values if labor_supply is not a dictionary
                unemployment = 0.05
                wage_differential = 1.8
            opportunity_effect = -0.01 * unemployment + 0.01 * (wage_differential - 1.5)
        
        # Calculate final rates
        formalization_rate = max(0, min(0.05, base_formalization + governance_effect_form + 
                                      financial_effect + opportunity_effect))
        
        # Informalization works in opposite direction
        informalization_rate = max(0, min(0.05, base_informalization - governance_effect_form - 
                                        financial_effect + 0.01 * (1 - opportunity_effect)))
        
        return formalization_rate, informalization_rate
    
    def _calculate_vulnerability_metrics(self, inflation_rate, environmental, labor_supply):
        """
        Calculate metrics of vulnerability in the informal economy.
        
        Args:
            inflation_rate (float): Current inflation rate
            environmental (dict): Environmental conditions and impacts
            labor_supply (dict): Labor market conditions
            
        Returns:
            dict: Vulnerability metrics
        """
        # Base vulnerability level
        vulnerability = self.labor_vulnerability
        
        # Inflation effect (informal workers often hit harder by inflation)
        inflation_effect = 0.0
        if inflation_rate:
            inflation_effect = max(0, (inflation_rate - 0.05) * 0.5)  # Effect above 5% inflation
        
        # Environmental effect
        environmental_effect = 0.0
        if environmental:
            if isinstance(environmental, dict):
                climate_vulnerability = environmental.get('climate_vulnerability', 0.6)
                disaster_impact = environmental.get('disaster_impact', 0.0)
            else:
                # If environmental is an object, try to get attributes or use defaults
                climate_vulnerability = getattr(environmental, 'climate_vulnerability', 0.6)
                disaster_impact = getattr(environmental, 'disaster_impact', 0.0)
            environmental_effect = (climate_vulnerability * 0.3) + (disaster_impact * 0.7)
        
        # Social protection effect
        social_protection_effect = 0.0
        if labor_supply:
            if isinstance(labor_supply, dict):
                social_protection = labor_supply.get('social_protection_coverage', 0.2)
            else:
                # If labor_supply is not a dictionary, use default value
                social_protection = 0.2
            social_protection_effect = -0.3 * social_protection  # Reduces vulnerability
        
        # Calculate final vulnerability metrics
        vulnerability += (inflation_effect + environmental_effect + social_protection_effect)
        vulnerability = max(min(vulnerability, 1.0), 0.0)  # Bound between 0 and 1
        
        return {
            'overall_vulnerability': vulnerability,
            'inflation_vulnerability': inflation_effect,
            'environmental_vulnerability': environmental_effect,
            'social_protection_gap': -social_protection_effect
        }
