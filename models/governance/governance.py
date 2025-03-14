#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Governance System module for Bangladesh simulation model.
This module integrates the institutional components, policy frameworks,
governance indicators, and decision-making processes that affect
Bangladesh's development trajectory.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, beta

from models.governance.institutions import GovernanceSystem as InstitutionalSystem


class GovernanceSystem:
    """
    Governance System for Bangladesh simulation, integrating institutional effectiveness,
    policy frameworks, decision-making processes, and governance quality metrics.
    This system models how governance affects and is affected by other simulation components.
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
        
        # Initialize base year and current year
        self.current_year = config.get('start_year', 2023)
        self.base_year = config.get('base_year', 2000)
        self.time_step = config.get('time_step', 1.0)
        
        # Initialize institutional subsystem
        self.institutional_system = InstitutionalSystem(config, data_loader)
        
        # Load governance data
        self.governance_data = data_loader.load_governance_data()
        
        # Initialize policy parameters
        self.initialize_policy_parameters()
        
        # Initialize governance quality indicators
        self.initialize_governance_indicators()
        
        # Initialize infrastructure investment allocation
        self.investment_allocation = {
            'transport': 0.35,
            'energy': 0.25,
            'water': 0.20,
            'telecom': 0.15,
            'urban': 0.05
        }
        
        # Initialize regional governance variation
        self.regions = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 
                        'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh']
        self.initialize_regional_governance()
        
        # Store historical governance indicators
        self.history = {
            'years': [],
            'governance_index': [],
            'policy_effectiveness': [],
            'institutional_effectiveness': [],
            'corruption_index': [],
            'infrastructure_investment': [],
            'education_investment': [],
            'health_investment': [],
            'environmental_policy': [],
            'trade_policy': [],
            'regulatory_quality': []
        }
        
        print("Governance system initialized")
    
    def initialize_policy_parameters(self):
        """Initialize policy-related parameters from governance data"""
        # Economic policies
        self.fiscal_policy = self.governance_data.get('fiscal_policy', 0.5)  # 0-1 scale (conservative to expansionary)
        self.monetary_policy = self.governance_data.get('monetary_policy', 0.5)  # 0-1 scale (tight to loose)
        self.trade_policy = self.governance_data.get('trade_policy', 0.6)  # 0-1 scale (protectionist to open)
        self.industrial_policy = self.governance_data.get('industrial_policy', 0.5)  # 0-1 scale
        self.agricultural_policy = self.governance_data.get('agricultural_policy', 0.45)  # 0-1 scale
        
        # Public investment priorities (as % of GDP)
        self.infrastructure_investment = self.governance_data.get('infrastructure_investment', 0.03)
        self.education_investment = self.governance_data.get('education_investment', 0.02)
        self.health_investment = self.governance_data.get('health_investment', 0.01)
        self.research_investment = self.governance_data.get('research_investment', 0.005)
        
        # Social policies
        self.welfare_policy = self.governance_data.get('welfare_policy', 0.4)  # 0-1 scale
        self.labor_policy = self.governance_data.get('labor_policy', 0.5)  # 0-1 scale
        self.education_policy = self.governance_data.get('education_policy', 0.55)  # 0-1 scale
        self.health_policy = self.governance_data.get('health_policy', 0.5)  # 0-1 scale
        
        # Environmental policies
        self.environmental_policy = self.governance_data.get('environmental_policy', 0.4)  # 0-1 scale
        self.climate_policy = self.governance_data.get('climate_policy', 0.35)  # 0-1 scale
        self.water_management_policy = self.governance_data.get('water_management_policy', 0.5)  # 0-1 scale
        
        # Policy implementation effectiveness
        self.policy_implementation = self.governance_data.get('policy_implementation', 0.4)  # 0-1 scale
        self.policy_coherence = self.governance_data.get('policy_coherence', 0.45)  # 0-1 scale
    
    def initialize_governance_indicators(self):
        """Initialize governance quality indicators"""
        # Calculate overall governance index from institutional data
        inst = self.institutional_system
        
        # Core governance quality metrics from institutional factors
        self.governance_index = 0.2 * (inst.government_effectiveness + 
                                      inst.regulatory_quality + 
                                      inst.rule_of_law + 
                                      (1 - inst.corruption_index) + 
                                      inst.voice_accountability)
        
        # Policy effectiveness is a function of institutional effectiveness and corruption
        self.policy_effectiveness = 0.7 * inst.institutional_effectiveness - 0.3 * inst.corruption_index
        self.policy_effectiveness = max(0.1, min(0.9, self.policy_effectiveness))
        
        # Calculate regulatory burden
        self.regulatory_burden = 0.6 * (1 - inst.regulatory_quality) + 0.4 * inst.corruption_index
        self.regulatory_burden = max(0.1, min(0.9, self.regulatory_burden))
    
    def initialize_regional_governance(self):
        """Initialize regional governance variation"""
        self.regional_governance = {}
        
        for region in self.regions:
            # Base values
            base_governance = {
                'governance_index': self.governance_index,
                'policy_effectiveness': self.policy_effectiveness,
                'infrastructure_investment': self.infrastructure_investment,
                'education_investment': self.education_investment,
                'health_investment': self.health_investment
            }
            
            # Add regional variation (±20%)
            regional_governance = {}
            for key, value in base_governance.items():
                if key.endswith('investment'):  # Investment values should stay positive
                    variation = 1.0 + np.random.uniform(-0.2, 0.2)
                else:  # Index values should stay in 0-1 range
                    variation = 1.0 + np.random.uniform(-0.2, 0.2)
                    
                # Regional adjustments
                if region == 'Dhaka':  # Capital region typically has better governance
                    variation += 0.1
                elif region in ['Sylhet', 'Rangpur']:  # Remote regions might have less effective governance
                    variation -= 0.1
                
                regional_governance[key] = value * variation
                
                # Ensure values stay in reasonable range
                if not key.endswith('investment'):  # For index values
                    regional_governance[key] = max(0.1, min(0.9, regional_governance[key]))
            
            self.regional_governance[region] = regional_governance
    
    def update_policies(self, year, economic_system=None, demographic_system=None, environmental_system=None):
        """
        Update policy parameters based on economic conditions, demographic changes,
        and environmental pressures.
        
        Args:
            year (int): Current simulation year
            economic_system: Economic system for economic indicators
            demographic_system: Demographic system for population indicators
            environmental_system: Environmental system for climate indicators
            
        Returns:
            dict: Updated policy parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {'policy_effectiveness': self.policy_effectiveness}
        
        # Get economic indicators if available
        gdp_growth = 0.05  # Default value
        inflation = 0.06   # Default value
        unemployment = 0.1  # Default value
        trade_balance = 0.0  # Default value
        
        if economic_system:
            gdp_growth = economic_system.gdp_growth_rate if hasattr(economic_system, 'gdp_growth_rate') else 0.05
            inflation = economic_system.inflation_rate if hasattr(economic_system, 'inflation_rate') else 0.06
            unemployment = economic_system.unemployment_rate if hasattr(economic_system, 'unemployment_rate') else 0.1
            trade_balance = economic_system.trade_balance if hasattr(economic_system, 'trade_balance') else 0.0
        
        # Update monetary policy based on inflation and growth
        if inflation > 0.08:  # High inflation
            self.monetary_policy = max(0.1, self.monetary_policy - 0.05 * time_delta)  # Tighten monetary policy
        elif inflation < 0.03 and gdp_growth < 0.04:  # Low inflation and growth
            self.monetary_policy = min(0.9, self.monetary_policy + 0.05 * time_delta)  # Loosen monetary policy
        
        # Update fiscal policy based on growth and unemployment
        if gdp_growth < 0.03 or unemployment > 0.12:  # Economic slowdown or high unemployment
            self.fiscal_policy = min(0.9, self.fiscal_policy + 0.04 * time_delta)  # More expansionary
        elif gdp_growth > 0.07:  # High growth
            self.fiscal_policy = max(0.1, self.fiscal_policy - 0.03 * time_delta)  # More conservative
        
        # Update trade policy based on trade balance
        if trade_balance < -0.05:  # Trade deficit > 5% of GDP
            self.trade_policy = max(0.3, self.trade_policy - 0.02 * time_delta)  # More protectionist
        else:
            # Gradual trend toward more open trade over time (global trend)
            self.trade_policy = min(0.9, self.trade_policy + 0.01 * time_delta)
        
        # Update environmental policy based on environmental pressures
        if environmental_system:
            # Strengthen environmental policies in response to environmental degradation
            env_degradation = (environmental_system.water_stress_index + 
                               environmental_system.land_degradation_index) / 2
            
            # Increase environmental policy stringency if degradation is high
            if env_degradation > 0.6:
                self.environmental_policy = min(0.9, self.environmental_policy + 0.03 * time_delta)
                self.climate_policy = min(0.9, self.climate_policy + 0.03 * time_delta)
            
            # Adjust water management policy based on water stress
            if environmental_system.water_stress_index > 0.7:
                self.water_management_policy = min(0.9, self.water_management_policy + 0.04 * time_delta)
        
        # Calculate policy coherence based on how aligned different policies are
        policy_values = [self.fiscal_policy, self.monetary_policy, self.trade_policy, 
                          self.industrial_policy, self.environmental_policy]
        policy_variance = np.var(policy_values)
        self.policy_coherence = 0.7 - policy_variance  # Lower variance means more coherent policies
        self.policy_coherence = max(0.1, min(0.9, self.policy_coherence))
        
        # Policy implementation quality depends on institutional effectiveness
        inst_effectiveness = self.institutional_system.institutional_effectiveness
        corruption = self.institutional_system.corruption_index
        self.policy_implementation = 0.7 * inst_effectiveness - 0.3 * corruption
        self.policy_implementation = max(0.1, min(0.9, self.policy_implementation))
        
        # Calculate overall policy effectiveness
        self.policy_effectiveness = 0.4 * self.policy_implementation + 0.3 * self.policy_coherence + 0.3 * inst_effectiveness
        self.policy_effectiveness = max(0.1, min(0.9, self.policy_effectiveness))
        
        # Update current year
        self.current_year = year
        
        return {
            'fiscal_policy': self.fiscal_policy,
            'monetary_policy': self.monetary_policy,
            'trade_policy': self.trade_policy,
            'environmental_policy': self.environmental_policy,
            'climate_policy': self.climate_policy,
            'policy_coherence': self.policy_coherence,
            'policy_implementation': self.policy_implementation,
            'policy_effectiveness': self.policy_effectiveness
        }
    
    def update_public_investments(self, year, economic_system=None, demographic_system=None):
        """
        Update public investment allocations based on economic conditions and needs.
        
        Args:
            year (int): Current simulation year
            economic_system: Economic system for GDP and fiscal space
            demographic_system: Demographic system for population needs
            
        Returns:
            dict: Updated investment parameters
        """
        # Calculate time step
        time_delta = year - self.current_year
        if time_delta <= 0:
            return {
                'infrastructure_investment': self.infrastructure_investment,
                'education_investment': self.education_investment,
                'health_investment': self.health_investment
            }
        
        # Base annual change rate
        base_change = 0.001  # 0.1 percentage point per year
        
        # Get economic indicators if available
        gdp_growth = 0.05  # Default value
        fiscal_space = 0.0  # Default is neutral fiscal space
        
        if economic_system:
            gdp_growth = economic_system.gdp_growth_rate if hasattr(economic_system, 'gdp_growth_rate') else 0.05
            
            # Calculate fiscal space (simplistic version)
            # Positive means more fiscal space, negative means constrained
            if hasattr(economic_system, 'government_debt_to_gdp'):
                fiscal_space = 0.6 - economic_system.government_debt_to_gdp  # Threshold at 60% debt-to-GDP ratio
            elif hasattr(economic_system, 'fiscal_balance'):
                fiscal_space = economic_system.fiscal_balance + 0.03  # Threshold at -3% deficit
            
        # Adjust infrastructure investment based on needs and fiscal space
        infra_need = 0.0
        if economic_system and hasattr(economic_system, 'infrastructure_gap'):
            infra_need = economic_system.infrastructure_gap * 0.1
        
        infra_change = base_change + infra_need + fiscal_space * 0.005
        self.infrastructure_investment = max(0.01, min(0.07, 
            self.infrastructure_investment + infra_change * time_delta))
        
        # Adjust education investment based on demographic factors
        edu_need = 0.0
        if demographic_system:
            # Higher youth population creates greater education investment needs
            youth_pop_ratio = demographic_system.age_distribution.get('0-14', 0.3)
            edu_need = (youth_pop_ratio - 0.25) * 0.02  # Baseline youth population of 25%
        
        edu_change = base_change + edu_need + fiscal_space * 0.003
        self.education_investment = max(0.01, min(0.05, 
            self.education_investment + edu_change * time_delta))
        
        # Adjust health investment based on demographic factors
        health_need = 0.0
        if demographic_system:
            # Aging population increases healthcare needs
            elderly_pop_ratio = demographic_system.age_distribution.get('65+', 0.05)
            health_need = (elderly_pop_ratio - 0.05) * 0.04  # Baseline elderly population of 5%
        
        health_change = base_change + health_need + fiscal_space * 0.003
        self.health_investment = max(0.005, min(0.04, 
            self.health_investment + health_change * time_delta))
        
        # Adjust research investment (tends to increase with economic development)
        research_change = base_change + gdp_growth * 0.02 + fiscal_space * 0.002
        self.research_investment = max(0.001, min(0.02, 
            self.research_investment + research_change * time_delta))
        
        # Return updated investment parameters
        return {
            'infrastructure_investment': self.infrastructure_investment,
            'education_investment': self.education_investment,
            'health_investment': self.health_investment,
            'research_investment': self.research_investment
        }
    
    def update_governance_indicators(self):
        """
        Update overall governance indicators based on institutional system updates.
        
        Returns:
            dict: Updated governance indicators
        """
        # Get updated institutional values
        inst = self.institutional_system
        
        # Update core governance quality metrics
        self.governance_index = 0.2 * (inst.government_effectiveness + 
                                      inst.regulatory_quality + 
                                      inst.rule_of_law + 
                                      (1 - inst.corruption_index) + 
                                      inst.voice_accountability)
        
        # Update regulatory burden
        self.regulatory_burden = 0.6 * (1 - inst.regulatory_quality) + 0.4 * inst.corruption_index
        self.regulatory_burden = max(0.1, min(0.9, self.regulatory_burden))
        
        return {
            'governance_index': self.governance_index,
            'regulatory_burden': self.regulatory_burden
        }
    
    def update_regional_governance(self):
        """
        Update regional governance variation based on national indicators.
        
        Returns:
            dict: Regional governance data
        """
        for region in self.regions:
            # Update based on national trends, maintaining regional differences
            self.regional_governance[region]['governance_index'] = (
                0.7 * self.regional_governance[region]['governance_index'] + 
                0.3 * self.governance_index
            )
            
            self.regional_governance[region]['policy_effectiveness'] = (
                0.7 * self.regional_governance[region]['policy_effectiveness'] + 
                0.3 * self.policy_effectiveness
            )
            
            # Investment levels should approximate national levels while keeping regional variance
            for investment_type in ['infrastructure_investment', 'education_investment', 'health_investment']:
                national_value = getattr(self, investment_type)
                self.regional_governance[region][investment_type] = (
                    0.8 * self.regional_governance[region][investment_type] + 
                    0.2 * national_value
                )
                
        return self.regional_governance
    
    def get_policy_impact(self, policy_area):
        """
        Calculate the effective impact of a specific policy area, 
        taking into account policy value and implementation effectiveness.
        
        Args:
            policy_area (str): Name of the policy area
            
        Returns:
            float: Effective policy impact value (0-1 scale)
        """
        if hasattr(self, policy_area):
            policy_value = getattr(self, policy_area)
            # Effective impact is a combination of policy value and implementation effectiveness
            return policy_value * self.policy_effectiveness
        else:
            return 0.0
    
    def get_investment_effectiveness(self, investment_area):
        """
        Calculate the effective impact of a specific investment area,
        taking into account investment level and governance effectiveness.
        
        Args:
            investment_area (str): Name of the investment area
            
        Returns:
            float: Effective investment impact
        """
        if not investment_area.endswith('_investment'):
            investment_area = f"{investment_area}_investment"
            
        if hasattr(self, investment_area):
            investment_level = getattr(self, investment_area)
            # Effective impact combines investment level with governance effectiveness
            # Corruption reduces effectiveness (waste, leakage)
            return investment_level * (0.7 * self.institutional_system.institutional_effectiveness + 
                                     0.3 * (1 - self.institutional_system.corruption_index))
        else:
            return 0.0
    
    def get_institutional_quality(self):
        """
        Get institutional quality metrics.
        
        Returns:
            dict: Key institutional quality indicators
        """
        inst = self.institutional_system
        return {
            'institutional_effectiveness': inst.institutional_effectiveness,
            'corruption_index': inst.corruption_index,
            'rule_of_law': inst.rule_of_law,
            'government_effectiveness': inst.government_effectiveness,
            'regulatory_quality': inst.regulatory_quality,
            'voice_accountability': inst.voice_accountability,
            'political_stability': inst.political_stability
        }
    
    def get_social_indicators(self):
        """
        Get social policy indicators.
        
        Returns:
            dict: Key social indicators
        """
        inst = self.institutional_system
        return {
            'gender_inequality_index': inst.gender_inequality_index,
            'female_labor_participation': inst.female_labor_participation,
            'education_effectiveness': inst.education_effectiveness,
            'healthcare_accessibility': inst.healthcare_accessibility,
            'social_safety_coverage': inst.social_safety_coverage
        }
    
    def get_climate_policy(self):
        """
        Get climate policy level adjusted for implementation effectiveness.
        
        Returns:
            float: Effective climate policy level (0-1 scale)
        """
        return self.get_policy_impact('climate_policy')
    
    def get_institutional_efficiency(self):
        """
        Get overall institutional efficiency.
        
        Returns:
            float: Institutional efficiency (0-1 scale)
        """
        return self.institutional_system.institutional_effectiveness
    
    def get_economic_policy_stance(self):
        """
        Get overall economic policy stance.
        
        Returns:
            dict: Economic policy indicators
        """
        return {
            'fiscal_policy': self.fiscal_policy,
            'monetary_policy': self.monetary_policy,
            'trade_policy': self.trade_policy,
            'industrial_policy': self.industrial_policy,
            'agricultural_policy': self.agricultural_policy,
            'regulatory_burden': self.regulatory_burden
        }
    
    def get_governance_quality(self):
        """
        Get overall governance quality metrics.
        
        Returns:
            dict: Governance quality indicators
        """
        return {
            'governance_index': self.governance_index,
            'policy_effectiveness': self.policy_effectiveness,
            'policy_coherence': self.policy_coherence,
            'policy_implementation': self.policy_implementation
        }
    
    def record_history(self, year):
        """
        Record governance indicators for historical tracking.
        
        Args:
            year (int): Current simulation year
        """
        self.history['years'].append(year)
        self.history['governance_index'].append(self.governance_index)
        self.history['policy_effectiveness'].append(self.policy_effectiveness) 
        self.history['institutional_effectiveness'].append(self.institutional_system.institutional_effectiveness)
        self.history['corruption_index'].append(self.institutional_system.corruption_index)
        self.history['infrastructure_investment'].append(self.infrastructure_investment)
        self.history['education_investment'].append(self.education_investment)
        self.history['health_investment'].append(self.health_investment)
        self.history['environmental_policy'].append(self.environmental_policy)
        self.history['trade_policy'].append(self.trade_policy)
        self.history['regulatory_quality'].append(self.institutional_system.regulatory_quality)
    
    def step(self, year, economic_system=None, environmental_system=None, demographic_system=None, infrastructure_system=None):
        """
        Execute a single time step of the governance system simulation.
        
        Args:
            year (int): Current simulation year
            economic_system: Economic system instance for accessing economic indicators
            environmental_system: Environmental system instance for accessing environmental indicators
            demographic_system: Demographic system instance for accessing demographic indicators
            infrastructure_system: Infrastructure system instance for accessing infrastructure indicators
            
        Returns:
            dict: Updated governance indicators and state
        """
        # Log the current step
        print(f"Executing governance step for year {year}")
        
        # Update current year
        self.current_year = year
        
        # Get inputs from other systems if available
        economic_inputs = self._get_economic_inputs(economic_system) if economic_system else {}
        environmental_inputs = self._get_environmental_inputs(environmental_system) if environmental_system else {}
        demographic_inputs = self._get_demographic_inputs(demographic_system) if demographic_system else {}
        infrastructure_inputs = self._get_infrastructure_inputs(infrastructure_system) if infrastructure_system else {}
        
        # Update institutional system
        if hasattr(self, 'institutional_system'):
            if hasattr(self.institutional_system, 'update'):
                self.institutional_system.update(
                    year=year,
                    economic_growth=economic_inputs.get('gdp_growth_rate', None),
                    education_level=demographic_inputs.get('education_index', None)
                )
            elif hasattr(self.institutional_system, 'step'):
                # If institutional_system is another GovernanceSystem, call step instead
                self.institutional_system.step(
                    year=year,
                    economic_system={'gdp_growth_rate': economic_inputs.get('gdp_growth_rate', None)},
                    environmental_system={},  # Pass empty dict to satisfy the required argument
                    demographic_system={'education_index': demographic_inputs.get('education_index', None)},
                    infrastructure_system=None  # This is optional
                )
            else:
                # Log a warning but continue execution
                print("Warning: institutional_system doesn't have update or step methods")
        
        # Update policies based on current conditions and other systems
        self.update_policies(
            year=year,
            economic_system=economic_system,
            demographic_system=demographic_system,
            environmental_system=environmental_system
        )
        
        # Update governance indicators based on institutional changes
        self.initialize_governance_indicators()
        
        # Update regional governance
        self._update_regional_governance(
            economic_inputs=economic_inputs,
            demographic_inputs=demographic_inputs
        )
        
        # Apply climate change adaptation policies if environmental system available
        if environmental_system:
            self._apply_climate_adaptation_policies(environmental_system)
        
        # Update investment allocations based on infrastructure needs
        if infrastructure_system:
            self._update_infrastructure_investment(infrastructure_system)
        
        # Record current state in history
        self._record_history()
        
        return {
            'governance_index': self.governance_index,
            'policy_effectiveness': self.policy_effectiveness,
            'institutional_effectiveness': self.institutional_system.institutional_effectiveness,
            'corruption_index': self.institutional_system.corruption_index,
            'infrastructure_investment': self.infrastructure_investment,
            'education_investment': self.education_investment,
            'health_investment': self.health_investment,
            'environmental_policy': self.environmental_policy,
            'trade_policy': self.trade_policy,
            'regulatory_quality': self.institutional_system.regulatory_quality
        }
    
    def _get_economic_inputs(self, economic_system):
        """Extract relevant inputs from the economic system."""
        inputs = {}
        
        if hasattr(economic_system, 'gdp_growth_rate'):
            inputs['gdp_growth_rate'] = economic_system.gdp_growth_rate
        
        if hasattr(economic_system, 'inflation_rate'):
            inputs['inflation_rate'] = economic_system.inflation_rate
        
        if hasattr(economic_system, 'unemployment_rate'):
            inputs['unemployment_rate'] = economic_system.unemployment_rate
        
        if hasattr(economic_system, 'trade_balance'):
            inputs['trade_balance'] = economic_system.trade_balance
        
        if hasattr(economic_system, 'gdp_per_capita'):
            inputs['gdp_per_capita'] = economic_system.gdp_per_capita
        
        return inputs
    
    def _get_environmental_inputs(self, environmental_system):
        """Extract relevant inputs from the environmental system."""
        inputs = {}
        
        if hasattr(environmental_system, 'flood_risk'):
            inputs['flood_risk'] = environmental_system.flood_risk
        
        if hasattr(environmental_system, 'climate_vulnerability_index'):
            inputs['climate_vulnerability_index'] = environmental_system.climate_vulnerability_index
        
        if hasattr(environmental_system, 'environmental_health_index'):
            inputs['environmental_health_index'] = environmental_system.environmental_health_index
        
        if hasattr(environmental_system, 'water_quality_index'):
            inputs['water_quality_index'] = environmental_system.water_quality_index
        
        return inputs
    
    def _get_demographic_inputs(self, demographic_system):
        """Extract relevant inputs from the demographic system."""
        inputs = {}
        
        if hasattr(demographic_system, 'hdi'):
            inputs['human_development_index'] = demographic_system.hdi
        
        if hasattr(demographic_system, 'education_index'):
            inputs['education_index'] = demographic_system.education_index
        
        if hasattr(demographic_system, 'urbanization_rate'):
            inputs['urbanization_rate'] = demographic_system.urbanization_rate
        
        if hasattr(demographic_system, 'population_growth_rate'):
            inputs['population_growth_rate'] = demographic_system.population_growth_rate
        
        return inputs
    
    def _get_infrastructure_inputs(self, infrastructure_system):
        """Extract relevant inputs from the infrastructure system."""
        inputs = {}
        
        if hasattr(infrastructure_system, 'infrastructure_quality_index'):
            inputs['infrastructure_quality_index'] = infrastructure_system.infrastructure_quality_index
        
        if hasattr(infrastructure_system, 'electricity_coverage'):
            inputs['electricity_coverage'] = infrastructure_system.electricity_coverage
        
        if hasattr(infrastructure_system, 'water_supply_coverage'):
            inputs['water_supply_coverage'] = infrastructure_system.water_supply_coverage
        
        if hasattr(infrastructure_system, 'road_quality'):
            inputs['road_quality'] = infrastructure_system.road_quality
        
        return inputs
    
    def _update_regional_governance(self, economic_inputs=None, demographic_inputs=None):
        """Update regional governance based on inputs from other systems."""
        for region in self.regions:
            # Base updates
            for key in self.regional_governance[region]:
                # Apply small random change
                self.regional_governance[region][key] *= (1 + np.random.uniform(-0.03, 0.03))
            
            # Apply economic effects if available
            if economic_inputs and 'gdp_growth_rate' in economic_inputs:
                growth_effect = economic_inputs['gdp_growth_rate'] * 0.1
                self.regional_governance[region]['governance_index'] += growth_effect
                self.regional_governance[region]['policy_effectiveness'] += growth_effect * 0.8
            
            # Apply demographic effects if available
            if demographic_inputs and 'education_index' in demographic_inputs:
                education_effect = demographic_inputs['education_index'] * 0.05
                self.regional_governance[region]['governance_index'] += education_effect
                
            # Ensure values stay in reasonable range
            for key in self.regional_governance[region]:
                if not key.endswith('investment'):  # For index values
                    self.regional_governance[region][key] = max(0.1, min(0.9, self.regional_governance[region][key]))
    
    def _apply_climate_adaptation_policies(self, environmental_system):
        """Apply climate change adaptation policies based on environmental conditions."""
        if hasattr(environmental_system, 'climate_vulnerability_index'):
            vulnerability = environmental_system.climate_vulnerability_index
            
            # Increase climate policy stringency with higher vulnerability
            self.climate_policy = min(0.9, self.climate_policy + vulnerability * 0.05)
            
            # Adjust water management policy based on water stress
            if hasattr(environmental_system, 'water_stress_index'):
                water_stress = environmental_system.water_stress_index
                self.water_management_policy = min(0.9, self.water_management_policy + water_stress * 0.05)
    
    def _update_infrastructure_investment(self, infrastructure_system):
        """Update infrastructure investment priorities based on infrastructure needs."""
        # Increase overall infrastructure investment with lower infrastructure quality
        if hasattr(infrastructure_system, 'infrastructure_quality_index'):
            quality_gap = 1 - infrastructure_system.infrastructure_quality_index
            self.infrastructure_investment = min(0.08, self.infrastructure_investment + quality_gap * 0.01)
        
        # Adjust investment allocation based on specific needs
        if hasattr(infrastructure_system, 'electricity_reliability') and hasattr(infrastructure_system, 'road_quality'):
            # If electricity reliability is worse than road quality, shift investment towards energy
            if infrastructure_system.electricity_reliability < infrastructure_system.road_quality:
                energy_shift = 0.05 * (infrastructure_system.road_quality - infrastructure_system.electricity_reliability)
                self.investment_allocation['energy'] = min(0.5, self.investment_allocation['energy'] + energy_shift)
                self.investment_allocation['transport'] = max(0.2, self.investment_allocation['transport'] - energy_shift)
            else:
                # Otherwise shift towards transport
                transport_shift = 0.05 * (infrastructure_system.electricity_reliability - infrastructure_system.road_quality)
                self.investment_allocation['transport'] = min(0.5, self.investment_allocation['transport'] + transport_shift)
                self.investment_allocation['energy'] = max(0.2, self.investment_allocation['energy'] - transport_shift)

    def _record_history(self):
        """Record current governance indicators to history."""
        self.history['years'].append(self.current_year)
        self.history['governance_index'].append(self.governance_index)
        self.history['policy_effectiveness'].append(self.policy_effectiveness) 
        self.history['institutional_effectiveness'].append(self.institutional_system.institutional_effectiveness)
        self.history['corruption_index'].append(self.institutional_system.corruption_index)
        self.history['infrastructure_investment'].append(self.infrastructure_investment)
        self.history['education_investment'].append(self.education_investment)
        self.history['health_investment'].append(self.health_investment)
        self.history['environmental_policy'].append(self.environmental_policy)
        self.history['trade_policy'].append(self.trade_policy)
        self.history['regulatory_quality'].append(self.institutional_system.regulatory_quality)
