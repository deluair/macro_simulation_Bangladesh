#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Financial markets model for the Bangladesh economic system.
This module simulates formal and informal credit markets, microfinance operations,
and financial frictions across rural and urban areas.
"""

import numpy as np
import pandas as pd


class FinancialMarkets:
    """
    Model of financial markets in Bangladesh including formal banking sector,
    informal moneylenders, and microfinance institutions.
    """
    
    def __init__(self, config, economic_data):
        """
        Initialize the financial markets model.
        
        Args:
            config (dict): Configuration parameters for financial markets
            economic_data (dict): Economic data including financial sector data
        """
        self.config = config
        self.economic_data = economic_data
        
        # Financial market sizes
        self.formal_credit_volume = economic_data.get('formal_credit_volume', 100.0)  # Billion BDT
        self.informal_credit_volume = economic_data.get('informal_credit_volume', 40.0)  # Billion BDT
        self.microfinance_credit_volume = economic_data.get('microfinance_credit_volume', 30.0)  # Billion BDT
        
        # Interest rates
        self.formal_lending_rate = economic_data.get('formal_lending_rate', 0.09)  # 9%
        self.informal_lending_rate = economic_data.get('informal_lending_rate', 0.30)  # 30%
        self.microfinance_lending_rate = economic_data.get('microfinance_lending_rate', 0.15)  # 15%
        
        # Access parameters (percentage of population with access)
        self.urban_formal_access = economic_data.get('urban_formal_access', 0.60)  # 60%
        self.rural_formal_access = economic_data.get('rural_formal_access', 0.25)  # 25%
        self.microfinance_coverage = economic_data.get('microfinance_coverage', 0.35)  # 35%
        
        # Default rates
        self.formal_default_rate = economic_data.get('formal_default_rate', 0.05)  # 5%
        self.informal_default_rate = economic_data.get('informal_default_rate', 0.15)  # 15%
        self.microfinance_default_rate = economic_data.get('microfinance_default_rate', 0.02)  # 2%
        
        # Collateral requirements (as multiple of loan amount)
        self.formal_collateral_req = economic_data.get('formal_collateral_req', 1.5)  # 150%
        self.informal_collateral_req = economic_data.get('informal_collateral_req', 0.5)  # 50%
        self.microfinance_collateral_req = economic_data.get('microfinance_collateral_req', 0.0)  # 0%
        
        # Regional distribution (urban vs rural share)
        self.urban_credit_share = economic_data.get('urban_credit_share', 0.70)  # 70% urban
        
        print("Financial markets initialized")
    
    def update_interest_rates(self, inflation_rate=None, policy_rate=None):
        """
        Update interest rates based on inflation and monetary policy.
        
        Args:
            inflation_rate (float): Current inflation rate
            policy_rate (float): Central bank policy rate
        """
        if inflation_rate is None:
            inflation_rate = 0.055  # Default 5.5%
            
        if policy_rate is None:
            policy_rate = 0.065  # Default 6.5%
            
        # Formal lending rate responds to monetary policy and inflation
        policy_impact = 0.8 * (policy_rate - 0.065)  # Sensitivity to policy changes
        inflation_impact = 0.3 * (inflation_rate - 0.055)  # Sensitivity to inflation changes
        
        # Update formal lending rate
        prev_rate = self.formal_lending_rate
        self.formal_lending_rate = self.formal_lending_rate + policy_impact + inflation_impact
        self.formal_lending_rate = max(min(self.formal_lending_rate, 0.15), 0.05)  # Bound rate
        
        # Informal lending rate has weaker correlation with policy but stronger with formal rate
        formal_rate_change = self.formal_lending_rate - prev_rate
        self.informal_lending_rate = self.informal_lending_rate + 0.2 * formal_rate_change + 0.5 * inflation_impact
        self.informal_lending_rate = max(min(self.informal_lending_rate, 0.50), 0.20)  # Bound rate
        
        # Microfinance rates are more stable but still affected by overall conditions
        self.microfinance_lending_rate = self.microfinance_lending_rate + 0.1 * formal_rate_change + 0.2 * inflation_impact
        self.microfinance_lending_rate = max(min(self.microfinance_lending_rate, 0.25), 0.12)  # Bound rate
        
        rates = {
            'formal_lending_rate': self.formal_lending_rate,
            'informal_lending_rate': self.informal_lending_rate,
            'microfinance_lending_rate': self.microfinance_lending_rate
        }
        
        return rates
    
    def update_access_rates(self, infrastructure_system=None, demographic_system=None):
        """
        Update financial access rates based on infrastructure and demographics.
        
        Args:
            infrastructure_system: Infrastructure system for connectivity impacts
            demographic_system: Demographic system for income and education effects
        """
        # Base trend of increasing financial inclusion
        base_formal_access_growth = 0.02  # 2% annual increase in formal access
        base_microfinance_growth = 0.01  # 1% annual increase in microfinance access
        
        # Infrastructure impact if available
        infra_impact_urban = 0.0
        infra_impact_rural = 0.0
        if infrastructure_system:
            telecom_effect = 0.05 * (infrastructure_system.get_telecom_coverage() - 0.5)
            transport_effect = 0.03 * (infrastructure_system.get_transport_efficiency() - 0.5)
            infra_impact_urban = 0.5 * (telecom_effect + transport_effect)
            infra_impact_rural = 0.8 * (telecom_effect + transport_effect)
        
        # Demographic impact if available
        demo_impact_urban = 0.0
        demo_impact_rural = 0.0
        if demographic_system:
            education_effect = 0.04 * (demographic_system.get_education_level() - 0.5)
            urbanization_effect = 0.02 * (demographic_system.get_urbanization_rate() - 0.5)
            demo_impact_urban = education_effect + 0.2 * urbanization_effect
            demo_impact_rural = education_effect - 0.1 * urbanization_effect
        
        # Update urban access
        self.urban_formal_access += base_formal_access_growth + infra_impact_urban + demo_impact_urban
        self.urban_formal_access = max(min(self.urban_formal_access, 0.95), self.urban_formal_access)
        
        # Update rural access
        self.rural_formal_access += base_formal_access_growth + infra_impact_rural + demo_impact_rural
        self.rural_formal_access = max(min(self.rural_formal_access, 0.80), self.rural_formal_access)
        
        # Update microfinance coverage
        self.microfinance_coverage += base_microfinance_growth + 0.5 * (infra_impact_rural + demo_impact_rural)
        self.microfinance_coverage = max(min(self.microfinance_coverage, 0.70), self.microfinance_coverage)
        
        access_rates = {
            'urban_formal_access': self.urban_formal_access,
            'rural_formal_access': self.rural_formal_access,
            'microfinance_coverage': self.microfinance_coverage
        }
        
        return access_rates
    
    def update_default_rates(self, environmental_system=None, economic_system=None):
        """
        Update loan default rates based on environmental and economic conditions.
        
        Args:
            environmental_system: Environmental system for disaster impacts
            economic_system: Economic system for overall economic conditions
        """
        # Base default rates adjustments based on annual trends
        base_formal_default_change = 0.001  # Slight annual increase
        base_informal_default_change = 0.002
        base_microfinance_default_change = 0.001
        
        # Environmental impacts if available
        env_impact = 0.0
        if environmental_system:
            flood_impact = 0.1 * environmental_system.get_flood_impacts()
            cyclone_impact = 0.2 * environmental_system.get_cyclone_damage()
            drought_impact = 0.05 * environmental_system.get_drought_severity()
            env_impact = flood_impact + cyclone_impact + drought_impact
        
        # Economic impacts if available
        econ_impact = 0.0
        if economic_system:
            growth_effect = -0.2 * (economic_system.gdp_growth_rate - 0.05)  # Lower growth increases defaults
            inflation_effect = 0.1 * (economic_system.inflation_rate - 0.05)  # Higher inflation increases defaults
            econ_impact = growth_effect + inflation_effect
        
        # Update default rates
        self.formal_default_rate += base_formal_default_change + 0.4 * env_impact + 0.6 * econ_impact
        self.formal_default_rate = max(min(self.formal_default_rate, 0.15), 0.01)
        
        self.informal_default_rate += base_informal_default_change + 0.6 * env_impact + 0.4 * econ_impact
        self.informal_default_rate = max(min(self.informal_default_rate, 0.30), 0.05)
        
        self.microfinance_default_rate += base_microfinance_default_change + 0.7 * env_impact + 0.3 * econ_impact
        self.microfinance_default_rate = max(min(self.microfinance_default_rate, 0.10), 0.01)
        
        default_rates = {
            'formal_default_rate': self.formal_default_rate,
            'informal_default_rate': self.informal_default_rate,
            'microfinance_default_rate': self.microfinance_default_rate
        }
        
        return default_rates
    
    def update_credit_volumes(self, gdp_growth=None):
        """
        Update credit volumes based on economic growth and financial conditions.
        
        Args:
            gdp_growth (float): GDP growth rate
        """
        if gdp_growth is None:
            gdp_growth = 0.06  # Default 6%
            
        # Credit volume growth is typically a multiple of GDP growth
        formal_credit_growth = 1.5 * gdp_growth
        informal_credit_growth = 0.8 * gdp_growth
        microfinance_credit_growth = 1.2 * gdp_growth
        
        # Access rate effects
        access_effect_formal = 0.5 * (self.urban_formal_access * self.urban_credit_share + 
                                     self.rural_formal_access * (1 - self.urban_credit_share) - 0.5)
        access_effect_microfinance = 0.7 * (self.microfinance_coverage - 0.35)
        
        # Interest rate effects (inverse relationship)
        rate_effect_formal = -0.2 * (self.formal_lending_rate - 0.09)
        rate_effect_informal = -0.1 * (self.informal_lending_rate - 0.30)
        rate_effect_microfinance = -0.15 * (self.microfinance_lending_rate - 0.15)
        
        # Update credit volumes
        self.formal_credit_volume *= (1 + formal_credit_growth + access_effect_formal + rate_effect_formal)
        self.informal_credit_volume *= (1 + informal_credit_growth + rate_effect_informal)
        self.microfinance_credit_volume *= (1 + microfinance_credit_growth + access_effect_microfinance + rate_effect_microfinance)
        
        # Ensure volumes stay positive
        self.formal_credit_volume = max(self.formal_credit_volume, 10.0)
        self.informal_credit_volume = max(self.informal_credit_volume, 5.0)
        self.microfinance_credit_volume = max(self.microfinance_credit_volume, 5.0)
        
        credit_volumes = {
            'formal_credit_volume': self.formal_credit_volume,
            'informal_credit_volume': self.informal_credit_volume,
            'microfinance_credit_volume': self.microfinance_credit_volume,
            'total_credit_volume': self.formal_credit_volume + self.informal_credit_volume + self.microfinance_credit_volume
        }
        
        return credit_volumes
    
    def step(self, exchange_rate=None, governance_system=None):
        """
        Advance financial markets by one time step.
        
        Args:
            exchange_rate (float): Current exchange rate (BDT per USD)
            governance_system: Governance system for monetary policy
            
        Returns:
            dict: Financial market results after the step
        """
        # Get monetary policy from governance system if available
        policy_rate = 0.05  # Default policy rate (5%)
        if governance_system:
            # Try different ways to get the policy rate
            if hasattr(governance_system, 'policy_rate'):
                policy_rate = governance_system.policy_rate
            elif hasattr(governance_system, 'get_policy_rate'):
                policy_rate = governance_system.get_policy_rate()
            elif hasattr(governance_system, 'get_policy_impact'):
                # Use policy impact as a proxy if available
                # Assume we want monetary policy impact
                policy_rate = 0.05 + governance_system.get_policy_impact('monetary') * 0.01
        
        # Get inflation from economic data
        inflation_rate = self.economic_data.get('inflation_rate', 0.055)
        
        # Get GDP growth from economic data
        gdp_growth = self.economic_data.get('gdp_growth', 0.06)
        
        # Update interest rates
        interest_rates = self.update_interest_rates(inflation_rate, policy_rate)
        
        # Update credit volumes
        credit_volumes = self.update_credit_volumes(gdp_growth)
        
        # Update default rates
        default_rates = self.update_default_rates()
        
        # Update access rates
        access_rates = self.update_access_rates()
        
        # Calculate financial friction indexes
        urban_rural_gap = self.urban_formal_access - self.rural_formal_access
        formal_informal_rate_gap = self.informal_lending_rate - self.formal_lending_rate
        
        # Compile results
        results = {
            'interest_rates': interest_rates,
            'credit_volumes': credit_volumes,
            'default_rates': default_rates,
            'access_rates': access_rates,
            'urban_rural_gap': urban_rural_gap,
            'formal_informal_rate_gap': formal_informal_rate_gap
        }
        
        return results
    
    def get_credit_allocation(self):
        """Get the sectoral allocation of credit."""
        # This would provide a breakdown of credit by sector
        # Simplified implementation for now
        total_credit = self.formal_credit_volume + self.informal_credit_volume + self.microfinance_credit_volume
        
        # Indicative sectoral allocation (would be more dynamic in full model)
        allocation = {
            'agriculture': 0.2 * total_credit,
            'garment': 0.25 * total_credit,
            'services': 0.15 * total_credit,
            'technology': 0.05 * total_credit,
            'housing': 0.15 * total_credit,
            'other_manufacturing': 0.1 * total_credit,
            'other': 0.1 * total_credit
        }
        
        return allocation
