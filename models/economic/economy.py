#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Economic system model for Bangladesh simulation.
This module implements multi-sector economic dynamics including garment manufacturing,
agriculture, remittances, and emerging tech sectors.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import networkx as nx

from models.economic.financial_markets import FinancialMarkets
from models.economic.trade import TradeSystem
from models.economic.sectors import (
    GarmentSector,
    AgriculturalSector,
    TechnologySector,
    InformalEconomySector
)


class EconomicSystem:
    """
    Multi-sector economic model of Bangladesh economy incorporating formal and informal
    sectors, trade dynamics, financial frictions, and sectoral interactions.
    """
    
    def __init__(self, config, data_loader):
        """
        Initialize the economic system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the economic system
            data_loader (DataLoader): Data loading utility for economic data
        """
        self.config = config
        self.data_loader = data_loader
        
        # Load initial economic data
        self.economic_data = data_loader.load_economic_data()
        
        # Set up time-related variables
        self.current_year = config.get('start_year', 2023)
        self.time_step = config.get('time_step', 1.0)
        
        # Initialize exchange rate and inflation parameters
        self.exchange_rate = self.economic_data.get('initial_exchange_rate', 85.0)  # BDT per USD
        self.inflation_rate = self.economic_data.get('initial_inflation_rate', 0.055)  # 5.5%
        
        # GDP components
        self.gdp = self.economic_data.get('initial_gdp', 416.0)  # Billion USD
        self.gdp_growth_rate = self.economic_data.get('initial_gdp_growth', 0.06)  # 6%
        
        # Initialize economic sectors
        self._init_sectors()
        
        # Initialize financial markets (formal, informal, microfinance)
        self.financial_markets = FinancialMarkets(
            config=config.get('financial_markets', {}),
            economic_data=self.economic_data
        )
        
        # Initialize trade system (exports, imports, tariffs)
        self.trade_system = TradeSystem(
            config=config.get('trade', {}),
            economic_data=self.economic_data,
            exchange_rate=self.exchange_rate
        )
        
        # Set up economic network for intersectoral relationships
        self._init_economic_network()
        
        print("Economic system initialized")
    
    def _init_sectors(self):
        """Initialize all economic sectors."""
        # Garment manufacturing sector (largest export sector)
        self.garment_sector = GarmentSector(
            config=self.config.get('garment_sector', {}),
            economic_data=self.economic_data
        )
        
        # Agricultural sector (rice, jute, tea)
        self.agricultural_sector = AgriculturalSector(
            config=self.config.get('agricultural_sector', {}),
            economic_data=self.economic_data
        )
        
        # Technology and services sector (emerging)
        self.technology_sector = TechnologySector(
            config=self.config.get('technology_sector', {}),
            economic_data=self.economic_data
        )
        
        # Informal economy sector
        self.informal_sector = InformalEconomySector(
            config=self.config.get('informal_sector', {}),
            economic_data=self.economic_data
        )
        
        # Store all sectors in a list for iteration
        self.sectors = [
            self.garment_sector,
            self.agricultural_sector,
            self.technology_sector,
            self.informal_sector
        ]
    
    def _init_economic_network(self):
        """Initialize the economic network representing intersectoral relationships."""
        self.economic_network = nx.DiGraph()
        
        # Add nodes for each sector
        for sector in self.sectors:
            self.economic_network.add_node(sector.name, 
                                          output=sector.output,
                                          employment=sector.employment)
        
        # Add node for households
        self.economic_network.add_node('households', 
                                      income=self.economic_data.get('household_income', 0.0),
                                      consumption=self.economic_data.get('household_consumption', 0.0))
        
        # Add node for government
        self.economic_network.add_node('government', 
                                      revenue=self.economic_data.get('government_revenue', 0.0),
                                      expenditure=self.economic_data.get('government_expenditure', 0.0))
        
        # Add node for external sector (rest of world)
        self.economic_network.add_node('external', 
                                      exports=self.trade_system.total_exports,
                                      imports=self.trade_system.total_imports,
                                      remittances=self.economic_data.get('remittances', 0.0))
        
        # Add edges with initial flow values from input-output tables
        io_table = self.economic_data.get('input_output_table', {})
        for source, targets in io_table.items():
            for target, value in targets.items():
                self.economic_network.add_edge(source, target, flow=value)
    
    def update_exchange_rate(self, environmental_system=None, governance_system=None):
        """
        Update the exchange rate based on trade balance, capital flows, and policy interventions.
        
        Args:
            environmental_system: Environmental system for climate impacts on exports
            governance_system: Governance system for monetary policy decisions
        """
        # Base exchange rate dynamics
        trade_balance_effect = 0.01 * (self.trade_system.total_imports - self.trade_system.total_exports) / self.gdp
        
        # External factors (global market conditions)
        global_volatility = np.random.normal(0, 0.02)
        
        # Policy intervention if governance system is provided
        policy_effect = 0.0
        if governance_system:
            policy_effect = getattr(governance_system, 'monetary_policy_effect', 0.0)
        
        # Climate event impacts if environmental system is provided
        climate_effect = 0.0
        if environmental_system and hasattr(environmental_system, 'extreme_event_impact'):
            climate_effect = 0.03 * environmental_system.extreme_event_impact
        
        # Calculate new exchange rate (BDT per USD)
        exchange_rate_change = trade_balance_effect + global_volatility + policy_effect + climate_effect
        self.exchange_rate *= (1 + exchange_rate_change)
        
        # Ensure exchange rate stays within realistic bounds
        self.exchange_rate = max(min(self.exchange_rate, 150.0), 50.0)
        
        return self.exchange_rate
    
    def update_inflation(self, governance_system=None):
        """
        Update inflation rate based on economic conditions and monetary policy.
        
        Args:
            governance_system: Governance system for monetary policy effects
        """
        # Base inflation dynamics
        money_supply_growth = 0.08  # Typical growth in money supply
        
        # Output gap effect (overheating economy increases inflation)
        output_gap = (self.gdp_growth_rate - 0.06) / 0.06  # Deviation from potential growth
        output_gap_effect = 0.5 * output_gap
        
        # External price shocks (e.g., energy prices)
        external_shock = np.random.normal(0, 0.01)
        
        # Exchange rate pass-through
        exchange_rate_effect = 0.2 * (self.exchange_rate / self.economic_data.get('initial_exchange_rate', 85.0) - 1)
        
        # Monetary policy effect if governance system is provided
        policy_effect = 0.0
        if governance_system:
            policy_effect = -0.5 * getattr(governance_system, 'interest_rate_adjustment', 0.0)
        
        # Calculate new inflation rate
        inflation_components = {
            'money_supply': money_supply_growth,
            'output_gap': output_gap_effect,
            'external_shock': external_shock,
            'exchange_rate': exchange_rate_effect,
            'policy_effect': policy_effect
        }
        
        self.inflation_rate = 0.4 * self.inflation_rate + 0.6 * sum(inflation_components.values())
        
        # Ensure inflation stays within realistic bounds
        self.inflation_rate = max(min(self.inflation_rate, 0.15), 0.01)
        
        return self.inflation_rate, inflation_components
    
    def update_gdp(self):
        """Update GDP based on sectoral outputs and trade."""
        # Sum output from all sectors
        sectoral_output = sum(sector.output for sector in self.sectors)
        
        # Add net exports
        net_exports = self.trade_system.total_exports - self.trade_system.total_imports
        
        # Calculate new GDP
        self.gdp = sectoral_output + net_exports
        
        # Calculate GDP growth rate
        self.gdp_growth_rate = (self.gdp / self.economic_data.get('previous_gdp', self.gdp)) - 1
        
        # Update previous GDP for next iteration
        self.economic_data['previous_gdp'] = self.gdp
        
        return self.gdp, self.gdp_growth_rate
    
    def step(self, year, environmental_system=None, demographic_system=None, 
             infrastructure_system=None, governance_system=None):
        """
        Advance the economic system by one time step.
        
        Args:
            year (int): Current simulation year
            environmental_system: Environmental system for climate impacts
            demographic_system: Demographic system for labor supply
            infrastructure_system: Infrastructure system for transport and utility impacts
            governance_system: Governance system for policy interventions
            
        Returns:
            dict: Results of the economic step
        """
        self.current_year = year
        print(f"Economic system step: year {year}")
        
        # Update exchange rate
        exchange_rate = self.update_exchange_rate(
            environmental_system=environmental_system,
            governance_system=governance_system
        )
        
        # Update financial markets
        financial_markets_results = self.financial_markets.step(
            exchange_rate=exchange_rate,
            governance_system=governance_system
        )
        
        # Process demographic inputs if available
        labor_supply = 100_000_000  # Default value - approximately 60% of total population
        if demographic_system:
            # Try to access labor directly if available
            if hasattr(demographic_system, 'labor_force'):
                labor_supply = demographic_system.labor_force
            elif hasattr(demographic_system, 'get_labor_supply'):
                labor_supply = demographic_system.get_labor_supply()
            else:
                # Estimate labor supply from total population
                labor_supply = demographic_system.total_population * 0.6  # Assume 60% labor participation
        
        # Process infrastructure inputs if available
        infrastructure_impacts = {
            'transport_efficiency': 0.7,  # Default value (70% efficiency)
            'electricity_reliability': 0.6,  # Default value (60% reliability)
            'telecom_coverage': 0.5  # Default value (50% coverage)
        }
        
        if infrastructure_system:
            # Access infrastructure attributes directly based on infrastructure module components
            # Transportation network metrics
            if hasattr(infrastructure_system, 'transport_efficiency'):
                infrastructure_impacts['transport_efficiency'] = infrastructure_system.transport_efficiency
            elif hasattr(infrastructure_system, 'transportation_quality_index'):
                infrastructure_impacts['transport_efficiency'] = infrastructure_system.transportation_quality_index
            
            # Energy infrastructure metrics
            if hasattr(infrastructure_system, 'electricity_reliability'):
                infrastructure_impacts['electricity_reliability'] = infrastructure_system.electricity_reliability
            elif hasattr(infrastructure_system, 'energy_reliability'):
                infrastructure_impacts['electricity_reliability'] = infrastructure_system.energy_reliability
            
            # Telecommunications metrics
            if hasattr(infrastructure_system, 'telecom_coverage'):
                infrastructure_impacts['telecom_coverage'] = infrastructure_system.telecom_coverage
            elif hasattr(infrastructure_system, 'telecommunications_coverage'):
                infrastructure_impacts['telecom_coverage'] = infrastructure_system.telecommunications_coverage
        
        # Process environmental inputs if available
        environmental_impacts = None
        if environmental_system:
            environmental_impacts = {
                'agricultural_productivity': environmental_system.get_agricultural_productivity(),
                'flood_impacts': environmental_system.get_flood_impacts(),
                'cyclone_damage': environmental_system.get_cyclone_damage()
            }
        
        # Update each sector
        sector_results = {}
        for sector in self.sectors:
            sector_result = sector.step(
                exchange_rate=exchange_rate,
                inflation_rate=self.inflation_rate,
                labor_supply=labor_supply,
                infrastructure=infrastructure_impacts,
                environmental=environmental_impacts,
                financial_markets=financial_markets_results,
                governance=governance_system
            )
            sector_results[sector.name] = sector_result
        
        # Update trade system
        trade_results = self.trade_system.step(
            exchange_rate=exchange_rate,
            sectors=self.sectors,
            infrastructure=infrastructure_impacts,
            governance=governance_system
        )
        
        # Update inflation
        inflation_rate, inflation_components = self.update_inflation(governance_system)
        
        # Update GDP
        gdp, gdp_growth = self.update_gdp()
        
        # Update economic network
        self._update_economic_network()
        
        # Compile results
        results = {
            'year': year,
            'gdp': gdp,
            'gdp_growth': gdp_growth,
            'exchange_rate': exchange_rate,
            'inflation_rate': inflation_rate,
            'inflation_components': inflation_components,
            'sectors': sector_results,
            'trade': trade_results,
            'financial_markets': financial_markets_results
        }
        
        return results
    
    def _update_economic_network(self):
        """Update the economic network with current flow values."""
        # Update node attributes
        for sector in self.sectors:
            self.economic_network.nodes[sector.name]['output'] = sector.output
            self.economic_network.nodes[sector.name]['employment'] = sector.employment
        
        self.economic_network.nodes['external']['exports'] = self.trade_system.total_exports
        self.economic_network.nodes['external']['imports'] = self.trade_system.total_imports
        
        # We would update edge attributes based on current inter-sectoral flows
        # This would typically come from updated input-output calculations
        # For simplicity, we're not implementing the full calculation here
    
    def get_sector_outputs(self):
        """Get the output values for all sectors."""
        return {sector.name: sector.output for sector in self.sectors}
    
    def get_sector_employment(self):
        """Get the employment values for all sectors."""
        return {sector.name: sector.employment for sector in self.sectors}
    
    def get_economic_indicators(self):
        """Get key economic indicators."""
        return {
            'gdp': self.gdp,
            'gdp_growth': self.gdp_growth_rate,
            'exchange_rate': self.exchange_rate,
            'inflation_rate': self.inflation_rate,
            'exports': self.trade_system.total_exports,
            'imports': self.trade_system.total_imports,
            'formal_credit': self.financial_markets.formal_credit_volume,
            'informal_credit': self.financial_markets.informal_credit_volume,
            'microfinance_credit': self.financial_markets.microfinance_credit_volume
        }
