#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trade system model for Bangladesh simulation.
This module implements international trade dynamics including exports,
imports, tariffs, and shipping logistics.
"""

import numpy as np
import pandas as pd


class TradeSystem:
    """
    Trade system model simulating Bangladesh's international trade relations,
    export competitiveness, import dependencies, and shipping logistics.
    """
    
    def __init__(self, config, economic_data, exchange_rate):
        """
        Initialize the trade system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the trade system
            economic_data (dict): Economic data including trade statistics
            exchange_rate (float): Initial exchange rate (BDT per USD)
        """
        self.config = config
        self.economic_data = economic_data
        self.exchange_rate = exchange_rate
        
        # Initialize export variables
        self.total_exports = economic_data.get('initial_exports', 40.0)  # Billion USD
        self.exports_by_sector = {
            'garment': economic_data.get('garment_exports', 30.0),  # Billion USD
            'agricultural': economic_data.get('agricultural_exports', 3.0),  # Billion USD
            'leather': economic_data.get('leather_exports', 1.0),  # Billion USD
            'jute': economic_data.get('jute_exports', 0.8),  # Billion USD
            'technology': economic_data.get('technology_exports', 1.2),  # Billion USD
            'other': economic_data.get('other_exports', 4.0)  # Billion USD
        }
        
        # Main export markets and their shares
        self.export_markets = {
            'eu': economic_data.get('eu_export_share', 0.51),  # 51%
            'us': economic_data.get('us_export_share', 0.18),  # 18%
            'canada': economic_data.get('canada_export_share', 0.04),  # 4%
            'japan': economic_data.get('japan_export_share', 0.03),  # 3%
            'china': economic_data.get('china_export_share', 0.02),  # 2%
            'india': economic_data.get('india_export_share', 0.03),  # 3%
            'other_asia': economic_data.get('other_asia_export_share', 0.12),  # 12%
            'other': economic_data.get('other_export_share', 0.07)  # 7%
        }
        
        # Initialize import variables
        self.total_imports = economic_data.get('initial_imports', 55.0)  # Billion USD
        self.imports_by_category = {
            'raw_materials': economic_data.get('raw_materials_imports', 20.0),  # Billion USD
            'machinery': economic_data.get('machinery_imports', 8.0),  # Billion USD
            'energy': economic_data.get('energy_imports', 6.0),  # Billion USD
            'food': economic_data.get('food_imports', 7.0),  # Billion USD
            'chemicals': economic_data.get('chemicals_imports', 5.0),  # Billion USD
            'consumer_goods': economic_data.get('consumer_goods_imports', 6.0),  # Billion USD
            'other': economic_data.get('other_imports', 3.0)  # Billion USD
        }
        
        # Main import sources and their shares
        self.import_sources = {
            'china': economic_data.get('china_import_share', 0.25),  # 25%
            'india': economic_data.get('india_import_share', 0.15),  # 15%
            'singapore': economic_data.get('singapore_import_share', 0.08),  # 8%
            'japan': economic_data.get('japan_import_share', 0.05),  # 5%
            'us': economic_data.get('us_import_share', 0.04),  # 4%
            'eu': economic_data.get('eu_import_share', 0.10),  # 10%
            'middle_east': economic_data.get('middle_east_import_share', 0.12),  # 12%
            'other': economic_data.get('other_import_share', 0.21)  # 21%
        }
        
        # Tariff rates (average import tariff rates by category)
        self.import_tariffs = {
            'raw_materials': economic_data.get('raw_materials_tariff', 0.05),  # 5%
            'machinery': economic_data.get('machinery_tariff', 0.10),  # 10%
            'energy': economic_data.get('energy_tariff', 0.02),  # 2%
            'food': economic_data.get('food_tariff', 0.15),  # 15%
            'chemicals': economic_data.get('chemicals_tariff', 0.08),  # 8%
            'consumer_goods': economic_data.get('consumer_goods_tariff', 0.25),  # 25%
            'other': economic_data.get('other_tariff', 0.15)  # 15%
        }
        
        # Export incentives (as equivalent subsidy rates)
        self.export_incentives = {
            'garment': economic_data.get('garment_incentives', 0.04),  # 4%
            'agricultural': economic_data.get('agricultural_incentives', 0.03),  # 3%
            'leather': economic_data.get('leather_incentives', 0.05),  # 5%
            'jute': economic_data.get('jute_incentives', 0.06),  # 6%
            'technology': economic_data.get('technology_incentives', 0.08),  # 8%
            'other': economic_data.get('other_incentives', 0.02)  # 2%
        }
        
        # Shipping and logistics parameters
        self.port_efficiency = economic_data.get('port_efficiency', 0.6)  # 60% efficiency
        self.shipping_costs = economic_data.get('shipping_costs', 0.08)  # 8% of trade value
        self.shipping_frequency = economic_data.get('shipping_frequency', 0.7)  # 70% of optimal
        self.customs_clearance_time = economic_data.get('customs_clearance_time', 8.5)  # days
        
        # Trade agreements (binary indicators)
        self.trade_agreements = {
            'eu_gsp': economic_data.get('eu_gsp', 1),  # EU GSP (Everything But Arms)
            'safta': economic_data.get('safta', 1),  # South Asian Free Trade Area
            'china_fta': economic_data.get('china_fta', 0),  # China FTA (not in effect)
            'india_fta': economic_data.get('india_fta', 0),  # Comprehensive India FTA (not in effect)
            'bimstec': economic_data.get('bimstec', 1),  # Bay of Bengal Initiative
            'apta': economic_data.get('apta', 1)  # Asia-Pacific Trade Agreement
        }
        
        print("Trade system initialized")
    
    def update_exports(self, exchange_rate, sectors=None, infrastructure=None, environmental=None, governance=None):
        """
        Update export values based on economic conditions and external factors.
        
        Args:
            exchange_rate (float): Current exchange rate (BDT per USD)
            sectors (list): List of economic sector objects
            infrastructure (dict): Infrastructure system impacts
            environmental (dict): Environmental system impacts
            governance (dict): Governance system impacts
        
        Returns:
            dict: Updated export values and statistics
        """
        # Base export growth rates by sector (trend growth)
        base_growth_rates = {
            'garment': 0.07,  # 7% annual growth
            'agricultural': 0.04,
            'leather': 0.06,
            'jute': 0.02,
            'technology': 0.15,
            'other': 0.05
        }
        
        # Exchange rate effect on export competitiveness
        # Depreciation (higher BDT per USD) improves export competitiveness
        exchange_rate_effect = 0.5 * (exchange_rate / self.exchange_rate - 1)
        
        # Sector-specific effects if sectors are provided
        sector_effects = {}
        if sectors:
            # Get data from garment sector if available
            garment_sector = next((s for s in sectors if getattr(s, 'name', '') == 'garment'), None)
            if garment_sector and hasattr(garment_sector, 'productivity'):
                sector_effects['garment'] = 0.3 * (garment_sector.productivity - 1)
            
            # Similarly for other sectors...
            agricultural_sector = next((s for s in sectors if getattr(s, 'name', '') == 'agricultural'), None)
            if agricultural_sector and hasattr(agricultural_sector, 'productivity'):
                sector_effects['agricultural'] = 0.4 * (agricultural_sector.productivity - 1)
        
        # Infrastructure effects if provided
        infrastructure_effect = 0.0
        if infrastructure:
            # Check if infrastructure is a dictionary or an object
            if isinstance(infrastructure, dict):
                port_effect = 0.15 * (infrastructure.get('transport_efficiency', 0.5) - 0.5)
            else:
                # Based on the infrastructure system memory, use the transportation component
                # Try various possible attribute names
                if hasattr(infrastructure, 'transport_efficiency'):
                    port_effect = 0.15 * (infrastructure.transport_efficiency - 0.5)
                elif hasattr(infrastructure, 'transportation_quality_index'):
                    port_effect = 0.15 * (infrastructure.transportation_quality_index - 0.5)
                else:
                    # Default if attributes not found
                    port_effect = 0.0
            
            infrastructure_effect = port_effect
        
        # Environmental effects if provided
        environmental_effect = 0.0
        if environmental:
            flood_effect = -0.1 * environmental.get('flood_impacts', 0)
            cyclone_effect = -0.2 * environmental.get('cyclone_damage', 0)
            environmental_effect = flood_effect + cyclone_effect
        
        # Governance and policy effects if provided
        policy_effect = 0.0
        if governance:
            # Check if governance is a dictionary or an object
            if isinstance(governance, dict):
                corruption_effect = -0.1 * governance.get('corruption_index', 0.5)
                trade_policy_effect = 0.2 * governance.get('trade_openness', 0.5)
            else:
                # Handle governance system object
                corruption_effect = -0.1 * getattr(governance, 'corruption_index', 0.5)
                trade_policy_effect = 0.2 * getattr(governance, 'trade_openness', 0.5)
            
            policy_effect = corruption_effect + trade_policy_effect
        
        # Global market conditions (random variation representing external demand)
        global_market_effect = np.random.normal(0, 0.02)
        
        # Calculate new export values for each sector
        updated_exports = {}
        for sector, value in self.exports_by_sector.items():
            # Combine all effects
            sector_effect = sector_effects.get(sector, 0)
            combined_effect = (
                base_growth_rates.get(sector, 0.05) + 
                exchange_rate_effect + 
                sector_effect + 
                infrastructure_effect + 
                environmental_effect + 
                policy_effect + 
                global_market_effect
            )
            
            # Update sector exports
            updated_exports[sector] = value * (1 + combined_effect)
        
        # Update total exports
        self.total_exports = sum(updated_exports.values())
        self.exports_by_sector = updated_exports
        
        # Update export market shares (simplified - would be more dynamic in full model)
        # In a full model, this would account for trade agreements, market demand, etc.
        
        # Compile export statistics
        export_stats = {
            'total_exports': self.total_exports,
            'exports_by_sector': self.exports_by_sector,
            'export_markets': self.export_markets,
            'exchange_rate_effect': exchange_rate_effect,
            'infrastructure_effect': infrastructure_effect,
            'environmental_effect': environmental_effect,
            'policy_effect': policy_effect,
            'global_market_effect': global_market_effect
        }
        
        return export_stats
    
    def update_imports(self, exchange_rate, gdp=None, infrastructure=None, governance=None):
        """
        Update import values based on economic conditions and external factors.
        
        Args:
            exchange_rate (float): Current exchange rate (BDT per USD)
            gdp (float): Current GDP level
            infrastructure (dict): Infrastructure system impacts
            governance (dict): Governance system impacts
            
        Returns:
            dict: Updated import values and statistics
        """
        # Exchange rate effect on imports
        # Appreciation (lower BDT per USD) increases imports, depreciation decreases
        exchange_rate_effect = -0.3 * (exchange_rate / self.exchange_rate - 1)
        
        # GDP effect on imports (income elasticity)
        gdp_effect = 0.0
        if gdp:
            gdp_effect = 0.8 * (gdp / self.economic_data.get('initial_gdp', 300.0) - 1)
        
        # Infrastructure effects if provided
        infrastructure_effect = 0.0
        if infrastructure:
            # Check if infrastructure is a dictionary or an object
            if isinstance(infrastructure, dict):
                port_effect = 0.15 * (infrastructure.get('transport_efficiency', 0.5) - 0.5)
            else:
                # Based on the infrastructure system memory, use the transportation component
                # Try various possible attribute names
                if hasattr(infrastructure, 'transport_efficiency'):
                    port_effect = 0.15 * (infrastructure.transport_efficiency - 0.5)
                elif hasattr(infrastructure, 'transportation_quality_index'):
                    port_effect = 0.15 * (infrastructure.transportation_quality_index - 0.5)
                else:
                    # Default if attributes not found
                    port_effect = 0.0
            
            infrastructure_effect = port_effect
        
        # Tariff effect (from governance policies)
        tariff_effect = 0.0
        if governance:
            if hasattr(governance, 'get_tariff_adjustment'):
                try:
                    tariff_effect = -0.5 * governance.get_tariff_adjustment()
                except (TypeError, AttributeError):
                    # If method exists but fails, use a default value
                    tariff_effect = 0.0
            elif isinstance(governance, dict):
                # If governance is a dictionary, try to get tariff_adjustment directly
                tariff_effect = -0.5 * governance.get('tariff_adjustment', 0.0)
            else:
                # Try to get the attribute directly
                tariff_effect = -0.5 * getattr(governance, 'tariff_adjustment', 0.0)
        
        # Global price effects (e.g., energy prices)
        global_price_effect = np.random.normal(0, 0.03)
        
        # Calculate new import values for each category
        updated_imports = {}
        for category, value in self.imports_by_category.items():
            # Different elasticities for different import categories
            if category == 'energy':
                category_elasticity = 0.6  # Less elastic
            elif category == 'raw_materials':
                category_elasticity = 0.9  # More elastic to exchange rate
            elif category == 'machinery':
                category_elasticity = 0.7
            else:
                category_elasticity = 0.8
                
            # Combine all effects with category-specific elasticity
            combined_effect = (
                exchange_rate_effect * category_elasticity + 
                gdp_effect + 
                infrastructure_effect + 
                tariff_effect + 
                global_price_effect
            )
            
            # Update category imports
            updated_imports[category] = value * (1 + combined_effect)
        
        # Update total imports
        self.total_imports = sum(updated_imports.values())
        self.imports_by_category = updated_imports
        
        # Compile import statistics
        import_stats = {
            'total_imports': self.total_imports,
            'imports_by_category': self.imports_by_category,
            'import_sources': self.import_sources,
            'exchange_rate_effect': exchange_rate_effect,
            'gdp_effect': gdp_effect,
            'infrastructure_effect': infrastructure_effect,
            'tariff_effect': tariff_effect,
            'global_price_effect': global_price_effect
        }
        
        return import_stats
    
    def update_trade_logistics(self, infrastructure=None, governance=None):
        """
        Update trade logistics parameters based on infrastructure and governance.
        
        Args:
            infrastructure (dict): Infrastructure system impacts
            governance (dict): Governance system impacts
            
        Returns:
            dict: Updated logistics parameters
        """
        # Base annual improvements (trend)
        base_port_improvement = 0.02  # 2% annual improvement
        base_customs_improvement = 0.03  # 3% annual improvement
        
        # Infrastructure effects
        infrastructure_effect = 0.0
        if infrastructure:
            # Check if infrastructure is a dictionary or an object
            if isinstance(infrastructure, dict):
                port_effect = 0.15 * (infrastructure.get('transport_efficiency', 0.5) - 0.5)
            else:
                # Based on the infrastructure system memory, use the transportation component
                # Try various possible attribute names
                if hasattr(infrastructure, 'transport_efficiency'):
                    port_effect = 0.15 * (infrastructure.transport_efficiency - 0.5)
                elif hasattr(infrastructure, 'transportation_quality_index'):
                    port_effect = 0.15 * (infrastructure.transportation_quality_index - 0.5)
                else:
                    # Default if attributes not found
                    port_effect = 0.0
            
            infrastructure_effect = port_effect
        
        # Governance effects
        governance_effect = 0.0
        if governance:
            # Check if governance is a dictionary or an object
            if isinstance(governance, dict):
                corruption_effect = -0.2 * governance.get('corruption_index', 0.5)
                efficiency_effect = 0.25 * governance.get('institutional_efficiency', 0.5)
            else:
                # Handle governance system object
                corruption_effect = -0.2 * getattr(governance, 'corruption_index', 0.5)
                efficiency_effect = 0.25 * getattr(governance, 'institutional_efficiency', 0.5)
            
            governance_effect = corruption_effect + efficiency_effect
        
        # Update port efficiency
        self.port_efficiency += base_port_improvement + 0.5 * infrastructure_effect + 0.5 * governance_effect
        self.port_efficiency = max(min(self.port_efficiency, 0.95), 0.4)  # Bound values
        
        # Update shipping frequency
        self.shipping_frequency += 0.5 * base_port_improvement + 0.4 * infrastructure_effect + 0.2 * governance_effect
        self.shipping_frequency = max(min(self.shipping_frequency, 0.95), 0.4)
        
        # Update customs clearance time (lower is better)
        clearance_improvement = base_customs_improvement + 0.3 * infrastructure_effect + 0.7 * governance_effect
        self.customs_clearance_time *= (1 - clearance_improvement)
        self.customs_clearance_time = max(min(self.customs_clearance_time, 15.0), 2.0)
        
        # Calculate logistics efficiency index (0-1 scale)
        logistics_efficiency = (
            0.4 * self.port_efficiency + 
            0.3 * self.shipping_frequency + 
            0.3 * (1 - self.customs_clearance_time / 15.0)
        )
        
        # Update shipping costs (affected by global energy prices and efficiency)
        global_shipping_trend = np.random.normal(0.01, 0.02)  # Slight upward trend with volatility
        efficiency_effect = -0.3 * (logistics_efficiency - 0.6)  # Higher efficiency reduces costs
        self.shipping_costs *= (1 + global_shipping_trend + efficiency_effect)
        self.shipping_costs = max(min(self.shipping_costs, 0.15), 0.04)
        
        # Compile logistics statistics
        logistics_stats = {
            'port_efficiency': self.port_efficiency,
            'shipping_frequency': self.shipping_frequency,
            'customs_clearance_time': self.customs_clearance_time,
            'shipping_costs': self.shipping_costs,
            'logistics_efficiency': logistics_efficiency,
            'infrastructure_effect': infrastructure_effect,
            'governance_effect': governance_effect
        }
        
        return logistics_stats
    
    def update_trade_agreements(self, governance=None):
        """
        Update trade agreement status based on governance decisions.
        
        Args:
            governance (dict): Governance system impacts
            
        Returns:
            dict: Updated trade agreement statuses
        """
        # In a full model, this would implement more complex logic for trade agreement
        # negotiations, implementation, and effects
        
        # For now, we'll just simulate the possibility of new agreements
        # based on governance trade policy orientation
        
        if governance and hasattr(governance, 'get_trade_policy_orientation'):
            orientation = governance.get_trade_policy_orientation()
            
            # Probability of new FTAs increases with more open trade orientation
            china_fta_prob = 0.05 * orientation if self.trade_agreements['china_fta'] == 0 else 0
            india_fta_prob = 0.04 * orientation if self.trade_agreements['india_fta'] == 0 else 0
            
            # Simulate potential new agreements
            if np.random.random() < china_fta_prob:
                self.trade_agreements['china_fta'] = 1
                print("New trade agreement implemented: China FTA")
                
            if np.random.random() < india_fta_prob:
                self.trade_agreements['india_fta'] = 1
                print("New trade agreement implemented: Comprehensive India FTA")
        
        return self.trade_agreements
    
    def calculate_trade_balance(self):
        """
        Calculate trade balance and related statistics.
        
        Returns:
            dict: Trade balance statistics
        """
        trade_balance = self.total_exports - self.total_imports
        trade_balance_pct_gdp = trade_balance / self.economic_data.get('initial_gdp', 300.0)
        
        trade_stats = {
            'exports': self.total_exports,
            'imports': self.total_imports,
            'trade_balance': trade_balance,
            'trade_balance_pct_gdp': trade_balance_pct_gdp,
            'export_import_ratio': self.total_exports / max(self.total_imports, 0.1)
        }
        
        return trade_stats
    
    def step(self, exchange_rate, sectors=None, infrastructure=None, governance=None):
        """
        Advance the trade system by one time step.
        
        Args:
            exchange_rate (float): Current exchange rate (BDT per USD)
            sectors (list): Economic sector objects
            infrastructure (dict): Infrastructure system impacts
            governance (dict): Governance system for policy impacts
            
        Returns:
            dict: Trade system results after the step
        """
        # Store previous exchange rate
        self.exchange_rate = exchange_rate
        
        # Get GDP if available
        gdp = None
        if hasattr(sectors[0], 'gdp') if sectors else False:
            gdp = sectors[0].gdp
        
        # Update exports
        export_stats = self.update_exports(
            exchange_rate=exchange_rate,
            sectors=sectors,
            infrastructure=infrastructure,
            governance=governance
        )
        
        # Update imports
        import_stats = self.update_imports(
            exchange_rate=exchange_rate,
            gdp=gdp,
            infrastructure=infrastructure,
            governance=governance
        )
        
        # Update logistics
        logistics_stats = self.update_trade_logistics(
            infrastructure=infrastructure,
            governance=governance
        )
        
        # Update trade agreements
        trade_agreements = self.update_trade_agreements(governance)
        
        # Calculate trade balance
        trade_balance_stats = self.calculate_trade_balance()
        
        # Compile results
        results = {
            'exports': export_stats,
            'imports': import_stats,
            'logistics': logistics_stats,
            'trade_agreements': trade_agreements,
            'trade_balance': trade_balance_stats
        }
        
        return results
    
    def get_export_competitiveness_index(self):
        """
        Calculate export competitiveness index based on various factors.
        
        Returns:
            float: Export competitiveness index (0-1 scale)
        """
        # Weighted combination of factors affecting export competitiveness
        logistics_factor = (self.port_efficiency + self.shipping_frequency) / 2
        cost_factor = 1 - self.shipping_costs / 0.15  # Normalize to 0-1 scale
        
        # Incentive factor based on export incentives
        incentive_factor = sum(self.export_incentives.values()) / len(self.export_incentives)
        
        # Trade agreement factor
        agreement_factor = sum(self.trade_agreements.values()) / len(self.trade_agreements)
        
        # Weighted combination
        competitiveness_index = (
            0.3 * logistics_factor +
            0.3 * cost_factor +
            0.2 * incentive_factor +
            0.2 * agreement_factor
        )
        
        return competitiveness_index
