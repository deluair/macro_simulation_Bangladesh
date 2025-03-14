from typing import Dict, Any, List
import numpy as np
from .base_model import BaseModel

class EconomicModel(BaseModel):
    """Model for Bangladesh's economic system."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sectors = {
            'garment': {},
            'agriculture': {},
            'remittances': {},
            'tech': {},
            'informal': {}
        }
        self.exchange_rate = 0.0
        self.port_efficiency = 0.0
        self.credit_markets = {
            'rural': {},
            'urban': {}
        }
        
    def get_required_parameters(self) -> List[str]:
        """
        Get list of required configuration parameters for the economic model.
        
        Returns:
            List[str]: List of required parameter names
        """
        # Return list of any required parameters, or empty list if none are strictly required
        return []
        
    def initialize(self) -> None:
        """Initialize economic model state."""
        # Initialize sector-specific parameters from config - using actual values from config
        self.sectors = {
            'garment': {
                'gdp_share': self.config.get('garment_gdp_share', 0.2),
                'growth_rate': self.config.get('garment_growth_rate', 0.05),
                'employment': self.config.get('garment_employment', 4000000),
                'productivity': self.config.get('garment_productivity', 1.0)
            },
            'agriculture': {
                'gdp_share': self.config.get('agriculture_gdp_share', 0.15),
                'growth_rate': self.config.get('agriculture_growth_rate', 0.03),
                'employment': self.config.get('agriculture_employment', 20000000),
                'productivity': self.config.get('agriculture_productivity', 1.0)
            },
            'remittances': {
                'gdp_share': self.config.get('remittances_gdp_share', 0.1),
                'growth_rate': self.config.get('remittances_growth_rate', 0.04),
                'employment': self.config.get('remittances_employment', 0),
                'productivity': self.config.get('remittances_productivity', 1.0)
            },
            'tech': {
                'gdp_share': self.config.get('tech_gdp_share', 0.05),
                'growth_rate': self.config.get('tech_growth_rate', 0.15),
                'employment': self.config.get('tech_employment', 500000),
                'productivity': self.config.get('tech_productivity', 1.5)
            },
            'informal': {
                'gdp_share': self.config.get('informal_gdp_share', 0.5),
                'growth_rate': self.config.get('informal_growth_rate', 0.02),
                'employment': self.config.get('informal_employment', 30000000),
                'productivity': self.config.get('informal_productivity', 0.8)
            }
        }
        
        # Initialize exchange rate and port efficiency
        self.exchange_rate = self.config.get('initial_exchange_rate', 85.0)
        self.port_efficiency = self.config.get('initial_port_efficiency', 0.7)
        
        # Initialize credit markets
        for market in self.credit_markets:
            self.credit_markets[market] = {
                'interest_rate': self.config.get(f'{market}_interest_rate', 0.1),
                'access_rate': self.config.get(f'{market}_access_rate', 0.5),
                'default_rate': self.config.get(f'{market}_default_rate', 0.05)
            }

        # Calculate total GDP in USD based on shares and baseline
        baseline_gdp = self.config.get('baseline_gdp', 1000000000)  # 1 billion USD default
        
        # Initialize state with actual calculated values instead of zeros
        self.state = {
            'total_gdp': self._calculate_total_gdp() * baseline_gdp,  # Actual GDP value
            'gdp_growth': 0.0,
            'unemployment_rate': self._calculate_unemployment(),
            'inflation_rate': self.config.get('initial_inflation_rate', 0.05),
            'trade_balance': self._calculate_trade_balance() * baseline_gdp
        }
    
    def step(self) -> None:
        """Execute one economic simulation step."""
        # Update exchange rate based on market conditions
        self._update_exchange_rate()
        
        # Update sector performances
        for sector in self.sectors:
            self._update_sector(sector)
        
        # Update credit market conditions
        self._update_credit_markets()
        
        # Calculate aggregate economic indicators
        self._calculate_aggregates()
    
    def update(self) -> None:
        """Update model state based on step results."""
        # The state is already updated in _calculate_aggregates, so we don't need
        # to recalculate everything again. Just verify that critical state variables exist.
        if 'total_gdp' not in self.state:
            self.state['total_gdp'] = self._calculate_total_gdp() * self.config.get('baseline_gdp', 1000000000)
        
        if 'gdp_growth' not in self.state:
            self.state['gdp_growth'] = 0.0
            
        if 'unemployment_rate' not in self.state:
            self.state['unemployment_rate'] = self._calculate_unemployment()
            
        if 'inflation_rate' not in self.state:
            self.state['inflation_rate'] = self._calculate_inflation()
            
        if 'trade_balance' not in self.state:
            self.state['trade_balance'] = self._calculate_trade_balance() * self.config.get('baseline_gdp', 1000000000)
    
    def _update_exchange_rate(self) -> None:
        """Update exchange rate based on market conditions."""
        # Implement exchange rate dynamics
        volatility = self.config.get('exchange_rate_volatility', 0.02)
        self.exchange_rate *= (1 + np.random.normal(0, volatility))
    
    def _update_sector(self, sector: str) -> None:
        """Update individual sector performance."""
        # Get baseline growth rate for this sector
        base_growth_rate = self.sectors[sector]['growth_rate']
        
        # Add some volatility to growth
        volatility = self.config.get('economic_volatility', 0.01)
        random_factor = np.random.normal(0, volatility)
        
        # Calculate actual growth for this step (base + random variation)
        actual_growth = base_growth_rate + random_factor
        
        # Adjust growth based on shared state if available
        if 'governance' in self.shared_state:
            # Better governance effectiveness improves growth
            governance_effect = self.shared_state['governance'].get('governance_effectiveness_index', 0.5)
            actual_growth *= (0.8 + 0.4 * governance_effect)  # 0.8-1.2x multiplier based on governance
        
        if 'infrastructure' in self.shared_state:
            # Better infrastructure improves growth
            infra_effect = self.shared_state['infrastructure'].get('infrastructure_quality_index', 0.5)
            actual_growth *= (0.8 + 0.4 * infra_effect)  # 0.8-1.2x multiplier based on infrastructure
        
        # Apply sector-specific effects
        if sector == 'garment':
            self._apply_garment_sector_dynamics(actual_growth)
        elif sector == 'agriculture':
            self._apply_agriculture_sector_dynamics(actual_growth)
        elif sector == 'remittances':
            # Remittances affected by global economic conditions (simulated)
            global_economic_factor = np.random.normal(1.0, 0.05)
            actual_growth *= global_economic_factor
        elif sector == 'tech':
            # Tech sector grows faster with better literacy/education
            if 'demographic' in self.shared_state:
                literacy_rate = self.shared_state['demographic'].get('literacy_rate', 0.75)
                actual_growth *= (0.7 + 0.6 * literacy_rate)  # Higher literacy boosts tech growth
        
        # Update sector GDP share
        self.sectors[sector]['gdp_share'] *= (1 + actual_growth)
        
        # Update sector employment (grows with sector GDP, but at a slower rate)
        employment_growth = actual_growth * 0.7  # Employment grows slower than GDP
        self.sectors[sector]['employment'] *= (1 + employment_growth)
        
        # Update productivity (slowly improves over time)
        productivity_improvement = self.config.get('productivity_improvement_rate', 0.01)
        self.sectors[sector]['productivity'] *= (1 + productivity_improvement + np.random.normal(0, 0.005))
    
    def _update_credit_markets(self) -> None:
        """Update credit market conditions."""
        for market in self.credit_markets:
            # Implement credit market dynamics
            self.credit_markets[market]['interest_rate'] *= (1 + np.random.normal(0, 0.005))
            self.credit_markets[market]['access_rate'] *= (1 + np.random.normal(0, 0.01))
    
    def _calculate_aggregates(self) -> None:
        """Calculate aggregate economic indicators."""
        # Calculate total GDP (baseline * sum of sector shares)
        baseline_gdp = self.config.get('baseline_gdp', 1000000000)  # 1 billion USD default
        
        # Calculate current GDP
        current_gdp = self._calculate_total_gdp() * baseline_gdp
        
        # Store the previous GDP to calculate growth rate
        previous_gdp = self.state.get('total_gdp', current_gdp)
        
        # Calculate GDP growth rate
        if previous_gdp > 0:
            gdp_growth = (current_gdp / previous_gdp) - 1
        else:
            gdp_growth = 0.0
        
        # Store calculated values for state update
        self.state['total_gdp'] = current_gdp
        self.state['gdp_growth'] = gdp_growth
        self.state['unemployment_rate'] = self._calculate_unemployment()
        self.state['inflation_rate'] = self._calculate_inflation()
        self.state['trade_balance'] = self._calculate_trade_balance() * baseline_gdp
    
    def _calculate_total_gdp(self) -> float:
        """Calculate total GDP."""
        return sum(sector['gdp_share'] for sector in self.sectors.values())
    
    def _calculate_unemployment(self) -> float:
        """Calculate unemployment rate."""
        total_employment = sum(sector['employment'] for sector in self.sectors.values())
        total_labor_force = self.config.get('total_labor_force', 100000000)
        return 1 - (total_employment / total_labor_force)
    
    def _calculate_inflation(self) -> float:
        """Calculate inflation rate."""
        return np.random.normal(0.05, 0.02)  # Placeholder implementation
    
    def _calculate_trade_balance(self) -> float:
        """Calculate trade balance."""
        exports = self.sectors['garment']['gdp_share'] * 0.8  # Assuming 80% of garment sector is exports
        imports = sum(sector['gdp_share'] * 0.3 for sector in self.sectors.values())  # Assuming 30% import content
        return exports - imports
    
    def _apply_garment_sector_dynamics(self, growth_rate) -> None:
        """Apply specific dynamics to garment sector."""
        # Garment sector is affected by:
        # 1. Global market conditions (random)
        # 2. Port efficiency
        # 3. Infrastructure
        
        # Global market factor (random variation)
        global_market = np.random.normal(1.0, 0.05)
        
        # Port efficiency impact
        port_impact = 0.8 + (0.4 * self.port_efficiency)  # 0.8-1.2x multiplier
        
        # Apply adjustments to growth
        adjusted_growth = growth_rate * global_market * port_impact
        
        # Update sector with additional growth factor
        self.sectors['garment']['growth_rate'] = self.config.get('garment_growth_rate', 0.05) * global_market * port_impact
    
    def _apply_agriculture_sector_dynamics(self, growth_rate) -> None:
        """Apply specific dynamics to agriculture sector."""
        # Agriculture is affected by:
        # 1. Weather/climate conditions
        # 2. Water availability
        # 3. Infrastructure (irrigation, storage)
        
        # Default adjustments
        weather_factor = 1.0
        water_factor = 1.0
        
        # Get environmental factors from shared state if available
        if 'environmental' in self.shared_state:
            # Check for extreme weather and water stress
            flood_risk = self.shared_state['environmental'].get('flood_risk', 0.0)
            water_stress = self.shared_state['environmental'].get('water_stress_index', 0.0)
            crop_yield = self.shared_state['environmental'].get('crop_yield_index', 1.0)
            
            # Adjust factors based on environmental conditions
            weather_factor = 1.0 - (flood_risk * 0.5)  # Floods reduce agricultural productivity
            water_factor = 1.0 - (water_stress * 0.3)  # Water stress reduces growth
            
            # Use crop yield as a direct factor
            if crop_yield > 0:
                weather_factor *= crop_yield
        
        # Apply adjustments to growth
        adjusted_growth = growth_rate * weather_factor * water_factor
        
        # Update sector with additional growth factor
        self.sectors['agriculture']['growth_rate'] = self.config.get('agriculture_growth_rate', 0.03) * weather_factor * water_factor 