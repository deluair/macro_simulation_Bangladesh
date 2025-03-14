from typing import Dict, Any, List
import numpy as np
from .base_model import BaseModel

class InfrastructureModel(BaseModel):
    """Model for Bangladesh's infrastructure systems."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.transportation = {}
        self.energy = {}
        self.telecommunications = {}
        self.housing = {}
        self.supply_chains = {}
        
    def get_required_parameters(self) -> List[str]:
        """
        Get list of required configuration parameters for the infrastructure model.
        
        Returns:
            List[str]: List of required parameter names
        """
        # Return list of any required parameters, or empty list if none are strictly required
        return []
        
    def initialize(self) -> None:
        """Initialize infrastructure model state."""
        # Initialize transportation systems
        self.transportation = {
            'road_network': {
                'total_length': self.config.get('initial_road_length', 350000),  # km
                'paved_ratio': self.config.get('initial_paved_ratio', 0.6),
                'congestion_level': self.config.get('initial_congestion', 0.5)
            },
            'waterways': {
                'total_length': self.config.get('initial_waterway_length', 6000),  # km
                'navigable_ratio': self.config.get('initial_navigable_ratio', 0.7),
                'efficiency': self.config.get('initial_waterway_efficiency', 0.6)
            },
            'railways': {
                'total_length': self.config.get('initial_rail_length', 3000),  # km
                'operational_ratio': self.config.get('initial_rail_operational', 0.8),
                'capacity_utilization': self.config.get('initial_rail_capacity', 0.7)
            }
        }
        
        # Initialize energy systems
        self.energy = {
            'generation': {
                'total_capacity': self.config.get('initial_power_capacity', 25000),  # MW
                'renewable_share': self.config.get('initial_renewable_share', 0.1),
                'load_shedding': self.config.get('initial_load_shedding', 0.1)
            },
            'distribution': {
                'grid_efficiency': self.config.get('initial_grid_efficiency', 0.85),
                'transmission_loss': self.config.get('initial_transmission_loss', 0.15),
                'access_rate': self.config.get('initial_electricity_access', 0.95)
            }
        }
        
        # Initialize telecommunications
        self.telecommunications = {
            'mobile': {
                'penetration_rate': self.config.get('initial_mobile_penetration', 0.8),
                'network_coverage': self.config.get('initial_mobile_coverage', 0.9),
                'data_usage': self.config.get('initial_data_usage', 0.5)  # GB per capita
            },
            'internet': {
                'penetration_rate': self.config.get('initial_internet_penetration', 0.4),
                'broadband_access': self.config.get('initial_broadband_access', 0.3),
                'speed': self.config.get('initial_internet_speed', 5.0)  # Mbps
            }
        }
        
        # Initialize housing
        self.housing = {
            'urban': {
                'total_units': self.config.get('initial_urban_housing', 10000000),
                'quality_index': self.config.get('initial_urban_housing_quality', 0.6),
                'slum_ratio': self.config.get('initial_slum_ratio', 0.3)
            },
            'rural': {
                'total_units': self.config.get('initial_rural_housing', 15000000),
                'quality_index': self.config.get('initial_rural_housing_quality', 0.5),
                'vulnerability_index': self.config.get('initial_housing_vulnerability', 0.4)
            }
        }
        
        # Initialize supply chains
        self.supply_chains = {
            'garment': {
                'resilience_index': self.config.get('initial_garment_resilience', 0.7),
                'logistics_efficiency': self.config.get('initial_garment_logistics', 0.6),
                'port_efficiency': self.config.get('initial_port_efficiency', 0.7)
            },
            'agriculture': {
                'resilience_index': self.config.get('initial_agri_resilience', 0.6),
                'storage_capacity': self.config.get('initial_storage_capacity', 0.5),
                'distribution_efficiency': self.config.get('initial_agri_distribution', 0.5)
            }
        }
        
        self.state = {
            'infrastructure_quality_index': 0.0,
            'connectivity_index': 0.0,
            'resilience_index': 0.0,
            'efficiency_index': 0.0
        }
    
    def step(self) -> None:
        """Execute one infrastructure simulation step."""
        # Update transportation systems
        self._update_transportation()
        
        # Update energy systems
        self._update_energy()
        
        # Update telecommunications
        self._update_telecommunications()
        
        # Update housing
        self._update_housing()
        
        # Update supply chains
        self._update_supply_chains()
        
        # Calculate infrastructure indicators
        self._calculate_infrastructure_indicators()
    
    def update(self) -> None:
        """Update model state based on step results."""
        self.state.update({
            'infrastructure_quality_index': self._calculate_quality_index(),
            'connectivity_index': self._calculate_connectivity_index(),
            'resilience_index': self._calculate_resilience_index(),
            'efficiency_index': self._calculate_efficiency_index()
        })
    
    def _update_transportation(self) -> None:
        """Update transportation systems."""
        # GDP impact factor - infrastructure development correlates with economic growth
        gdp_impact = np.clip(self.shared_state.get('economic', {}).get('gdp_growth', 0.03), 0.01, 0.08)
        
        # Government effectiveness impact
        gov_effectiveness = np.clip(self.shared_state.get('governance', {}).get('governance_effectiveness_index', 0.5), 0.2, 0.9)
        
        # Climate impact factor - extreme weather events slow infrastructure development
        climate_impact = 1 - (self.shared_state.get('environmental', {}).get('extreme_weather_frequency', 0.1) * 0.2)
        
        # Budget constraint - random variation to simulate funding fluctuations
        budget_factor = np.random.normal(1.0, 0.05) * gov_effectiveness
        
        # Update road network with realistic constraints and variability
        self.transportation['road_network']['paved_ratio'] = np.clip(
            self.transportation['road_network']['paved_ratio'] * (
                1 + self.config.get('road_improvement_rate', 0.005) * budget_factor * climate_impact
            ),
            0.0, 0.95  # Maximum paved ratio constraint (95% is realistic ceiling)
        )
        
        # Congestion increases with urbanization and decreases with infrastructure investment
        urbanization_impact = self.shared_state.get('demographic', {}).get('urbanization_rate', 0.02)
        self.transportation['road_network']['congestion_level'] = np.clip(
            self.transportation['road_network']['congestion_level'] * (
                1 + (urbanization_impact * 0.1) - (self.config.get('road_improvement_rate', 0.005) * budget_factor * 0.5)
            ),
            0.2, 0.9  # Congestion level constraints
        )
        
        # Update waterways with seasonal variation (monsoon impacts)
        seasonal_factor = np.random.normal(1.0, 0.1)  # Simulate seasonal variability
        self.transportation['waterways']['navigable_ratio'] = np.clip(
            self.transportation['waterways']['navigable_ratio'] * (
                1 + self.config.get('waterway_improvement_rate', 0.003) * budget_factor * climate_impact * seasonal_factor
            ),
            0.5, 0.9  # Realistic constraints for navigable waterways
        )
        
        # Update railways with maintenance factor
        maintenance_factor = np.random.normal(0.98, 0.02)  # Railway infrastructure degrades without maintenance
        investment_factor = gdp_impact * gov_effectiveness * budget_factor
        self.transportation['railways']['operational_ratio'] = np.clip(
            self.transportation['railways']['operational_ratio'] * (
                maintenance_factor + self.config.get('rail_improvement_rate', 0.004) * investment_factor
            ),
            0.6, 0.95  # Realistic constraints for railway operation
        )
    
    def _update_telecommunications(self) -> None:
        """Update telecommunications systems."""
        # Initialize telecommunications if not already done
        if not hasattr(self, 'telecommunications') or not self.telecommunications:
            self.telecommunications = {
                'mobile_coverage': self.config.get('initial_mobile_coverage', 0.9),
                'internet_coverage': self.config.get('initial_internet_coverage', 0.6),
                'broadband_quality': self.config.get('initial_broadband_quality', 0.5),
                'digital_inclusion': self.config.get('initial_digital_inclusion', 0.4),
                'network_resilience': self.config.get('initial_network_resilience', 0.6)
            }
        
        # Update mobile coverage (approaching saturation)
        max_mobile_coverage = self.config.get('max_mobile_coverage', 0.99)
        mobile_growth_rate = self.config.get('mobile_growth_rate', 0.02)
        remaining_mobile_gap = max(0, max_mobile_coverage - self.telecommunications.get('mobile_coverage', 0.8))
        self.telecommunications['mobile_coverage'] = min(max_mobile_coverage, 
                                                        self.telecommunications.get('mobile_coverage', 0.8) + 
                                                        remaining_mobile_gap * mobile_growth_rate)
        
        # Update internet coverage
        max_internet_coverage = self.config.get('max_internet_coverage', 0.95)
        internet_growth_rate = self.config.get('internet_growth_rate', 0.04)
        remaining_internet_gap = max(0, max_internet_coverage - self.telecommunications.get('internet_coverage', 0.5))
        self.telecommunications['internet_coverage'] = min(max_internet_coverage,
                                                          self.telecommunications.get('internet_coverage', 0.5) + 
                                                          remaining_internet_gap * internet_growth_rate)
        
        # Update broadband quality
        self.telecommunications['broadband_quality'] = min(
            1.0,
            self.telecommunications.get('broadband_quality', 0.4) + 
            self.config.get('broadband_improvement_rate', 0.03)
        )
        
        # Update digital inclusion (affected by education and economic factors)
        self.telecommunications['digital_inclusion'] = min(
            1.0,
            self.telecommunications.get('digital_inclusion', 0.3) + 
            self.config.get('digital_inclusion_rate', 0.02)
        )
        
        # Update network resilience
        self.telecommunications['network_resilience'] = min(
            1.0,
            self.telecommunications.get('network_resilience', 0.5) + 
            self.config.get('network_resilience_improvement', 0.01)
        )
            
    def _update_energy(self) -> None:
        """Update energy systems."""
        # GDP and population impact on energy demand
        gdp_growth = self.shared_state.get('economic', {}).get('gdp_growth', 0.03)
        population_growth = self.shared_state.get('demographic', {}).get('population_growth', 0.01)
        
        # Energy demand growth based on economic and population factors
        energy_demand_growth = (gdp_growth * 1.2) + (population_growth * 0.5)
        
        # Government policy impact
        renewable_policy = self.shared_state.get('governance', {}).get('environmental_policy_strength', 0.5)
        
        # Investment constraints
        investment_factor = np.random.normal(1.0, 0.1) * self.shared_state.get('governance', {}).get('governance_effectiveness_index', 0.5)
        
        # Update generation with realistic growth rates and tech adoption curves
        # Power capacity grows with demand but constrained by investment 
        self.energy['generation']['total_capacity'] = np.clip(
            self.energy['generation']['total_capacity'] * (
                1 + self.config.get('power_capacity_growth', 0.03) * energy_demand_growth * investment_factor
            ),
            self.energy['generation']['total_capacity'] * 0.98,  # Small depreciation possible
            self.energy['generation']['total_capacity'] * 1.15   # Realistic annual growth ceiling
        )
        
        # S-curve adoption for renewables with policy impact
        current_renewable = self.energy['generation']['renewable_share']
        # Use logistic growth curve that slows as it approaches ceiling
        renewable_growth_rate = self.config.get('renewable_growth_rate', 0.05) * renewable_policy
        renewable_ceiling = 0.7  # Maximum realistic renewable percentage
        renewable_s_factor = 1 - (current_renewable / renewable_ceiling)
        
        self.energy['generation']['renewable_share'] = np.clip(
            current_renewable + (renewable_growth_rate * renewable_s_factor * current_renewable),
            current_renewable * 0.98,  # Allow slight decreases due to policy changes
            renewable_ceiling  # Ceiling constraint
        )
        
        # Load shedding decreases with capacity and grid improvements but increases with demand spikes
        capacity_ratio = self.energy['generation']['total_capacity'] / (self.shared_state.get('demographic', {}).get('total_population', 160000000) * 0.0002)  # approx per capita capacity need
        demand_spike = np.random.normal(1.0, 0.2)  # Random demand fluctuations
        
        self.energy['generation']['load_shedding'] = np.clip(
            self.energy['generation']['load_shedding'] * (
                1 - (self.config.get('grid_improvement_rate', 0.02) * investment_factor) + 
                (0.005 * (demand_spike - capacity_ratio))
            ),
            0.0,  # Minimum load shedding
            0.3   # Maximum realistic load shedding
        )
        
        # Grid efficiency improvements with technology adoption and investment
        self.energy['distribution']['grid_efficiency'] = np.clip(
            self.energy['distribution']['grid_efficiency'] + (
                self.config.get('grid_improvement_rate', 0.01) * investment_factor * 
                (1 - self.energy['distribution']['grid_efficiency'])  # Diminishing returns as efficiency approaches 100%
            ),
            self.energy['distribution']['grid_efficiency'] * 0.99,  # Allow slight degradation
            0.97  # Realistic maximum efficiency
        )
        
        # Transmission loss reduction with grid modernization
        self.energy['distribution']['transmission_loss'] = np.clip(
            self.energy['distribution']['transmission_loss'] * (
                1 - self.config.get('grid_improvement_rate', 0.02) * investment_factor
            ),
            0.03,  # Minimum realistic transmission loss
            self.energy['distribution']['transmission_loss'] * 1.02  # Allow slight increases with grid aging
        )
        
        # Access rate improvement based on rural electrification programs
        rural_focus = self.shared_state.get('governance', {}).get('rural_development_index', 0.5)
        current_access = self.energy['distribution']['access_rate']
        # Use logistic growth that slows as it approaches 100%
        access_growth_factor = (1 - current_access) * self.config.get('electrification_rate', 0.02) * rural_focus
        
        self.energy['distribution']['access_rate'] = np.clip(
            current_access + access_growth_factor,
            current_access,  # Access shouldn't decrease
            1.0  # Maximum possible access rate
        )
    
    def _update_housing(self) -> None:
        """Update housing conditions."""
        # Initialize housing if not already done
        if not hasattr(self, 'housing') or not self.housing:
            self.housing = {
                'urban': {
                    'total_units': self.config.get('initial_urban_housing', 10000000),
                    'quality_index': self.config.get('initial_urban_housing_quality', 0.6),
                    'slum_ratio': self.config.get('initial_slum_ratio', 0.3)
                },
                'rural': {
                    'total_units': self.config.get('initial_rural_housing', 15000000),
                    'quality_index': self.config.get('initial_rural_housing_quality', 0.5),
                    'vulnerability_index': self.config.get('initial_housing_vulnerability', 0.4)
                }
            }
            
        # Get relevant factors from shared state
        population_growth = self.shared_state.get('demographic', {}).get('population_growth', 0.01)
        urbanization_rate = self.shared_state.get('demographic', {}).get('urbanization_rate', 0.02)
        gdp_growth = self.shared_state.get('economic', {}).get('gdp_growth', 0.03)
        gov_effectiveness = self.shared_state.get('governance', {}).get('governance_effectiveness_index', 0.5)
        disaster_preparedness = self.shared_state.get('environmental', {}).get('disaster_preparedness', 0.5)
        
        # Update urban housing
        urban_growth_rate = population_growth + urbanization_rate
        
        # Urban housing units increase with urbanization and population growth
        self.housing['urban']['total_units'] = self.housing['urban']['total_units'] * (1 + urban_growth_rate)
        
        # Housing quality improves with economic growth and governance
        quality_improvement_rate = self.config.get('housing_quality_improvement', 0.01) * gdp_growth * gov_effectiveness
        self.housing['urban']['quality_index'] = np.clip(
            self.housing['urban']['quality_index'] + quality_improvement_rate * (1 - self.housing['urban']['quality_index']),
            self.housing['urban']['quality_index'] * 0.99,  # Allow slight degradation
            0.95  # Maximum realistic quality
        )
        
        # Slum ratio decreases with economic growth and good governance
        slum_reduction_rate = self.config.get('slum_reduction_rate', 0.02) * gdp_growth * gov_effectiveness
        self.housing['urban']['slum_ratio'] = np.clip(
            self.housing['urban']['slum_ratio'] * (1 - slum_reduction_rate),
            0.05,  # Minimum realistic slum ratio
            self.housing['urban']['slum_ratio'] * 1.01  # Allow slight increases with rapid urbanization
        )
        
        # Rural housing updates
        rural_growth_rate = population_growth - urbanization_rate
        
        # Rural housing units change with population shifts
        self.housing['rural']['total_units'] = max(
            self.housing['rural']['total_units'] * (1 + rural_growth_rate),
            self.housing['rural']['total_units'] * 0.99  # Prevent too rapid decline
        )
        
        # Rural housing quality improves more slowly than urban
        rural_quality_improvement = self.config.get('rural_housing_improvement', 0.005) * gdp_growth * gov_effectiveness
        self.housing['rural']['quality_index'] = np.clip(
            self.housing['rural']['quality_index'] + rural_quality_improvement * (1 - self.housing['rural']['quality_index']),
            self.housing['rural']['quality_index'] * 0.99,  # Allow slight degradation
            0.9  # Maximum realistic rural quality
        )
        
        # Housing vulnerability decreases with disaster preparedness measures
        vulnerability_reduction = self.config.get('vulnerability_reduction', 0.01) * disaster_preparedness
        self.housing['rural']['vulnerability_index'] = np.clip(
            self.housing['rural']['vulnerability_index'] * (1 - vulnerability_reduction),
            0.1,  # Minimum vulnerability
            self.housing['rural']['vulnerability_index'] * 1.02  # Allow slight increases with climate change
        )
        
    def _update_supply_chains(self) -> None:
        """Update supply chain systems."""
        # GDP impact factor - infrastructure development correlates with economic growth
        gdp_impact = np.clip(self.shared_state.get('economic', {}).get('gdp_growth', 0.03), 0.01, 0.08)
        
        # Government effectiveness impact
        gov_effectiveness = np.clip(self.shared_state.get('governance', {}).get('governance_effectiveness_index', 0.5), 0.2, 0.9)
        
        # Climate impact factor - extreme weather events slow infrastructure development
        climate_impact = 1 - (self.shared_state.get('environmental', {}).get('extreme_weather_frequency', 0.1) * 0.2)
        
        # Budget constraint - random variation to simulate funding fluctuations
        budget_factor = np.random.normal(1.0, 0.05) * gov_effectiveness
        
        # Update garment resilience
        self.supply_chains['garment']['resilience_index'] = np.clip(
            self.supply_chains['garment']['resilience_index'] * (
                1 + self.config.get('garment_improvement_rate', 0.005) * budget_factor * climate_impact
            ),
            0.0, 1.0
        )
        
        # Update garment logistics efficiency
        self.supply_chains['garment']['logistics_efficiency'] = np.clip(
            self.supply_chains['garment']['logistics_efficiency'] * (
                1 + self.config.get('garment_logistics_improvement', 0.005) * budget_factor * climate_impact
            ),
            0.0, 1.0
        )
        
        # Update garment port efficiency
        self.supply_chains['garment']['port_efficiency'] = np.clip(
            self.supply_chains['garment']['port_efficiency'] * (
                1 + self.config.get('garment_port_improvement', 0.005) * budget_factor * climate_impact
            ),
            0.0, 1.0
        )
        
        # Update agriculture resilience
        self.supply_chains['agriculture']['resilience_index'] = np.clip(
            self.supply_chains['agriculture']['resilience_index'] * (
                1 + self.config.get('agriculture_improvement_rate', 0.005) * budget_factor * climate_impact
            ),
            0.0, 1.0
        )
        
        # Update agriculture storage capacity
        self.supply_chains['agriculture']['storage_capacity'] = np.clip(
            self.supply_chains['agriculture']['storage_capacity'] * (
                1 + self.config.get('agriculture_storage_improvement', 0.005) * budget_factor * climate_impact
            ),
            0.0, 1.0
        )
        
        # Update agriculture distribution efficiency
        self.supply_chains['agriculture']['distribution_efficiency'] = np.clip(
            self.supply_chains['agriculture']['distribution_efficiency'] * (
                1 + self.config.get('agriculture_distribution_improvement', 0.005) * budget_factor * climate_impact
            ),
            0.0, 1.0
        )
    
    def _calculate_infrastructure_indicators(self) -> None:
        """Calculate various infrastructure indicators."""
        # Quality index - weighted average of sub-indices with realistic weights
        transport_index = (
            self.transportation['road_network']['paved_ratio'] * 0.5 +
            self.transportation['waterways']['navigable_ratio'] * 0.2 +
            self.transportation['railways']['operational_ratio'] * 0.3
        ) * 0.25  # Transportation contributes 25% to overall quality
        
        energy_index = (
            (1 - self.energy['generation']['load_shedding']) * 0.5 +
            self.energy['distribution']['grid_efficiency'] * 0.3 +
            self.energy['distribution']['access_rate'] * 0.2
        ) * 0.25  # Energy contributes 25% to overall quality
        
        telecom_index = (
            self.telecommunications['mobile']['network_coverage'] * 0.4 +
            self.telecommunications['internet']['penetration_rate'] * 0.3 +
            self.telecommunications['internet']['broadband_access'] * 0.3
        ) * 0.2  # Telecom contributes 20% to overall quality
        
        housing_index = (
            self.housing['urban']['quality_index'] * 0.5 +
            self.housing['rural']['quality_index'] * 0.3 +
            (1 - self.housing['urban']['slum_ratio']) * 0.2
        ) * 0.15  # Housing contributes 15% to overall quality
        
        supply_chain_index = (
            self.supply_chains['garment']['logistics_efficiency'] * 0.5 +
            self.supply_chains['agriculture']['distribution_efficiency'] * 0.5
        ) * 0.15  # Supply chains contribute 15% to overall quality
        
        # Add natural constraints and random variations to make the model more realistic
        natural_disaster_impact = self.shared_state.get('environmental', {}).get('extreme_weather_impact', 0.0)
        funding_variability = np.random.normal(1.0, 0.02)  # Small random fluctuations
        
        # Calculate overall index with disaster impact and funding variability
        self.state['infrastructure_quality_index'] = np.clip(
            (transport_index + energy_index + telecom_index + housing_index + supply_chain_index) * 
            (1 - natural_disaster_impact * 0.2) * funding_variability,
            0.0, 1.0
        )
    
    def _calculate_quality_index(self) -> float:
        """Calculate overall infrastructure quality index."""
        return (
            self.transportation['road_network']['paved_ratio'] * 0.2 +
            self.energy['distribution']['grid_efficiency'] * 0.2 +
            self.housing['urban']['quality_index'] * 0.2 +
            self.housing['rural']['quality_index'] * 0.2 +
            self.telecommunications['mobile']['network_coverage'] * 0.2
        )
    
    def _calculate_connectivity_index(self) -> float:
        """Calculate connectivity index."""
        return (
            self.transportation['road_network']['paved_ratio'] * 0.3 +
            self.transportation['waterways']['navigable_ratio'] * 0.2 +
            self.transportation['railways']['operational_ratio'] * 0.2 +
            self.telecommunications['internet']['penetration_rate'] * 0.3
        )
    
    def _calculate_resilience_index(self) -> float:
        """Calculate infrastructure resilience index."""
        return (
            self.supply_chains['garment']['resilience_index'] * 0.3 +
            self.supply_chains['agriculture']['resilience_index'] * 0.3 +
            self.energy['generation']['renewable_share'] * 0.2 +
            self.housing['rural']['vulnerability_index'] * 0.2
        )
    
    def _calculate_efficiency_index(self) -> float:
        """Calculate infrastructure efficiency index."""
        return (
            self.transportation['waterways']['efficiency'] * 0.2 +
            self.energy['distribution']['grid_efficiency'] * 0.2 +
            self.supply_chains['garment']['logistics_efficiency'] * 0.2 +
            self.supply_chains['agriculture']['distribution_efficiency'] * 0.2 +
            self.telecommunications['internet']['speed'] * 0.2
        ) 