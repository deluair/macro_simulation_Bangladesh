#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Infrastructure system model for Bangladesh simulation.
This module implements transportation networks, energy infrastructure,
water systems, telecommunications, and urban development.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm, poisson
import networkx as nx


class InfrastructureSystem:
    """
    Infrastructure system model representing Bangladesh's physical infrastructure
    including transportation, energy, water, telecommunications, and urban systems.
    """
    
    def __init__(self, config, data_loader):
        """
        Initialize the infrastructure system with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the infrastructure system
            data_loader (DataLoader): Data loading utility for infrastructure data
        """
        self.config = config
        self.data_loader = data_loader
        
        # Load initial infrastructure data
        self.infrastructure_data = data_loader.load_infrastructure_data()
        
        # Set up time-related variables
        self.current_year = config.get('start_year', 2023)
        self.time_step = config.get('time_step', 1.0)
        self.base_year = config.get('base_year', 2000)
        
        # Transportation networks
        self.transport_network = self._initialize_transport_network()
        self.road_coverage = self.infrastructure_data.get('road_coverage', 0.65)  # Fraction of potential connections
        self.road_quality = self.infrastructure_data.get('road_quality', 0.45)  # 0-1 scale
        self.rail_coverage = self.infrastructure_data.get('rail_coverage', 0.15)  # Fraction of potential connections
        self.rail_quality = self.infrastructure_data.get('rail_quality', 0.4)  # 0-1 scale
        self.port_capacity = self.infrastructure_data.get('port_capacity', 2.5e6)  # TEU annual capacity
        self.airport_capacity = self.infrastructure_data.get('airport_capacity', 8e6)  # Passengers per year
        
        # Energy infrastructure
        self.electricity_coverage = self.infrastructure_data.get('electricity_coverage', 0.85)  # % of population covered
        self.electricity_reliability = self.infrastructure_data.get('electricity_reliability', 0.7)  # 0-1 scale
        self.generation_capacity = self.infrastructure_data.get('generation_capacity', 20000)  # MW
        self.energy_mix = self.infrastructure_data.get('energy_mix', {
            'natural_gas': 0.6,
            'coal': 0.1,
            'oil': 0.1,
            'hydro': 0.02,
            'solar': 0.05,
            'wind': 0.01,
            'nuclear': 0.0,
            'biomass': 0.12
        })
        self.transmission_losses = self.infrastructure_data.get('transmission_losses', 0.15)  # % of generation lost
        
        # Water infrastructure
        self.water_supply_coverage = self.infrastructure_data.get('water_supply_coverage', 0.7)  # % of population
        self.sanitation_coverage = self.infrastructure_data.get('sanitation_coverage', 0.6)  # % of population
        self.water_treatment_capacity = self.infrastructure_data.get('water_treatment_capacity', 2e9)  # liters per day
        self.irrigation_coverage = self.infrastructure_data.get('irrigation_coverage', 0.55)  # % of agricultural land
        
        # Telecommunications
        self.mobile_coverage = self.infrastructure_data.get('mobile_coverage', 0.95)  # % of population
        self.internet_coverage = self.infrastructure_data.get('internet_coverage', 0.65)  # % of population
        self.broadband_penetration = self.infrastructure_data.get('broadband_penetration', 0.15)  # % of population
        self.digital_services = self.infrastructure_data.get('digital_services', 0.5)  # 0-1 scale
        
        # Urban infrastructure
        self.urbanization_level = self.infrastructure_data.get('urbanization_level', 0.38)  # % population in urban areas
        self.urban_planning_quality = self.infrastructure_data.get('urban_planning_quality', 0.4)  # 0-1 scale
        self.housing_adequacy = self.infrastructure_data.get('housing_adequacy', 0.55)  # 0-1 scale
        self.waste_management = self.infrastructure_data.get('waste_management', 0.4)  # 0-1 scale
        
        # Investment parameters
        self.annual_infrastructure_investment = self.infrastructure_data.get('annual_infrastructure_investment', 0.05)  # % of GDP
        self.investment_allocation = self.infrastructure_data.get('investment_allocation', {
            'transport': 0.35,
            'energy': 0.3,
            'water': 0.15,
            'telecom': 0.1,
            'urban': 0.1
        })
        
        # Regional infrastructure variation
        self.regions = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh']
        self.regional_infrastructure = {}
        for region in self.regions:
            self.regional_infrastructure[region] = {
                'road_coverage': self.road_coverage,
                'electricity_coverage': self.electricity_coverage,
                'water_supply_coverage': self.water_supply_coverage,
                'internet_coverage': self.internet_coverage
            }
            
            # Add regional variation (±20%)
            for key in self.regional_infrastructure[region]:
                variation = 1.0 + np.random.uniform(-0.2, 0.2)
                # Dhaka and Chittagong tend to have better infrastructure
                if region in ['Dhaka', 'Chittagong'] and np.random.random() < 0.7:
                    variation = 1.0 + np.random.uniform(0, 0.2)
                # Remote areas tend to have worse infrastructure
                if region in ['Sylhet', 'Rangpur', 'Barisal'] and np.random.random() < 0.7:
                    variation = 1.0 - np.random.uniform(0, 0.2)
                    
                self.regional_infrastructure[region][key] *= variation
                
                # Ensure values stay in reasonable range (0.1 to 1.0)
                self.regional_infrastructure[region][key] = max(0.1, min(0.99, self.regional_infrastructure[region][key]))
        
        print("Infrastructure system initialized")
    
    def _initialize_transport_network(self):
        """
        Initialize the transportation network using NetworkX.
        Represents major cities/districts as nodes and transport links as edges.
        
        Returns:
            dict: Dictionary of transport networks (road, rail, etc.)
        """
        # Create road network
        road_network = nx.Graph()
        
        # Add major cities as nodes
        # Population data is approximate for illustration
        cities = {
            'Dhaka': {'population': 21000000, 'coords': (23.8103, 90.4125)},
            'Chittagong': {'population': 8500000, 'coords': (22.3569, 91.7832)},
            'Khulna': {'population': 2500000, 'coords': (22.8456, 89.5403)},
            'Rajshahi': {'population': 1800000, 'coords': (24.3745, 88.6042)},
            'Sylhet': {'population': 900000, 'coords': (24.8949, 91.8687)},
            'Barisal': {'population': 600000, 'coords': (22.7010, 90.3535)},
            'Rangpur': {'population': 800000, 'coords': (25.7439, 89.2752)},
            'Mymensingh': {'population': 450000, 'coords': (24.7471, 90.4203)},
            'Cox\'s Bazar': {'population': 250000, 'coords': (21.4272, 92.0116)},
            'Comilla': {'population': 350000, 'coords': (23.4607, 91.1809)},
            'Jessore': {'population': 250000, 'coords': (23.1697, 89.2137)},
            'Dinajpur': {'population': 200000, 'coords': (25.6279, 88.6332)},
            'Bogra': {'population': 300000, 'coords': (24.8510, 89.3711)},
            'Narayanganj': {'population': 700000, 'coords': (23.6238, 90.5000)}
        }
        
        # Add nodes to the road network
        for city, attributes in cities.items():
            road_network.add_node(city, **attributes)
        
        # Define adjacency based on geographical proximity and importance
        # Connect major cities with roads
        road_connections = [
            ('Dhaka', 'Chittagong', {'quality': 0.8, 'capacity': 15000}),
            ('Dhaka', 'Sylhet', {'quality': 0.7, 'capacity': 8000}),
            ('Dhaka', 'Mymensingh', {'quality': 0.6, 'capacity': 5000}),
            ('Dhaka', 'Rajshahi', {'quality': 0.7, 'capacity': 10000}),
            ('Dhaka', 'Khulna', {'quality': 0.7, 'capacity': 9000}),
            ('Dhaka', 'Barisal', {'quality': 0.5, 'capacity': 4000}),
            ('Dhaka', 'Narayanganj', {'quality': 0.9, 'capacity': 20000}),
            ('Dhaka', 'Comilla', {'quality': 0.75, 'capacity': 7000}),
            ('Chittagong', 'Cox\'s Bazar', {'quality': 0.7, 'capacity': 6000}),
            ('Chittagong', 'Comilla', {'quality': 0.6, 'capacity': 5000}),
            ('Khulna', 'Jessore', {'quality': 0.6, 'capacity': 4000}),
            ('Khulna', 'Barisal', {'quality': 0.5, 'capacity': 3000}),
            ('Rajshahi', 'Bogra', {'quality': 0.55, 'capacity': 3500}),
            ('Rajshahi', 'Dinajpur', {'quality': 0.5, 'capacity': 2500}),
            ('Rangpur', 'Dinajpur', {'quality': 0.5, 'capacity': 2000}),
            ('Rangpur', 'Bogra', {'quality': 0.45, 'capacity': 2000}),
            ('Sylhet', 'Mymensingh', {'quality': 0.4, 'capacity': 1500})
        ]
        
        # Add edges to road network
        road_network.add_edges_from(road_connections)
        
        # Create rail network (subset of road network with different properties)
        rail_network = nx.Graph()
        
        # Add nodes to rail network (all major cities have rail stations)
        for city, attributes in cities.items():
            rail_network.add_node(city, **attributes)
        
        # Rail connections (fewer than roads)
        rail_connections = [
            ('Dhaka', 'Chittagong', {'quality': 0.65, 'capacity': 8000}),
            ('Dhaka', 'Sylhet', {'quality': 0.55, 'capacity': 4000}),
            ('Dhaka', 'Rajshahi', {'quality': 0.6, 'capacity': 5000}),
            ('Dhaka', 'Khulna', {'quality': 0.6, 'capacity': 5000}),
            ('Dhaka', 'Mymensingh', {'quality': 0.5, 'capacity': 3000}),
            ('Chittagong', 'Comilla', {'quality': 0.5, 'capacity': 2500}),
            ('Khulna', 'Jessore', {'quality': 0.45, 'capacity': 2000}),
            ('Rajshahi', 'Bogra', {'quality': 0.4, 'capacity': 1500})
        ]
        
        # Add edges to rail network
        rail_network.add_edges_from(rail_connections)
        
        # Ports (maritime infrastructure)
        ports = {
            'Chittagong Port': {
                'capacity': 2000000,  # TEU
                'efficiency': 0.7,
                'connected_city': 'Chittagong'
            },
            'Mongla Port': {
                'capacity': 500000,  # TEU
                'efficiency': 0.6,
                'connected_city': 'Khulna'
            },
            'Payra Port': {
                'capacity': 100000,  # TEU
                'efficiency': 0.5,
                'connected_city': 'Barisal'
            }
        }
        
        # Airports
        airports = {
            'Hazrat Shahjalal International Airport': {
                'capacity': 7000000,  # passengers/year
                'international': True,
                'connected_city': 'Dhaka'
            },
            'Shah Amanat International Airport': {
                'capacity': 1000000,
                'international': True,
                'connected_city': 'Chittagong'
            },
            'Osmani International Airport': {
                'capacity': 500000,
                'international': True,
                'connected_city': 'Sylhet'
            },
            'Khan Jahan Ali Airport': {
                'capacity': 200000,
                'international': False,
                'connected_city': 'Khulna'
            },
            'Shah Makhdum Airport': {
                'capacity': 150000,
                'international': False,
                'connected_city': 'Rajshahi'
            },
            'Saidpur Airport': {
                'capacity': 100000,
                'international': False,
                'connected_city': 'Rangpur'
            },
            'Cox\'s Bazar Airport': {
                'capacity': 500000,
                'international': False,
                'connected_city': 'Cox\'s Bazar'
            }
        }
        
        # Return all networks in a dictionary
        return {
            'road_network': road_network,
            'rail_network': rail_network,
            'ports': ports,
            'airports': airports
        }

    def step(self, year, environmental_system=None, demographic_system=None, governance_system=None):
        """
        Execute a single time step of the infrastructure system simulation.
        
        Args:
            year (int): Current simulation year
            environmental_system: Environmental system instance for accessing environmental indicators
            demographic_system: Demographic system instance for accessing demographic indicators
            governance_system: Governance system instance for accessing governance indicators
            
        Returns:
            dict: Updated infrastructure indicators and state
        """
        # Log the current step
        print(f"Executing infrastructure step for year {year}")
        
        # Update current year
        self.current_year = year
        
        # Get inputs from other systems if available
        environmental_inputs = self._get_environmental_inputs(environmental_system) if environmental_system else {}
        demographic_inputs = self._get_demographic_inputs(demographic_system) if demographic_system else {}
        governance_inputs = self._get_governance_inputs(governance_system) if governance_system else {}
        
        # Calculate investment resources (based on governance if available)
        investment = self._calculate_investment_resources(governance_inputs)
        
        # Update transportation networks
        transport_results = self._update_transportation(
            investment=investment.get('transport', self.annual_infrastructure_investment * self.investment_allocation.get('transport', 0.35)),
            demographic_inputs=demographic_inputs,
            environmental_inputs=environmental_inputs
        )
        
        # Update energy infrastructure
        energy_results = self._update_energy_infrastructure(
            investment=investment.get('energy', self.annual_infrastructure_investment * self.investment_allocation.get('energy', 0.3)),
            demographic_inputs=demographic_inputs,
            environmental_inputs=environmental_inputs
        )
        
        # Update water infrastructure
        water_results = self._update_water_infrastructure(
            investment=investment.get('water', self.annual_infrastructure_investment * self.investment_allocation.get('water', 0.15)),
            demographic_inputs=demographic_inputs,
            environmental_inputs=environmental_inputs
        )
        
        # Update telecommunications
        telecom_results = self._update_telecommunications(
            investment=investment.get('telecom', self.annual_infrastructure_investment * self.investment_allocation.get('telecom', 0.1)),
            demographic_inputs=demographic_inputs
        )
        
        # Update urban infrastructure
        urban_results = self._update_urban_infrastructure(
            investment=investment.get('urban', self.annual_infrastructure_investment * self.investment_allocation.get('urban', 0.1)),
            demographic_inputs=demographic_inputs,
            environmental_inputs=environmental_inputs
        )
        
        # Update regional infrastructure
        self._update_regional_infrastructure(
            transport_results=transport_results,
            energy_results=energy_results,
            water_results=water_results,
            telecom_results=telecom_results
        )
        
        # Calculate overall infrastructure quality index
        overall_quality = self._calculate_overall_quality()
        
        # Compile infrastructure metrics
        results = {
            'year': year,
            'transport': transport_results,
            'energy': energy_results,
            'water': water_results,
            'telecom': telecom_results,
            'urban': urban_results,
            'overall_quality': overall_quality,
            'infrastructure_quality_index': self.get_infrastructure_quality_index(),
            'transportation_index': self.get_transportation_index(),
            'energy_index': self.get_energy_index(),
            'water_index': self.get_water_index(),
            'telecom_index': self.get_telecom_index(),
            'urban_index': self.get_urban_index()
        }
        
        return results
    
    def _get_environmental_inputs(self, environmental_system):
        """Extract relevant inputs from the environmental system."""
        inputs = {}
        
        if hasattr(environmental_system, 'flood_risk'):
            inputs['flood_risk'] = environmental_system.flood_risk
        
        if hasattr(environmental_system, 'climate_vulnerability_index'):
            inputs['climate_vulnerability'] = environmental_system.climate_vulnerability_index
        
        if hasattr(environmental_system, 'water_stress_index'):
            inputs['water_stress'] = environmental_system.water_stress_index
        
        if hasattr(environmental_system, 'disaster_probability'):
            inputs['disaster_probability'] = environmental_system.disaster_probability
        
        return inputs
    
    def _get_demographic_inputs(self, demographic_system):
        """Extract relevant inputs from the demographic system."""
        inputs = {}
        
        if hasattr(demographic_system, 'population'):
            inputs['population'] = demographic_system.population
        
        if hasattr(demographic_system, 'urbanization_rate'):
            inputs['urbanization_rate'] = demographic_system.urbanization_rate
        
        if hasattr(demographic_system, 'population_density'):
            inputs['population_density'] = demographic_system.population_density
        
        if hasattr(demographic_system, 'population_growth_rate'):
            inputs['population_growth_rate'] = demographic_system.population_growth_rate
        
        return inputs
    
    def _get_governance_inputs(self, governance_system):
        """Extract relevant inputs from the governance system."""
        inputs = {}
        
        if hasattr(governance_system, 'infrastructure_investment'):
            inputs['infrastructure_investment'] = governance_system.infrastructure_investment
        
        if hasattr(governance_system, 'investment_allocation'):
            inputs['investment_allocation'] = governance_system.investment_allocation
        
        if hasattr(governance_system, 'policy_effectiveness'):
            inputs['policy_effectiveness'] = governance_system.policy_effectiveness
        
        if hasattr(governance_system, 'corruption_index'):
            inputs['corruption_index'] = governance_system.corruption_index
        
        return inputs
    
    def _calculate_investment_resources(self, governance_inputs):
        """Calculate investment resources based on governance inputs."""
        investment_resources = {}
        
        # Base annual infrastructure investment (% of GDP)
        base_investment = self.annual_infrastructure_investment
        
        # If governance provides investment information, use that instead
        if 'infrastructure_investment' in governance_inputs:
            base_investment = governance_inputs['infrastructure_investment']
        
        # Default allocation
        allocation = self.investment_allocation.copy()
        
        # If governance provides allocation information, use that instead
        if 'investment_allocation' in governance_inputs:
            allocation = governance_inputs['investment_allocation']
        
        # Calculate investment for each sector
        for sector, share in allocation.items():
            investment_resources[sector] = base_investment * share
            
            # Apply corruption effect if available (reduces effective investment)
            if 'corruption_index' in governance_inputs:
                corruption_effect = governance_inputs['corruption_index'] * 0.3  # Up to 30% reduction
                investment_resources[sector] *= (1 - corruption_effect)
            
            # Apply policy effectiveness if available (improves investment efficiency)
            if 'policy_effectiveness' in governance_inputs:
                policy_effect = governance_inputs['policy_effectiveness'] * 0.2  # Up to 20% improvement
                investment_resources[sector] *= (1 + policy_effect)
        
        return investment_resources
    
    def _update_transportation(self, investment, demographic_inputs=None, environmental_inputs=None):
        """Update transportation infrastructure based on investment and other factors."""
        results = {}
        
        # Population pressure effect on infrastructure
        population_pressure = 1.0
        if demographic_inputs and 'population_growth_rate' in demographic_inputs:
            population_pressure = 1.0 + demographic_inputs['population_growth_rate']
        
        # Baseline annual degradation rate (infrastructure deterioration)
        degradation_rate = 0.03  # 3% annual degradation
        
        # Environmental impacts increase degradation
        if environmental_inputs and 'flood_risk' in environmental_inputs:
            degradation_rate += environmental_inputs['flood_risk'] * 0.02
        
        # Apply degradation to road quality
        self.road_quality = max(0.1, self.road_quality * (1 - degradation_rate))
        
        # Apply degradation to rail quality
        self.rail_quality = max(0.1, self.rail_quality * (1 - degradation_rate))
        
        # Calculate road investment impact
        road_investment = investment * 0.6  # 60% of transport investment goes to roads
        road_improvement = road_investment * 0.5  # Investment effectiveness factor
        self.road_quality = min(0.95, self.road_quality + road_improvement)
        
        # Expand road coverage based on investment and urbanization
        road_expansion = road_investment * 0.3  # Coverage expansion factor
        if demographic_inputs and 'urbanization_rate' in demographic_inputs:
            road_expansion *= (1 + demographic_inputs['urbanization_rate'] * 0.5)  # Urbanization boosts expansion
        self.road_coverage = min(0.95, self.road_coverage + road_expansion * 0.01)
        
        # Calculate rail investment impact
        rail_investment = investment * 0.3  # 30% of transport investment goes to rail
        rail_improvement = rail_investment * 0.4  # Investment effectiveness factor
        self.rail_quality = min(0.95, self.rail_quality + rail_improvement)
        
        # Expand rail coverage based on investment
        rail_expansion = rail_investment * 0.2  # Coverage expansion factor
        self.rail_coverage = min(0.5, self.rail_coverage + rail_expansion * 0.005)
        
        # Calculate port capacity investment impact
        port_investment = investment * 0.05  # 5% of transport investment goes to ports
        port_capacity_increase = port_investment * 2e5  # Each unit of investment adds 200,000 TEU
        self.port_capacity += port_capacity_increase
        
        # Calculate airport capacity investment impact
        airport_investment = investment * 0.05  # 5% of transport investment goes to airports
        airport_capacity_increase = airport_investment * 5e5  # Each unit of investment adds 500,000 passengers
        self.airport_capacity += airport_capacity_increase
        
        # Compile results
        results = {
            'road_quality': self.road_quality,
            'road_coverage': self.road_coverage,
            'rail_quality': self.rail_quality,
            'rail_coverage': self.rail_coverage,
            'port_capacity': self.port_capacity,
            'airport_capacity': self.airport_capacity,
            'investment': investment
        }
        
        return results
    
    def _update_energy_infrastructure(self, investment, demographic_inputs=None, environmental_inputs=None):
        """Update energy infrastructure based on investment and other factors."""
        results = {}
        
        # Baseline annual degradation
        degradation_rate = 0.02  # 2% annual degradation
        
        # Apply degradation to reliability
        self.electricity_reliability = max(0.3, self.electricity_reliability * (1 - degradation_rate))
        
        # Calculate population pressure on electricity demand
        population_factor = 1.0
        if demographic_inputs and 'population_growth_rate' in demographic_inputs:
            population_factor = 1.0 + demographic_inputs['population_growth_rate'] * 2
        
        # Increase generation capacity based on investment
        capacity_increase = investment * 1000  # Each unit of investment adds 1000 MW
        self.generation_capacity += capacity_increase
        
        # Improve coverage based on investment
        coverage_improvement = investment * 0.2  # Coverage improvement factor
        self.electricity_coverage = min(0.99, self.electricity_coverage + coverage_improvement * 0.05)
        
        # Improve reliability based on investment
        reliability_improvement = investment * 0.3  # Reliability improvement factor
        self.electricity_reliability = min(0.95, self.electricity_reliability + reliability_improvement)
        
        # Reduce transmission losses based on investment
        loss_reduction = investment * 0.1  # Loss reduction factor
        self.transmission_losses = max(0.05, self.transmission_losses - loss_reduction * 0.01)
        
        # Shift energy mix based on investment and climate policies
        if 'climate_vulnerability' in environmental_inputs:
            # Higher climate vulnerability accelerates clean energy transition
            clean_energy_shift = investment * 0.2 * (1 + environmental_inputs['climate_vulnerability'])
            
            # Reduce fossil fuels
            self.energy_mix['natural_gas'] = max(0.2, self.energy_mix['natural_gas'] - clean_energy_shift * 0.02)
            self.energy_mix['coal'] = max(0.05, self.energy_mix['coal'] - clean_energy_shift * 0.01)
            self.energy_mix['oil'] = max(0.05, self.energy_mix['oil'] - clean_energy_shift * 0.01)
            
            # Increase renewables
            self.energy_mix['solar'] = min(0.3, self.energy_mix['solar'] + clean_energy_shift * 0.02)
            self.energy_mix['wind'] = min(0.15, self.energy_mix['wind'] + clean_energy_shift * 0.01)
            self.energy_mix['biomass'] = min(0.2, self.energy_mix['biomass'] + clean_energy_shift * 0.01)
            
            # Normalize energy mix to ensure sum is 1.0
            total = sum(self.energy_mix.values())
            for source in self.energy_mix:
                self.energy_mix[source] /= total
        
        # Compile results
        results = {
            'electricity_coverage': self.electricity_coverage,
            'electricity_reliability': self.electricity_reliability,
            'generation_capacity': self.generation_capacity,
            'transmission_losses': self.transmission_losses,
            'energy_mix': self.energy_mix,
            'investment': investment
        }
        
        return results
    
    def _update_water_infrastructure(self, investment, demographic_inputs=None, environmental_inputs=None):
        """Update water infrastructure based on investment and other factors."""
        results = {}
        
        # Baseline annual degradation
        degradation_rate = 0.025  # 2.5% annual degradation
        
        # Environmental stress increases degradation
        if environmental_inputs and 'water_stress' in environmental_inputs:
            degradation_rate += environmental_inputs['water_stress'] * 0.02
        
        # Calculate population pressure
        population_factor = 1.0
        if demographic_inputs and 'population_growth_rate' in demographic_inputs:
            population_factor = 1.0 + demographic_inputs['population_growth_rate'] * 1.5
        
        # Improve water supply coverage based on investment
        coverage_improvement = investment * 0.3
        self.water_supply_coverage = min(0.95, self.water_supply_coverage + coverage_improvement * 0.05)
        
        # Improve sanitation coverage based on investment
        sanitation_improvement = investment * 0.3
        self.sanitation_coverage = min(0.9, self.sanitation_coverage + sanitation_improvement * 0.05)
        
        # Increase water treatment capacity based on investment
        capacity_increase = investment * 1e8  # Each unit of investment adds 100 million liters per day
        self.water_treatment_capacity += capacity_increase
        
        # Improve irrigation coverage based on investment
        irrigation_improvement = investment * 0.2
        self.irrigation_coverage = min(0.8, self.irrigation_coverage + irrigation_improvement * 0.03)
        
        # Compile results
        results = {
            'water_supply_coverage': self.water_supply_coverage,
            'sanitation_coverage': self.sanitation_coverage,
            'water_treatment_capacity': self.water_treatment_capacity,
            'irrigation_coverage': self.irrigation_coverage,
            'investment': investment
        }
        
        return results
    
    def _update_telecommunications(self, investment, demographic_inputs=None):
        """Update telecommunications infrastructure based on investment."""
        results = {}
        
        # Calculate population pressure
        population_factor = 1.0
        if demographic_inputs and 'population_growth_rate' in demographic_inputs:
            population_factor = 1.0 + demographic_inputs['population_growth_rate']
        
        # Improve mobile coverage based on investment
        mobile_improvement = investment * 0.2
        self.mobile_coverage = min(0.99, self.mobile_coverage + mobile_improvement * 0.02)
        
        # Improve internet coverage based on investment
        internet_improvement = investment * 0.3
        self.internet_coverage = min(0.95, self.internet_coverage + internet_improvement * 0.04)
        
        # Improve broadband penetration based on investment
        broadband_improvement = investment * 0.3
        self.broadband_penetration = min(0.8, self.broadband_penetration + broadband_improvement * 0.05)
        
        # Improve digital services based on investment
        services_improvement = investment * 0.2
        self.digital_services = min(0.9, self.digital_services + services_improvement * 0.05)
        
        # Compile results
        results = {
            'mobile_coverage': self.mobile_coverage,
            'internet_coverage': self.internet_coverage,
            'broadband_penetration': self.broadband_penetration,
            'digital_services': self.digital_services,
            'investment': investment
        }
        
        return results
    
    def _update_urban_infrastructure(self, investment, demographic_inputs=None, environmental_inputs=None):
        """Update urban infrastructure based on investment and other factors."""
        results = {}
        
        # Calculate urbanization pressure
        urbanization_pressure = 1.0
        if demographic_inputs and 'urbanization_rate' in demographic_inputs:
            urbanization_pressure = 1.0 + demographic_inputs['urbanization_rate'] * 2
        
        # Update urbanization level based on demographic inputs
        if demographic_inputs and 'urbanization_rate' in demographic_inputs:
            self.urbanization_level = demographic_inputs['urbanization_rate']
        else:
            # Default urbanization increase
            self.urbanization_level = min(0.65, self.urbanization_level + 0.005)
        
        # Improve urban planning quality based on investment
        planning_improvement = investment * 0.3
        self.urban_planning_quality = min(0.9, self.urban_planning_quality + planning_improvement * 0.05)
        
        # Improve housing adequacy based on investment
        housing_improvement = investment * 0.4
        self.housing_adequacy = min(0.9, self.housing_adequacy + housing_improvement * 0.03)
        
        # Improve waste management based on investment
        waste_improvement = investment * 0.3
        self.waste_management = min(0.85, self.waste_management + waste_improvement * 0.04)
        
        # Compile results
        results = {
            'urbanization_level': self.urbanization_level,
            'urban_planning_quality': self.urban_planning_quality,
            'housing_adequacy': self.housing_adequacy,
            'waste_management': self.waste_management,
            'investment': investment
        }
        
        return results
    
    def _update_regional_infrastructure(self, transport_results=None, energy_results=None, water_results=None, telecom_results=None):
        """Update regional infrastructure variations."""
        for region in self.regions:
            # Apply small random variations
            for key in self.regional_infrastructure[region]:
                # Apply small random change (±2%)
                self.regional_infrastructure[region][key] *= (1 + np.random.uniform(-0.02, 0.02))
            
            # Update regional road coverage based on transport results
            if transport_results and 'road_coverage' in transport_results:
                # Regions change at different rates
                if region in ['Dhaka', 'Chittagong']:
                    # Major urban areas develop faster
                    adjustment_factor = 1.2
                elif region in ['Sylhet', 'Rangpur', 'Barisal']:
                    # Remote regions develop slower
                    adjustment_factor = 0.8
                else:
                    adjustment_factor = 1.0
                
                # Update the regional value, maintaining regional differences
                current_ratio = self.regional_infrastructure[region]['road_coverage'] / self.road_coverage
                new_value = transport_results['road_coverage'] * current_ratio * adjustment_factor
                self.regional_infrastructure[region]['road_coverage'] = max(0.1, min(0.99, new_value))
            
            # Update regional electricity coverage based on energy results
            if energy_results and 'electricity_coverage' in energy_results:
                # Regions change at different rates
                if region in ['Dhaka', 'Chittagong']:
                    # Major urban areas develop faster
                    adjustment_factor = 1.2
                elif region in ['Sylhet', 'Rangpur', 'Barisal']:
                    # Remote regions develop slower
                    adjustment_factor = 0.8
                else:
                    adjustment_factor = 1.0
                
                # Update the regional value, maintaining regional differences
                current_ratio = self.regional_infrastructure[region]['electricity_coverage'] / self.electricity_coverage
                new_value = energy_results['electricity_coverage'] * current_ratio * adjustment_factor
                self.regional_infrastructure[region]['electricity_coverage'] = max(0.1, min(0.99, new_value))
            
            # Update regional water supply coverage based on water results
            if water_results and 'water_supply_coverage' in water_results:
                # Regions change at different rates
                if region in ['Dhaka', 'Chittagong']:
                    # Major urban areas develop faster
                    adjustment_factor = 1.1
                elif region in ['Sylhet', 'Rangpur', 'Barisal']:
                    # Remote regions develop slower
                    adjustment_factor = 0.9
                else:
                    adjustment_factor = 1.0
                
                # Update the regional value, maintaining regional differences
                current_ratio = self.regional_infrastructure[region]['water_supply_coverage'] / self.water_supply_coverage
                new_value = water_results['water_supply_coverage'] * current_ratio * adjustment_factor
                self.regional_infrastructure[region]['water_supply_coverage'] = max(0.1, min(0.99, new_value))
            
            # Update regional internet coverage based on telecom results
            if telecom_results and 'internet_coverage' in telecom_results:
                # Regions change at different rates
                if region in ['Dhaka', 'Chittagong']:
                    # Major urban areas develop faster
                    adjustment_factor = 1.3
                elif region in ['Sylhet', 'Rangpur', 'Barisal']:
                    # Remote regions develop slower
                    adjustment_factor = 0.7
                else:
                    adjustment_factor = 1.0
                
                # Update the regional value, maintaining regional differences
                current_ratio = self.regional_infrastructure[region]['internet_coverage'] / self.internet_coverage
                new_value = telecom_results['internet_coverage'] * current_ratio * adjustment_factor
                self.regional_infrastructure[region]['internet_coverage'] = max(0.1, min(0.99, new_value))
    
    def get_infrastructure_quality_index(self):
        """Calculate an overall infrastructure quality index (0-1 scale)."""
        return (
            self.get_transportation_index() * 0.3 +
            self.get_energy_index() * 0.25 +
            self.get_water_index() * 0.2 +
            self.get_telecom_index() * 0.15 +
            self.get_urban_index() * 0.1
        )
    
    def get_transportation_index(self):
        """Calculate a transportation quality index (0-1 scale)."""
        return (
            self.road_quality * 0.4 +
            self.road_coverage * 0.2 +
            self.rail_quality * 0.2 +
            self.rail_coverage * 0.1 +
            min(1.0, self.port_capacity / 5e6) * 0.05 +
            min(1.0, self.airport_capacity / 1.5e7) * 0.05
        )
    
    def get_energy_index(self):
        """Calculate an energy infrastructure index (0-1 scale)."""
        # Calculate renewable energy share
        renewable_share = (
            self.energy_mix.get('solar', 0) +
            self.energy_mix.get('wind', 0) +
            self.energy_mix.get('hydro', 0) +
            self.energy_mix.get('biomass', 0)
        )
        
        return (
            self.electricity_coverage * 0.4 +
            self.electricity_reliability * 0.3 +
            (1 - self.transmission_losses) * 0.15 +
            renewable_share * 0.15
        )
    
    def get_water_index(self):
        """Calculate a water infrastructure index (0-1 scale)."""
        return (
            self.water_supply_coverage * 0.4 +
            self.sanitation_coverage * 0.3 +
            min(1.0, self.water_treatment_capacity / 5e9) * 0.15 +
            self.irrigation_coverage * 0.15
        )
    
    def get_telecom_index(self):
        """Calculate a telecommunications index (0-1 scale)."""
        return (
            self.mobile_coverage * 0.3 +
            self.internet_coverage * 0.3 +
            self.broadband_penetration * 0.2 +
            self.digital_services * 0.2
        )
    
    def get_urban_index(self):
        """Calculate an urban infrastructure index (0-1 scale)."""
        return (
            self.urban_planning_quality * 0.4 +
            self.housing_adequacy * 0.3 +
            self.waste_management * 0.3
        )
    
    def get_overall_quality(self):
        """Get overall infrastructure quality (alias for infrastructure_quality_index)."""
        return self.get_infrastructure_quality_index()

    def _calculate_overall_quality(self):
        """
        Calculate the overall infrastructure quality index.
        This is a wrapper for get_overall_quality for internal use.
        
        Returns:
            float: Overall infrastructure quality index (0-1)
        """
        return self.get_overall_quality()
