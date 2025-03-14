#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
System Integrator for the Bangladesh simulation model.
This module provides utilities for connecting and integrating different subsystems
to ensure proper interaction and feedback loops between components.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemIntegrator:
    """
    Utility for managing interactions and feedback loops between different
    subsystems of the Bangladesh simulation model.
    """
    
    def __init__(self, config):
        """
        Initialize the system integrator with configuration.
        
        Args:
            config (dict): Configuration dictionary for the simulation
        """
        self.config = config
        self.interaction_matrices = {}
        self.feedback_loops = {}
        
        # Initialize interaction matrices
        self._init_interaction_matrices()
        
        logger.info("SystemIntegrator initialized successfully")
    
    def _init_interaction_matrices(self):
        """Initialize interaction matrices between subsystems."""
        
        # Define the strength of interactions between subsystems (0-1 scale)
        # Row influences column
        systems = ['economic', 'environmental', 'demographic', 'infrastructure', 'governance']
        n_systems = len(systems)
        
        # Default interaction matrix if not specified in config
        default_matrix = np.array([
            # Economic, Environmental, Demographic, Infrastructure, Governance
            [0.0, 0.7, 0.6, 0.8, 0.5],  # Economic impact on others
            [0.6, 0.0, 0.5, 0.7, 0.4],  # Environmental impact on others
            [0.7, 0.6, 0.0, 0.6, 0.5],  # Demographic impact on others
            [0.8, 0.5, 0.6, 0.0, 0.4],  # Infrastructure impact on others
            [0.8, 0.7, 0.5, 0.9, 0.0],  # Governance impact on others
        ])
        
        # Use config values if available, otherwise use defaults
        matrix = self.config.get('system_interactions', {}).get('matrix', default_matrix)
        
        # Store as DataFrame for easier access
        self.interaction_matrices['main'] = pd.DataFrame(
            matrix,
            index=systems,
            columns=systems
        )
        
        # Create more specific interaction matrices
        
        # Governance-Infrastructure interactions
        gov_infra_matrix = self._create_governance_infrastructure_matrix()
        self.interaction_matrices['governance_infrastructure'] = gov_infra_matrix
        
        logger.info(f"Initialized {len(self.interaction_matrices)} interaction matrices")
    
    def _create_governance_infrastructure_matrix(self):
        """
        Create a detailed interaction matrix between governance and infrastructure.
        
        Returns:
            pd.DataFrame: Matrix of interactions
        """
        # Governance factors
        gov_factors = [
            'institutional_effectiveness', 
            'corruption_index',
            'policy_effectiveness', 
            'regulatory_quality',
            'political_stability'
        ]
        
        # Infrastructure components
        infra_components = [
            'transport_quality',
            'energy_reliability',
            'water_sanitation',
            'telecom_coverage',
            'urban_planning'
        ]
        
        # Default impact matrix (governance row -> infrastructure column)
        default_matrix = np.array([
            # Transport, Energy, Water, Telecom, Urban
            [0.8, 0.7, 0.6, 0.5, 0.9],  # Institutional effectiveness impact
            [-0.7, -0.6, -0.7, -0.5, -0.8],  # Corruption impact (negative)
            [0.6, 0.7, 0.8, 0.6, 0.7],  # Policy effectiveness impact
            [0.7, 0.8, 0.7, 0.8, 0.8],  # Regulatory quality impact
            [0.5, 0.6, 0.4, 0.5, 0.6],  # Political stability impact
        ])
        
        # Use config values if available, otherwise use defaults
        matrix = self.config.get('system_interactions', {}).get(
            'governance_infrastructure', default_matrix)
        
        # Store as DataFrame
        return pd.DataFrame(
            matrix,
            index=gov_factors,
            columns=infra_components
        )
    
    def calculate_infrastructure_governance_impacts(self, governance_system, infrastructure_system):
        """
        Calculate impacts between governance and infrastructure systems.
        
        Args:
            governance_system: The governance system component
            infrastructure_system: The infrastructure system component
            
        Returns:
            dict: Dictionary of calculated impacts
        """
        impacts = {
            'governance_to_infrastructure': {},
            'infrastructure_to_governance': {}
        }
        
        # Get the governance-infrastructure interaction matrix
        gov_infra_matrix = self.interaction_matrices['governance_infrastructure']
        
        # Calculate governance impact on infrastructure
        gov_indicators = {
            'institutional_effectiveness': getattr(governance_system, 'institutional_effectiveness', 
                                                 getattr(governance_system.institutional_system, 'institutional_effectiveness', 0.5)),
            'corruption_index': getattr(governance_system, 'corruption_index', 
                                      getattr(governance_system.institutional_system, 'corruption_index', 0.5)),
            'policy_effectiveness': getattr(governance_system, 'policy_effectiveness', 0.5),
            'regulatory_quality': getattr(governance_system, 'regulatory_quality', 
                                        getattr(governance_system.institutional_system, 'regulatory_quality', 0.5)),
            'political_stability': getattr(governance_system, 'political_stability', 0.5)
        }
        
        # Calculate weighted impact on each infrastructure component
        for infra_component in gov_infra_matrix.columns:
            impact = 0
            for gov_factor, value in gov_indicators.items():
                if gov_factor in gov_infra_matrix.index:
                    impact += value * gov_infra_matrix.loc[gov_factor, infra_component]
            
            impacts['governance_to_infrastructure'][infra_component] = impact
        
        # Calculate infrastructure impact on governance
        # This could be based on infrastructure quality, investment efficiency, etc.
        infra_quality = infrastructure_system.get_overall_quality()
        
        # Define how infrastructure affects governance
        # Higher infrastructure quality generally improves governance capacity
        governance_impact = {
            'institutional_effectiveness': infra_quality * 0.3,
            'policy_effectiveness': infra_quality * 0.2,
            'public_service_delivery': infra_quality * 0.5
        }
        
        impacts['infrastructure_to_governance'] = governance_impact
        
        return impacts
    
    def apply_cross_system_effects(self, simulation):
        """
        Apply cross-system effects for the current simulation step.
        This ensures that changes in one system properly affect other systems.
        
        Args:
            simulation: The main simulation object with all subsystems
            
        Returns:
            dict: Summary of applied cross-system effects
        """
        effects_summary = {}
        
        # Apply governance effects on infrastructure
        if hasattr(simulation, 'governance_system') and hasattr(simulation, 'infrastructure_system'):
            gov_infra_impacts = self.calculate_infrastructure_governance_impacts(
                simulation.governance_system, 
                simulation.infrastructure_system
            )
            
            # Apply the calculated impacts to the respective systems
            self._apply_governance_to_infrastructure_effects(
                simulation.governance_system,
                simulation.infrastructure_system,
                gov_infra_impacts['governance_to_infrastructure']
            )
            
            self._apply_infrastructure_to_governance_effects(
                simulation.infrastructure_system,
                simulation.governance_system,
                gov_infra_impacts['infrastructure_to_governance']
            )
            
            effects_summary['governance_infrastructure'] = gov_infra_impacts
        
        # Apply other cross-system effects here
        # ...
        
        return effects_summary
    
    def _apply_governance_to_infrastructure_effects(self, governance_system, 
                                                   infrastructure_system, impacts):
        """
        Apply governance effects to infrastructure system.
        
        Args:
            governance_system: The governance system component
            infrastructure_system: The infrastructure system component
            impacts (dict): Calculated impacts to apply
        """
        # Map impact categories to infrastructure system methods or attributes
        impact_mapping = {
            'transport_quality': 'transport_quality_adjustment',
            'energy_reliability': 'energy_reliability_adjustment',
            'water_sanitation': 'water_sanitation_adjustment',
            'telecom_coverage': 'telecom_coverage_adjustment',
            'urban_planning': 'urban_planning_adjustment'
        }
        
        # Apply each impact
        for impact_type, impact_value in impacts.items():
            if impact_type in impact_mapping:
                method_name = impact_mapping[impact_type]
                if hasattr(infrastructure_system, method_name):
                    method = getattr(infrastructure_system, method_name)
                    method(impact_value)
                    logger.debug(f"Applied governance impact on {impact_type}: {impact_value:.4f}")
    
    def _apply_infrastructure_to_governance_effects(self, infrastructure_system, 
                                                  governance_system, impacts):
        """
        Apply infrastructure effects to governance system.
        
        Args:
            infrastructure_system: The infrastructure system component
            governance_system: The governance system component
            impacts (dict): Calculated impacts to apply
        """
        # Map impact categories to governance system methods or attributes
        impact_mapping = {
            'institutional_effectiveness': 'adjust_institutional_effectiveness',
            'policy_effectiveness': 'adjust_policy_effectiveness',
            'public_service_delivery': 'adjust_public_service_delivery'
        }
        
        # Apply each impact
        for impact_type, impact_value in impacts.items():
            if impact_type in impact_mapping:
                method_name = impact_mapping[impact_type]
                if hasattr(governance_system, method_name):
                    method = getattr(governance_system, method_name)
                    method(impact_value)
                    logger.debug(f"Applied infrastructure impact on {impact_type}: {impact_value:.4f}")


# Utility functions for integration
def calculate_combined_effect(primary_effect, secondary_effect, weight=0.7):
    """
    Calculate a combined effect from primary and secondary factors.
    
    Args:
        primary_effect (float): Main effect value (0-1 scale)
        secondary_effect (float): Secondary effect value (0-1 scale)
        weight (float): Weight of primary effect (0-1)
        
    Returns:
        float: Combined effect value
    """
    return (primary_effect * weight) + (secondary_effect * (1 - weight))


def normalize_effect(effect_value, min_value=0, max_value=1):
    """
    Normalize an effect value to a specified range.
    
    Args:
        effect_value (float): Value to normalize
        min_value (float): Minimum of target range
        max_value (float): Maximum of target range
        
    Returns:
        float: Normalized value
    """
    return min(max(effect_value, min_value), max_value)


def apply_diminishing_returns(effect_value, diminishing_factor=0.7):
    """
    Apply diminishing returns to an effect value.
    
    Args:
        effect_value (float): Original effect value
        diminishing_factor (float): Factor controlling diminishing returns (0-1)
        
    Returns:
        float: Effect value with diminishing returns applied
    """
    return effect_value ** (1 / diminishing_factor)


# Example integration test function
def test_governance_infrastructure_integration(config):
    """
    Test the integration between governance and infrastructure systems.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if tests pass, False otherwise
    """
    try:
        # Create integrator
        integrator = SystemIntegrator(config)
        
        # Check if interaction matrices are created
        if 'governance_infrastructure' not in integrator.interaction_matrices:
            logger.error("Governance-infrastructure interaction matrix not found")
            return False
        
        # Check matrix dimensions
        matrix = integrator.interaction_matrices['governance_infrastructure']
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            logger.error(f"Governance-infrastructure matrix has unexpected shape: {matrix.shape}")
            return False
        
        logger.info("Governance-infrastructure integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        return False
