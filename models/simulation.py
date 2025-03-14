from typing import Dict, Any, List
import numpy as np
from .economic_model import EconomicModel
from .environmental_model import EnvironmentalModel
from .demographic_model import DemographicModel
from .infrastructure_model import InfrastructureModel
from .governance_model import GovernanceModel
from datetime import datetime

class BangladeshSimulation:
    """Main simulation class integrating all models for Bangladesh's development trajectory."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation.
        
        Args:
            config: Dictionary containing simulation configuration parameters
        """
        self.config = config
        self.time_step = 0
        self.models = {
            'economic': EconomicModel(config),
            'environmental': EnvironmentalModel(config),
            'demographic': DemographicModel(config),
            'infrastructure': InfrastructureModel(config),
            'governance': GovernanceModel(config)
        }
        self.history: List[Dict[str, Any]] = []
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize all models and simulation state."""
        for model in self.models.values():
            model.initialize()
        
        self.state = {
            'development_index': 0.0,
            'sustainability_index': 0.0,
            'resilience_index': 0.0,
            'wellbeing_index': 0.0
        }
    
    def step(self) -> None:
        """Execute one simulation step."""
        # Share state between models before each update
        self._share_state_between_models()
        
        # Update all models
        for model in self.models.values():
            model.step()
            model.update()
            model.save_state()  # Save model state after update
        
        # Handle model interactions
        self._handle_model_interactions()
        
        # Update simulation state
        self._update_state()
        
        # Save state to history
        self._save_state()
        
        self.time_step += 1
    
    def _handle_model_interactions(self) -> None:
        """Handle interactions between different models."""
        # Economic-Environmental interactions
        self._handle_economic_environmental_interactions()
        
        # Economic-Demographic interactions
        self._handle_economic_demographic_interactions()
        
        # Environmental-Demographic interactions
        self._handle_environmental_demographic_interactions()
        
        # Infrastructure-Economic interactions
        self._handle_infrastructure_economic_interactions()
        
        # Governance-Economic interactions
        self._handle_governance_economic_interactions()
    
    def _handle_economic_environmental_interactions(self) -> None:
        """Handle interactions between economic and environmental models."""
        # Impact of environmental conditions on economic performance
        env_state = self.models['environmental'].state
        econ_state = self.models['economic'].state
        
        # Climate impact on agriculture
        climate_factor = 1 - env_state['flood_risk'] * 0.5
        self.models['economic'].sectors['agriculture']['productivity'] *= climate_factor
        
        # Environmental regulations impact on industry
        env_policy = self.models['governance'].policies['environmental']['effectiveness']
        self.models['economic'].sectors['garment']['productivity'] *= (1 - env_policy * 0.1)
    
    def _handle_economic_demographic_interactions(self) -> None:
        """Handle interactions between economic and demographic models."""
        # Impact of economic conditions on migration
        econ_state = self.models['economic'].state
        demo_state = self.models['demographic'].state
        
        # Economic opportunity impact on rural-urban migration
        opportunity_factor = econ_state['total_gdp'] / self.config.get('baseline_gdp', 1000000000)
        self.models['demographic'].migration['rural_to_urban'] *= opportunity_factor
        
        # Employment impact on poverty
        employment_factor = 1 - self.models['demographic'].employment['labor_force_participation']
        self.models['demographic'].income_distribution['poverty_rate'] *= employment_factor
    
    def _handle_environmental_demographic_interactions(self) -> None:
        """Handle interactions between environmental and demographic models."""
        # Impact of environmental conditions on population
        env_state = self.models['environmental'].state
        demo_state = self.models['demographic'].state
        
        # Climate vulnerability impact on migration
        vulnerability_factor = env_state['flood_risk'] * 0.5
        self.models['demographic'].migration['rural_to_urban'] *= (1 + vulnerability_factor)
        
        # Water quality impact on health
        water_quality = self.models['environmental'].water_systems['water_quality_index']
        self.models['demographic'].population['total'] *= (1 - (1 - water_quality) * 0.01)
    
    def _handle_infrastructure_economic_interactions(self) -> None:
        """Handle interactions between infrastructure and economic models."""
        # Impact of infrastructure on economic performance
        infra_state = self.models['infrastructure'].state
        econ_state = self.models['economic'].state
        
        # Infrastructure quality impact on productivity
        quality_factor = infra_state['infrastructure_quality_index']
        for sector in self.models['economic'].sectors:
            self.models['economic'].sectors[sector]['productivity'] *= quality_factor
        
        # Supply chain efficiency impact on trade
        efficiency_factor = infra_state['efficiency_index']
        self.models['economic'].state['trade_balance'] *= efficiency_factor
    
    def _handle_governance_economic_interactions(self) -> None:
        """Handle interactions between governance and economic models."""
        # Impact of governance on economic performance
        gov_state = self.models['governance'].state
        econ_state = self.models['economic'].state
        
        # Governance effectiveness impact on investment
        effectiveness_factor = gov_state['governance_effectiveness_index']
        for sector in self.models['economic'].sectors:
            self.models['economic'].sectors[sector]['growth_rate'] *= effectiveness_factor
        
        # Policy responsiveness impact on sector performance
        responsiveness_factor = gov_state['policy_responsiveness']
        self.models['economic'].state['total_gdp'] *= (1 + responsiveness_factor * 0.01)
    
    def _update_state(self) -> None:
        """Update simulation state based on all model states."""
        # Calculate development index
        self.state['development_index'] = self._calculate_development_index()
        
        # Calculate sustainability index
        self.state['sustainability_index'] = self._calculate_sustainability_index()
        
        # Calculate resilience index
        self.state['resilience_index'] = self._calculate_resilience_index()
        
        # Calculate wellbeing index
        self.state['wellbeing_index'] = self._calculate_wellbeing_index()
    
    def _calculate_development_index(self) -> float:
        """Calculate overall development index."""
        return (
            self.models['economic'].state['total_gdp'] * 0.3 +
            self.models['demographic'].state['human_development_index'] * 0.3 +
            self.models['infrastructure'].state['infrastructure_quality_index'] * 0.2 +
            self.models['governance'].state['governance_effectiveness_index'] * 0.2
        )
    
    def _calculate_sustainability_index(self) -> float:
        """Calculate sustainability index."""
        return (
            self.models['environmental'].state['environmental_health_index'] * 0.4 +
            self.models['infrastructure'].state['resilience_index'] * 0.3 +
            self.models['governance'].state['policy_responsiveness'] * 0.3
        )
    
    def _calculate_resilience_index(self) -> float:
        """Calculate resilience index."""
        return (
            self.models['environmental'].state['flood_risk'] * 0.3 +
            self.models['infrastructure'].state['resilience_index'] * 0.3 +
            self.models['governance'].state['institutional_quality'] * 0.2 +
            self.models['demographic'].state['social_cohesion_index'] * 0.2
        )
    
    def _calculate_wellbeing_index(self) -> float:
        """Calculate wellbeing index."""
        return (
            self.models['demographic'].state['human_development_index'] * 0.3 +
            self.models['demographic'].state['social_cohesion_index'] * 0.2 +
            self.models['environmental'].state['environmental_health_index'] * 0.2 +
            self.models['governance'].state['social_progress_index'] * 0.3
        )
    
    def _save_state(self) -> None:
        """Save current state to history."""
        state_copy = {
            'time_step': self.time_step,
            'state': self.state.copy(),
            'models': {
                name: model.state.copy() for name, model in self.models.items()
            }
        }
        self.history.append(state_copy)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return self.state
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get simulation history."""
        return self.history
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.time_step = 0
        self.history = []
        self.initialize()
    
    def _share_state_between_models(self) -> None:
        """Share state between models to enable realistic interactions."""
        # Collect current state from all models
        shared_state = {}
        
        # First ensure all models have valid state
        for model_name, model in self.models.items():
            # Make sure state isn't empty
            if not model.state or len(model.state) == 0:
                self.logger.warning(f"Model {model_name} has empty state, initializing it")
                model.state['timestamp'] = datetime.now().isoformat()
                
            # Ensure the model passes its own validation
            try:
                if not model.validate_state():
                    self.logger.warning(f"Model {model_name} state validation failed, using default state")
                    model.state = {'timestamp': datetime.now().isoformat()}
            except Exception as e:
                self.logger.error(f"Error validating state for model {model_name}: {str(e)}")
                model.state = {'timestamp': datetime.now().isoformat()}
                
            # Add state to shared state
            shared_state[model_name] = model.get_state()
        
        # Update shared state in all models
        for model in self.models.values():
            model.update_shared_state(shared_state)