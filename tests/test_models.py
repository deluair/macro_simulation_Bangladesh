import unittest
import numpy as np
from models.economic_model import EconomicModel
from models.environmental_model import EnvironmentalModel
from models.demographic_model import DemographicModel
from models.infrastructure_model import InfrastructureModel
from models.governance_model import GovernanceModel
from models.simulation import BangladeshSimulation

class TestModels(unittest.TestCase):
    """Test cases for model components."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'simulation': {
                'duration': 30,
                'time_step': 1,
                'random_seed': 42
            },
            'economic': {
                'initial_exchange_rate': 85.0,
                'initial_port_efficiency': 0.7,
                'baseline_gdp': 1000000000,
                'garment_gdp_share': 0.2,
                'agriculture_gdp_share': 0.15,
                'remittances_gdp_share': 0.1,
                'tech_gdp_share': 0.05,
                'informal_gdp_share': 0.5
            },
            'environmental': {
                'initial_temperature': 25.0,
                'initial_precipitation': 0.0,
                'initial_sea_level': 0.0,
                'initial_cyclone_probability': 0.1
            },
            'demographic': {
                'initial_population': 170000000,
                'initial_population_0_14': 0.3,
                'initial_population_15_64': 0.65,
                'initial_population_65_plus': 0.05
            },
            'infrastructure': {
                'initial_road_length': 350000,
                'initial_paved_ratio': 0.6,
                'initial_congestion': 0.5
            },
            'governance': {
                'initial_bureaucracy_efficiency': 0.5,
                'initial_corruption_index': 0.3,
                'initial_transparency': 0.4
            }
        }
    
    def test_economic_model(self):
        """Test economic model functionality."""
        model = EconomicModel(self.config)
        model.initialize()
        
        # Test initial state
        self.assertIn('total_gdp', model.state)
        self.assertIn('unemployment_rate', model.state)
        
        # Test step and update
        model.step()
        model.update()
        
        # Test state updates
        self.assertGreater(model.state['total_gdp'], 0)
        self.assertGreaterEqual(model.state['unemployment_rate'], 0)
        self.assertLessEqual(model.state['unemployment_rate'], 1)
    
    def test_environmental_model(self):
        """Test environmental model functionality."""
        model = EnvironmentalModel(self.config)
        model.initialize()
        
        # Test initial state
        self.assertIn('flood_risk', model.state)
        self.assertIn('crop_yield_index', model.state)
        
        # Test step and update
        model.step()
        model.update()
        
        # Test state updates
        self.assertGreaterEqual(model.state['flood_risk'], 0)
        self.assertLessEqual(model.state['flood_risk'], 1)
        self.assertGreater(model.state['crop_yield_index'], 0)
    
    def test_demographic_model(self):
        """Test demographic model functionality."""
        model = DemographicModel(self.config)
        model.initialize()
        
        # Test initial state
        self.assertIn('population_growth_rate', model.state)
        self.assertIn('human_development_index', model.state)
        
        # Test step and update
        model.step()
        model.update()
        
        # Test state updates
        self.assertGreater(model.state['human_development_index'], 0)
        self.assertLessEqual(model.state['human_development_index'], 1)
    
    def test_infrastructure_model(self):
        """Test infrastructure model functionality."""
        model = InfrastructureModel(self.config)
        model.initialize()
        
        # Test initial state
        self.assertIn('infrastructure_quality_index', model.state)
        self.assertIn('connectivity_index', model.state)
        
        # Test step and update
        model.step()
        model.update()
        
        # Test state updates
        self.assertGreaterEqual(model.state['infrastructure_quality_index'], 0)
        self.assertLessEqual(model.state['infrastructure_quality_index'], 1)
    
    def test_governance_model(self):
        """Test governance model functionality."""
        model = GovernanceModel(self.config)
        model.initialize()
        
        # Test initial state
        self.assertIn('governance_effectiveness_index', model.state)
        self.assertIn('social_progress_index', model.state)
        
        # Test step and update
        model.step()
        model.update()
        
        # Test state updates
        self.assertGreaterEqual(model.state['governance_effectiveness_index'], 0)
        self.assertLessEqual(model.state['governance_effectiveness_index'], 1)
    
    def test_simulation(self):
        """Test main simulation functionality."""
        simulation = BangladeshSimulation(self.config)
        simulation.initialize()
        
        # Test initial state
        self.assertIn('development_index', simulation.state)
        self.assertIn('sustainability_index', simulation.state)
        
        # Test step
        simulation.step()
        
        # Test state updates
        self.assertGreaterEqual(simulation.state['development_index'], 0)
        self.assertLessEqual(simulation.state['development_index'], 1)
        
        # Test history
        self.assertEqual(len(simulation.history), 1)
    
    def test_model_interactions(self):
        """Test interactions between models."""
        simulation = BangladeshSimulation(self.config)
        simulation.initialize()
        
        # Run multiple steps
        for _ in range(5):
            simulation.step()
        
        # Test economic-environmental interaction
        econ_state = simulation.models['economic'].state
        env_state = simulation.models['environmental'].state
        
        # Test economic-demographic interaction
        demo_state = simulation.models['demographic'].state
        
        # Test infrastructure-economic interaction
        infra_state = simulation.models['infrastructure'].state
        
        # Test governance-economic interaction
        gov_state = simulation.models['governance'].state
        
        # Verify interactions have occurred
        self.assertGreater(len(simulation.history), 0)
        self.assertNotEqual(econ_state['total_gdp'], self.config['economic']['baseline_gdp'])
    
    def test_error_handling(self):
        """Test error handling in models."""
        # Test missing required parameters
        invalid_config = {'simulation': {'duration': 30}}
        
        with self.assertRaises(Exception):
            EconomicModel(invalid_config)
        
        # Test invalid state updates
        model = EconomicModel(self.config)
        model.initialize()
        
        with self.assertRaises(Exception):
            model.set_parameter('nonexistent_param', 1.0)

if __name__ == '__main__':
    unittest.main() 