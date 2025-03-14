"""
Report generation module for the Bangladesh Development Simulation Model.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

import jinja2
import jsonschema

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from models.simulation import BangladeshSimulation

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Class for generating simulation reports in various formats."""
    
    def __init__(self, simulation: Any):
        """
        Initialize the report generator with a simulation instance.
        
        Args:
            simulation: BangladeshSimulation instance with results to report
        """
        self.simulation = simulation
        self.output_dir = Path(simulation.config['output']['output_dir'])
        self.report_dir = self.output_dir / 'reports'
        self.report_dir.mkdir(exist_ok=True)
        
        # Get simulation ID for report filenames
        if hasattr(simulation, 'result_processor') and hasattr(simulation.result_processor, 'simulation_id'):
            self.simulation_id = simulation.result_processor.simulation_id
        else:
            self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Set up Jinja2 environment
        self.template_dir = Path('templates')
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Initialize report data
        self.summary_data = {}
        self.detailed_data = {}
        self.json_data = {}
        
        # Load templates
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=True
        )
        
        # Load JSON schema
        with open(self.template_dir / 'report_schema.json', 'r') as f:
            self.schema = json.load(f)
    
    def _prepare_summary_data(self) -> Dict[str, Any]:
        """Prepare data for the summary report."""
        return {
            'summary': {
                'start_time': self.simulation.start_time,
                'end_time': self.simulation.end_time,
                'duration': self.simulation.duration,
                'development_index': {
                    'initial': self.simulation.initial_state['development_index'],
                    'final': self.simulation.final_state['development_index'],
                    'change': self.simulation.final_state['development_index'] - 
                             self.simulation.initial_state['development_index']
                },
                'sustainability_index': {
                    'initial': self.simulation.initial_state['sustainability_index'],
                    'final': self.simulation.final_state['sustainability_index'],
                    'change': self.simulation.final_state['sustainability_index'] - 
                             self.simulation.initial_state['sustainability_index']
                },
                'resilience_index': {
                    'initial': self.simulation.initial_state['resilience_index'],
                    'final': self.simulation.final_state['resilience_index'],
                    'change': self.simulation.final_state['resilience_index'] - 
                             self.simulation.initial_state['resilience_index']
                },
                'wellbeing_index': {
                    'initial': self.simulation.initial_state['wellbeing_index'],
                    'final': self.simulation.final_state['wellbeing_index'],
                    'change': self.simulation.final_state['wellbeing_index'] - 
                             self.simulation.initial_state['wellbeing_index']
                }
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _prepare_detailed_data(self) -> Dict[str, Any]:
        """Prepare data for the detailed report."""
        return {
            'details': {
                'economic': {
                    'gdp_growth': self.simulation.final_state['gdp_growth'],
                    'sector_contributions': self.simulation.final_state['sector_contributions'],
                    'employment': {
                        'initial': self.simulation.initial_state['employment_rate'],
                        'final': self.simulation.final_state['employment_rate'],
                        'change': self.simulation.final_state['employment_rate'] - 
                                 self.simulation.initial_state['employment_rate']
                    }
                },
                'environmental': {
                    'climate': {
                        'temperature': self.simulation.final_state['temperature_change'],
                        'precipitation': self.simulation.final_state['precipitation_change'],
                        'sea_level': self.simulation.final_state['sea_level_rise']
                    },
                    'health': {
                        'flood_risk': self.simulation.final_state['flood_risk_change'],
                        'crop_yield': self.simulation.final_state['crop_yield_change'],
                        'water_stress': self.simulation.final_state['water_stress_change']
                    }
                },
                'demographic': {
                    'population': {
                        'initial': self.simulation.initial_state['population'],
                        'final': self.simulation.final_state['population'],
                        'growth_rate': self.simulation.final_state['population_growth_rate']
                    },
                    'social': {
                        'hdi': self.simulation.final_state['hdi_change'],
                        'poverty': self.simulation.final_state['poverty_rate_change'],
                        'inequality': self.simulation.final_state['inequality_change']
                    }
                },
                'infrastructure': {
                    'physical': {
                        'road_length': self.simulation.final_state['road_length_change'],
                        'paved_ratio': self.simulation.final_state['paved_ratio_change'],
                        'electricity': self.simulation.final_state['electricity_access_change']
                    },
                    'quality': {
                        'infrastructure_quality': self.simulation.final_state['infrastructure_quality_change'],
                        'connectivity': self.simulation.final_state['connectivity_change'],
                        'resilience': self.simulation.final_state['infrastructure_resilience_change']
                    }
                },
                'governance': {
                    'institutional': {
                        'bureaucracy': self.simulation.final_state['bureaucracy_efficiency_change'],
                        'corruption': self.simulation.final_state['corruption_index_change'],
                        'transparency': self.simulation.final_state['transparency_change']
                    },
                    'policy': {
                        'effectiveness': self.simulation.final_state['policy_effectiveness_change'],
                        'implementation': self.simulation.final_state['policy_implementation_change'],
                        'coordination': self.simulation.final_state['policy_coordination_change']
                    }
                }
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _prepare_json_data(self) -> Dict[str, Any]:
        """Prepare data for the JSON report."""
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': self.simulation.version,
                'simulation_duration': {
                    'start_year': self.simulation.start_year,
                    'end_year': self.simulation.end_year,
                    'total_years': self.simulation.duration
                }
            },
            'indices': {
                'development': {
                    'initial': self.simulation.initial_state['development_index'],
                    'final': self.simulation.final_state['development_index'],
                    'change': self.simulation.final_state['development_index'] - 
                             self.simulation.initial_state['development_index']
                },
                'sustainability': {
                    'initial': self.simulation.initial_state['sustainability_index'],
                    'final': self.simulation.final_state['sustainability_index'],
                    'change': self.simulation.final_state['sustainability_index'] - 
                             self.simulation.initial_state['sustainability_index']
                },
                'resilience': {
                    'initial': self.simulation.initial_state['resilience_index'],
                    'final': self.simulation.final_state['resilience_index'],
                    'change': self.simulation.final_state['resilience_index'] - 
                             self.simulation.initial_state['resilience_index']
                },
                'wellbeing': {
                    'initial': self.simulation.initial_state['wellbeing_index'],
                    'final': self.simulation.final_state['wellbeing_index'],
                    'change': self.simulation.final_state['wellbeing_index'] - 
                             self.simulation.initial_state['wellbeing_index']
                }
            },
            **self._prepare_detailed_data()['details']
        }
    
    def generate_summary_report(self) -> None:
        """Generate the summary HTML report."""
        try:
            template = self.env.get_template('summary_report.html')
            data = self._prepare_summary_data()
            output = template.render(**data)
            
            output_path = self.report_dir / f'summary_report_{self.simulation_id}.html'
            with open(output_path, 'w') as f:
                f.write(output)
            
            logger.info(f"Summary report generated at {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            raise
    
    def generate_detailed_report(self) -> None:
        """Generate the detailed HTML report."""
        try:
            template = self.env.get_template('detailed_report.html')
            data = self._prepare_detailed_data()
            output = template.render(**data)
            
            output_path = self.report_dir / f'detailed_report_{self.simulation_id}.html'
            with open(output_path, 'w') as f:
                f.write(output)
            
            logger.info(f"Detailed report generated at {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {str(e)}")
            raise
    
    def generate_json_report(self) -> None:
        """Generate the JSON report."""
        try:
            data = self._prepare_json_data()
            
            # Validate against schema
            jsonschema.validate(instance=data, schema=self.schema)
            
            output_path = self.report_dir / f'simulation_report_{self.simulation_id}.json'
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"JSON report generated at {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            raise 