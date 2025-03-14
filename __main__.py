"""
Main entry point for the Bangladesh Development Simulation Model.
"""

import sys
import logging
from pathlib import Path
import yaml

from .models.simulation import BangladeshSimulation
from .visualization.plotter import Plotter
from .visualization.reports import ReportGenerator

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get the package directory
        package_dir = Path(__file__).parent
        config_path = package_dir / 'config' / 'simulation_config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main entry point for the simulation."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize simulation
        simulation = BangladeshSimulation(config)
        logger.info("Simulation initialized")
        
        # Run simulation
        simulation.run()
        logger.info("Simulation completed")
        
        # Generate visualizations
        plotter = Plotter()
        plotter.plot_development_trajectory(simulation.history)
        plotter.plot_sector_performance(simulation.history)
        plotter.plot_environmental_impact(simulation.history)
        plotter.plot_demographic_indicators(simulation.history)
        plotter.plot_infrastructure_development(simulation.history)
        plotter.plot_governance_indicators(simulation.history)
        plotter.plot_correlation_matrix(simulation.history)
        logger.info("Visualizations generated")
        
        # Generate reports
        report_generator = ReportGenerator(simulation.history)
        report_generator.generate_summary_report()
        report_generator.generate_detailed_report()
        report_generator.generate_json_report()
        logger.info("Reports generated")
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 