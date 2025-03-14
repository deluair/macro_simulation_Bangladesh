#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify that the Bangladesh simulation is working properly.
"""

import yaml
from models.simulation import BangladeshSimulation

def main():
    """Run a simple test of the simulation."""
    print("Loading configuration...")
    with open('config/simulation_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Initializing simulation...")
    simulation = BangladeshSimulation(config)
    
    print("Running one simulation step...")
    simulation.step()
    
    print("\nSimulation state after one step:")
    print(f"Development Index: {simulation.state.get('development_index', 'N/A')}")
    print(f"Sustainability Index: {simulation.state.get('sustainability_index', 'N/A')}")
    print(f"Resilience Index: {simulation.state.get('resilience_index', 'N/A')}")
    print(f"Wellbeing Index: {simulation.state.get('wellbeing_index', 'N/A')}")
    
    print("\nModel States:")
    for model_name, model in simulation.models.items():
        print(f"\n{model_name.capitalize()} Model State:")
        for key, value in model.state.items():
            if key != 'timestamp':
                print(f"  {key}: {value}")
    
    print("\nSimulation step completed successfully!")

if __name__ == "__main__":
    main() 