#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main simulation runner for the Bangladesh Integrated Socioeconomic and Environmental System Model.
This file orchestrates the simulation by initializing all system components, running the simulation
steps, and producing the final outputs and visualizations.
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional

# Import model components
from models.economic.economy import EconomicSystem
from models.environmental.environment import EnvironmentalSystem
from models.demographic.population import DemographicSystem
from models.infrastructure.infrastructure import InfrastructureSystem
from models.governance.governance import GovernanceSystem

# Import utilities
from utils.data_loader import DataLoader
from utils.config_manager import ConfigManager
from utils.result_processor import ResultProcessor
from utils.validation import ValidationMetrics
from utils.monte_carlo import MonteCarloSimulator
from utils.system_integrator import SystemIntegrator
from utils.visualizer import Visualizer

# Import visualization tools
from visualization.dashboard import Dashboard
from visualization.geo_visualizer import GeoVisualizer
from visualization.network_visualizer import NetworkVisualizer
from visualization.time_series_plotter import TimeSeriesPlotter


class TimeStepUnit(Enum):
    """Enumeration of supported time step units."""
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TimeConfiguration:
    """Configuration for simulation time steps and scheduling."""
    
    def __init__(self, start_date: datetime, time_step_unit: TimeStepUnit, 
                time_step_size: int = 1, end_date: Optional[datetime] = None, 
                num_steps: Optional[int] = None):
        """
        Initialize time configuration.
        
        Args:
            start_date: Starting date of the simulation
            time_step_unit: Unit for time steps (day, month, quarter, year)
            time_step_size: Number of units per step
            end_date: End date of simulation (either this or num_steps must be provided)
            num_steps: Number of steps to run (either this or end_date must be provided)
        """
        self.start_date = start_date
        self.time_step_unit = time_step_unit
        self.time_step_size = time_step_size
        self.end_date = end_date
        self.num_steps = num_steps
        
        # Validate configuration
        if end_date is None and num_steps is None:
            raise ValueError("Either end_date or num_steps must be provided")
            
        # Calculate derived properties
        if end_date is not None:
            # Calculate number of steps from date range
            self.num_steps = self._calculate_num_steps()
        else:
            # Calculate end date from number of steps
            self.end_date = self._calculate_end_date()
            
    def _calculate_num_steps(self) -> int:
        """Calculate number of steps between start and end date."""
        if self.time_step_unit == TimeStepUnit.DAY:
            delta = (self.end_date - self.start_date).days
            return delta // self.time_step_size
            
        elif self.time_step_unit == TimeStepUnit.MONTH:
            months_diff = (self.end_date.year - self.start_date.year) * 12 + \
                         (self.end_date.month - self.start_date.month)
            return months_diff // self.time_step_size
            
        elif self.time_step_unit == TimeStepUnit.QUARTER:
            months_diff = (self.end_date.year - self.start_date.year) * 12 + \
                         (self.end_date.month - self.start_date.month)
            quarters_diff = months_diff // 3
            return quarters_diff // self.time_step_size
            
        elif self.time_step_unit == TimeStepUnit.YEAR:
            years_diff = self.end_date.year - self.start_date.year
            return years_diff // self.time_step_size
            
    def _calculate_end_date(self) -> datetime:
        """Calculate end date based on number of steps."""
        if self.time_step_unit == TimeStepUnit.DAY:
            return self.start_date + pd.DateOffset(days=self.num_steps * self.time_step_size)
            
        elif self.time_step_unit == TimeStepUnit.MONTH:
            return self.start_date + pd.DateOffset(months=self.num_steps * self.time_step_size)
            
        elif self.time_step_unit == TimeStepUnit.QUARTER:
            return self.start_date + pd.DateOffset(months=self.num_steps * self.time_step_size * 3)
            
        elif self.time_step_unit == TimeStepUnit.YEAR:
            return self.start_date + pd.DateOffset(years=self.num_steps * self.time_step_size)
            
    def get_date_for_step(self, step: int) -> datetime:
        """Get the date for a specific step number."""
        if self.time_step_unit == TimeStepUnit.DAY:
            return self.start_date + pd.DateOffset(days=step * self.time_step_size)
            
        elif self.time_step_unit == TimeStepUnit.MONTH:
            return self.start_date + pd.DateOffset(months=step * self.time_step_size)
            
        elif self.time_step_unit == TimeStepUnit.QUARTER:
            return self.start_date + pd.DateOffset(months=step * self.time_step_size * 3)
            
        elif self.time_step_unit == TimeStepUnit.YEAR:
            return self.start_date + pd.DateOffset(years=step * self.time_step_size)
            
    def get_all_dates(self) -> List[datetime]:
        """Get a list of all dates in the simulation."""
        return [self.get_date_for_step(i) for i in range(self.num_steps + 1)]
    
    def get_formatted_date(self, step: int) -> str:
        """Get a formatted date string for a specific step."""
        date = self.get_date_for_step(step)
        
        if self.time_step_unit == TimeStepUnit.DAY:
            return date.strftime("%Y-%m-%d")
            
        elif self.time_step_unit == TimeStepUnit.MONTH:
            return date.strftime("%Y-%m")
            
        elif self.time_step_unit == TimeStepUnit.QUARTER:
            quarter = (date.month - 1) // 3 + 1
            return f"{date.year}-Q{quarter}"
            
        elif self.time_step_unit == TimeStepUnit.YEAR:
            return str(date.year)


class SystemScheduler:
    """Scheduler for controlling the execution of different systems at different time steps."""
    
    def __init__(self, time_config: TimeConfiguration):
        """
        Initialize the system scheduler.
        
        Args:
            time_config: Time step configuration
        """
        self.time_config = time_config
        self.schedules = {}
        
    def add_system_schedule(self, system_name: str, 
                           frequency: int = 1, 
                           offset: int = 0) -> None:
        """
        Add a schedule for a system component.
        
        Args:
            system_name: Name of the system
            frequency: How often the system should be updated (every N steps)
            offset: Step offset for execution
        """
        self.schedules[system_name] = {
            'frequency': frequency,
            'offset': offset,
            'last_execution': None
        }
        
    def should_execute(self, system_name: str, current_step: int) -> bool:
        """
        Check if a system should be executed at the current step.
        
        Args:
            system_name: Name of the system
            current_step: Current simulation step
            
        Returns:
            Whether the system should be executed
        """
        if system_name not in self.schedules:
            # Default: execute every step
            return True
            
        schedule = self.schedules[system_name]
        
        # Check if this step is scheduled
        should_run = (current_step - schedule['offset']) % schedule['frequency'] == 0
        
        if should_run:
            # Update last execution time
            schedule['last_execution'] = current_step
            
        return should_run
    
    def get_step_delta(self, system_name: str, current_step: int) -> int:
        """
        Calculate the number of time steps since last execution.
        
        Args:
            system_name: Name of the system
            current_step: Current simulation step
            
        Returns:
            Number of steps since last execution
        """
        if system_name not in self.schedules:
            return 1
            
        schedule = self.schedules[system_name]
        last_exec = schedule['last_execution']
        
        if last_exec is None:
            # First execution
            if current_step % schedule['frequency'] == schedule['offset']:
                return schedule['frequency']
            else:
                return current_step - schedule['offset']
        else:
            return current_step - last_exec


class BangladeshSimulation:
    """
    Main simulation class that coordinates all components of the Bangladesh integrated system model.
    """
    
    def __init__(self, config_path_or_dict='config/config.yaml'):
        """
        Initialize the simulation with configuration settings.
        
        Args:
            config_path_or_dict (str or dict): Path to the main configuration file or config dictionary
        """
        print(f"Initializing Bangladesh Integrated System Simulation...")
        self.start_time = time.time()
        
        # Load configuration
        if isinstance(config_path_or_dict, dict):
            # Config is already loaded as a dictionary
            self.config = config_path_or_dict
            self.config_manager = None
        else:
            # Config is a file path
            self.config_manager = ConfigManager(config_path_or_dict)
            self.config = self.config_manager.config
        
        # Set random seed for reproducibility
        np.random.seed(self.config.get('simulation', {}).get('random_seed', 42))
        
        # Initialize time configuration
        self._init_time_config()
        
        # Initialize system scheduler
        self._init_scheduler()
        
        # Initialize data loader
        data_dir = self.config.get('data', {}).get('data_dir', 'data/')
        self.data_loader = DataLoader(data_dir)
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Setup result processor
        output_dir = self.config.get('output', {}).get('output_dir', 'results/')
        self.result_processor = ResultProcessor(
            output_dir=output_dir,
            simulation_id=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        # Initialize Monte Carlo simulator if enabled
        monte_carlo_config = self.config.get('simulation', {}).get('monte_carlo', {})
        if monte_carlo_config.get('enabled', False):
            self.monte_carlo = MonteCarloSimulator(
                self,
                n_runs=monte_carlo_config.get('n_runs', 100),
                parameters=monte_carlo_config.get('parameters', [])
            )
        else:
            self.monte_carlo = None
        
        # Initialize additional attributes
        self.current_step = 0
        self.current_date = self.time_config.start_date
        self.history = {
            'economic': [],
            'environmental': [],
            'demographic': [],
            'infrastructure': [],
            'governance': []
        }
        
        print(f"Simulation initialized in {time.time() - self.start_time:.2f} seconds")
    
    def _init_time_config(self):
        """Initialize time configuration from config."""
        # Check if we have the time configuration section
        if 'time' in self.config.get('simulation', {}):
            # New format with detailed time configuration
            time_config = self.config['simulation']['time']
            
            # Parse start date
            start_date = datetime.strptime(time_config['start_date'], "%Y-%m-%d")
            
            # Parse time step unit
            time_step_unit = TimeStepUnit(time_config['time_step_unit'])
            
            # Get time step size
            time_step_size = time_config.get('time_step_size', 1)
            
            # Parse end date or num steps
            end_date = None
            num_steps = None
            
            if 'end_date' in time_config:
                end_date = datetime.strptime(time_config['end_date'], "%Y-%m-%d")
            elif 'num_steps' in time_config:
                num_steps = time_config['num_steps']
            else:
                # Default to running for 20 years
                num_steps = 20
        else:
            # Old format with just start_year and end_year
            # Provide defaults if fields are missing
            start_year = self.config.get('simulation', {}).get('start_year', 2023)
            end_year = self.config.get('simulation', {}).get('end_year', 2043)
            
            # Convert years to dates
            start_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year, 1, 1)
            
            # Default to years
            time_step_unit = TimeStepUnit.YEAR
            time_step_size = 1
            num_steps = None
            
        # Create time configuration
        self.time_config = TimeConfiguration(
            start_date=start_date,
            time_step_unit=time_step_unit,
            time_step_size=time_step_size,
            end_date=end_date,
            num_steps=num_steps
        )
        
    def _init_scheduler(self):
        """Initialize system scheduler with component execution frequencies."""
        self.scheduler = SystemScheduler(self.time_config)
        
        # Set up schedules based on config
        schedules = self.config.get('simulation', {}).get('schedules', {})
        
        for system_name, schedule in schedules.items():
            frequency = schedule.get('frequency', 1)
            offset = schedule.get('offset', 0)
            self.scheduler.add_system_schedule(system_name, frequency, offset)
            
    def _init_subsystems(self):
        """Initialize all subsystems of the simulation."""
        print("Initializing subsystems...")
        
        # Initialize economic system
        self.economic_system = EconomicSystem(
            config=self.config.get('economic', {}),
            data_loader=self.data_loader
        )
        
        # Initialize environmental system
        self.environmental_system = EnvironmentalSystem(
            config=self.config.get('environmental', {}),
            data_loader=self.data_loader
        )
        
        # Initialize demographic system
        self.demographic_system = DemographicSystem(
            config=self.config.get('demographic', {}),
            data_loader=self.data_loader
        )
        
        # Initialize infrastructure system
        self.infrastructure_system = InfrastructureSystem(
            config=self.config.get('infrastructure', {}),
            data_loader=self.data_loader
        )
        
        # Initialize governance system
        self.governance_system = GovernanceSystem(
            config=self.config.get('governance', {}),
            data_loader=self.data_loader
        )
        
        # Initialize system integrator
        self.system_integrator = SystemIntegrator(
            config=self.config
        )
        
        print("All subsystems initialized.")
    
    def step(self):
        """Execute one simulation step, updating subsystems according to their schedules."""
        # Get current date
        self.current_date = self.time_config.get_date_for_step(self.current_step)
        formatted_date = self.time_config.get_formatted_date(self.current_step)
        
        # Log the current step
        print(f"Executing simulation step {self.current_step} for {formatted_date}")
        
        # Define the order of system updates
        system_order = [
            ('environmental', self.environmental_system),
            ('demographic', self.demographic_system),
            ('infrastructure', self.infrastructure_system),
            ('economic', self.economic_system),
            ('governance', self.governance_system)
        ]
        
        # Store system states for this step
        step_results = {
            'step': self.current_step,
            'date': formatted_date,
            'systems': {}
        }
        
        # Update each system according to its schedule
        for system_name, system in system_order:
            # Check if this system should be executed at this step
            if self.scheduler.should_execute(system_name, self.current_step):
                # Get the number of steps since last execution
                step_delta = self.scheduler.get_step_delta(system_name, self.current_step)
                
                # Execute the system with the appropriate time delta
                system_results = self._execute_system(
                    system_name, system, step_delta, formatted_date
                )
                
                # Store results
                self.history[system_name].append(system_results)
                step_results['systems'][system_name] = system_results
        
        # Apply cross-system effects if the method exists
        if hasattr(self.system_integrator, 'apply_cross_system_effects'):
            self.system_integrator.apply_cross_system_effects(self)
        
        # Increment step counter
        self.current_step += 1
        
        return step_results
        
    def _execute_system(self, system_name: str, system: Any, 
                       step_delta: int, current_date: str) -> Dict[str, Any]:
        """
        Execute a specific system with the appropriate inputs.
        
        Args:
            system_name: Name of the system
            system: System object
            step_delta: Number of time steps to simulate
            current_date: Current date in formatted string
            
        Returns:
            System outputs and state
        """
        try:
            # Get the latest state of other systems to pass as context
            context = {
                'environmental_system': self.environmental_system,
                'demographic_system': self.demographic_system,
                'infrastructure_system': self.infrastructure_system,
                'economic_system': self.economic_system,
                'governance_system': self.governance_system
            }
            
            # Remove self from context
            context.pop(f"{system_name}_system", None)
            
            # Extract the year from the current_date
            year = int(current_date.split('-')[0])
            
            # Execute the system step - adapt to the existing API
            # Check if the system accepts the new parameters
            if hasattr(system, 'step_with_date'):
                result = system.step_with_date(
                    date=current_date,
                    step_delta=step_delta,
                    **context
                )
            else:
                # Fall back to the old API, but pass the year parameter
                result = system.step(year=year)
            
            return result
            
        except Exception as e:
            print(f"Error executing {system_name} system: {e}")
            # Return empty result to avoid breaking the simulation
            return {'error': str(e)}
    
    def run(self, num_steps: int = None):
        """
        Run the simulation for the specified number of steps or until completion.
        
        Args:
            num_steps: Number of steps to run (default: run all)
        """
        total_steps = num_steps or self.time_config.num_steps
        start_time = time.time()
        
        print(f"Starting simulation for {total_steps} steps...")
        print(f"  Start date: {self.time_config.start_date}")
        print(f"  End date: {self.time_config.end_date}")
        print(f"  Time step: {self.time_config.time_step_size} {self.time_config.time_step_unit.value}")
        
        results = []
        
        for _ in range(total_steps):
            step_result = self.step()
            results.append(step_result)
            
            # Report progress at intervals
            if (self.current_step % 10 == 0) or (self.current_step == total_steps):
                elapsed = time.time() - start_time
                steps_per_second = self.current_step / elapsed if elapsed > 0 else 0
                
                print(f"  Completed step {self.current_step}/{total_steps} "
                     f"({self.current_step/total_steps*100:.1f}%) - "
                     f"{steps_per_second:.2f} steps/sec")
        
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
        
        # Process and return results
        return self.result_processor.process_results(results)
    
    def run_with_policy(self, policy_scenario):
        """
        Run the simulation with a specific policy scenario.
        
        Args:
            policy_scenario: Policy scenario configuration
            
        Returns:
            Simulation results
        """
        # Reset simulation state
        self.reset()
        
        # Apply policy interventions
        self._apply_policy_scenario(policy_scenario)
        
        # Run the simulation
        return self.run()
    
    def _apply_policy_scenario(self, policy_scenario):
        """
        Apply a policy scenario to the simulation.
        
        Args:
            policy_scenario: Policy scenario configuration
        """
        print(f"Applying policy scenario: {policy_scenario.name}")
        
        for intervention in policy_scenario.interventions:
            print(f"  Applying intervention: {intervention.name}")
            
            # Determine target system
            target_system = getattr(self, f"{intervention.target_system}_system", None)
            
            if target_system is None:
                print(f"    Warning: Unknown target system '{intervention.target_system}'")
                continue
                
            # Check if intervention is active from the start
            if intervention.start_year <= self.time_config.start_date.year:
                # Apply intervention parameters
                target_system.apply_policy_intervention(intervention)
    
    def reset(self):
        """Reset simulation to initial state for a new run."""
        print("Resetting simulation to initial state...")
        
        # Reset step counter and date
        self.current_step = 0
        self.current_date = self.time_config.start_date
        
        # Reset history
        self.history = {
            'economic': [],
            'environmental': [],
            'demographic': [],
            'infrastructure': [],
            'governance': []
        }
        
        # Reset all subsystems
        self.economic_system.reset()
        self.environmental_system.reset()
        self.demographic_system.reset()
        self.infrastructure_system.reset()
        self.governance_system.reset()
        
        # Reset scheduler execution history
        for schedule in self.scheduler.schedules.values():
            schedule['last_execution'] = None
    
    def get_history(self):
        """Get the simulation history."""
        return self.history
    
    def get_current_state(self):
        """Get the current state of all systems."""
        return {
            'step': self.current_step,
            'date': self.time_config.get_formatted_date(self.current_step),
            'economic': self.economic_system.get_state(),
            'environmental': self.environmental_system.get_state(),
            'demographic': self.demographic_system.get_state(),
            'infrastructure': self.infrastructure_system.get_state(),
            'governance': self.governance_system.get_state()
        }