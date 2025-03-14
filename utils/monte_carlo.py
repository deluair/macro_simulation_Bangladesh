#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monte Carlo simulation utility for the Bangladesh simulation model.
This module allows for running multiple simulations with varying parameters
to analyze sensitivity and uncertainty.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt


class MonteCarloSimulator:
    """
    Utility for running Monte Carlo simulations by varying model parameters
    to analyze sensitivity and uncertainty in the model outcomes.
    """
    
    def __init__(self, simulation, n_runs=100, parameters=None, output_dir=None, 
                parallel=True, n_processes=None):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            simulation: The main simulation object to run
            n_runs (int): Number of simulation runs
            parameters (list): List of parameter dictionaries to vary
                Each parameter dict should have: name, min, max, (optional: distribution)
            output_dir (str, optional): Directory to save MC results
            parallel (bool): Whether to run simulations in parallel
            n_processes (int, optional): Number of processes for parallel execution
        """
        self.simulation = simulation
        self.n_runs = n_runs
        self.parameters = parameters or []
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            main_output_dir = Path(simulation.config['output']['output_dir'])
            self.output_dir = main_output_dir / 'monte_carlo'
        
        if not self.output_dir.exists():
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Created Monte Carlo output directory: {self.output_dir}")
        
        # Parallel processing settings
        self.parallel = parallel
        if n_processes is None:
            self.n_processes = mp.cpu_count() - 1
        else:
            self.n_processes = n_processes
        
        # Results storage
        self.results = {}
        self.parameter_samples = []
        
        print(f"Initialized Monte Carlo simulator with {n_runs} runs for {len(parameters)} parameters")
    
    def run(self):
        """
        Run Monte Carlo simulations using the initialized parameters.
        
        Returns:
            dict: Summary of Monte Carlo simulation results
        """
        print(f"Starting Monte Carlo simulation with {self.n_runs} runs...")
        start_time = time.time()
        
        # Generate parameter samples
        self._generate_parameter_samples()
        
        # Run simulations based on parameter samples
        if self.parallel and self.n_runs > 1:
            self._run_parallel()
        else:
            self._run_sequential()
        
        # Process results
        self._process_results()
        
        elapsed_time = time.time() - start_time
        print(f"Monte Carlo simulation completed in {elapsed_time:.2f} seconds")
        
        return self.get_summary()
    
    def _generate_parameter_samples(self):
        """Generate parameter value samples for all simulation runs."""
        
        self.parameter_samples = []
        
        for i in range(self.n_runs):
            # Generate parameters for this run
            run_params = {}
            
            for param in self.parameters:
                param_name = param['name']
                min_val = param['min']
                max_val = param['max']
                
                # Generate value from specified distribution
                distribution = param.get('distribution', 'uniform')
                
                if distribution == 'uniform':
                    value = np.random.uniform(min_val, max_val)
                elif distribution == 'normal':
                    # For normal, min and max are treated as mean and standard deviation
                    value = np.random.normal(min_val, max_val)
                elif distribution == 'triangular':
                    # For triangular, we need mode (most likely) value
                    mode = param.get('mode', (min_val + max_val) / 2)
                    value = np.random.triangular(min_val, mode, max_val)
                elif distribution == 'lognormal':
                    # For lognormal, min and max are treated as mean and standard deviation of log values
                    value = np.random.lognormal(min_val, max_val)
                else:
                    # Default to uniform
                    value = np.random.uniform(min_val, max_val)
                
                # Store parameter value
                run_params[param_name] = value
            
            # Add to parameter samples
            self.parameter_samples.append(run_params)
    
    def _run_sequential(self):
        """Run simulations sequentially."""
        
        self.results = {
            'runs': [],
            'parameters': self.parameter_samples.copy()
        }
        
        for i, params in enumerate(self.parameter_samples):
            print(f"Running simulation {i+1}/{self.n_runs}...")
            
            # Apply parameters to a cloned simulation
            sim_clone = self._prepare_simulation(params)
            
            # Run the simulation
            sim_clone.run()
            
            # Store results
            self.results['runs'].append(sim_clone.results)
    
    def _run_parallel(self):
        """Run simulations in parallel using multiprocessing."""
        
        # Prepare arguments for each simulation run
        args_list = [(i, params) for i, params in enumerate(self.parameter_samples)]
        
        # Create a pool of workers
        with mp.Pool(processes=self.n_processes) as pool:
            # Map the _run_simulation function to the arguments
            run_results = pool.map(self._run_simulation_wrapper, args_list)
        
        # Aggregate results
        self.results = {
            'runs': [result for result in run_results],
            'parameters': self.parameter_samples.copy()
        }
    
    def _run_simulation_wrapper(self, args):
        """Wrapper for running a single simulation in parallel."""
        i, params = args
        
        print(f"Running simulation {i+1}/{self.n_runs} in parallel process...")
        
        # Apply parameters to a cloned simulation
        sim_clone = self._prepare_simulation(params)
        
        # Run the simulation
        sim_clone.run()
        
        # Return results
        return sim_clone.results
    
    def _prepare_simulation(self, parameters):
        """
        Prepare a simulation with specified parameters.
        
        Args:
            parameters (dict): Parameter values to apply
            
        Returns:
            object: Configured simulation object
        """
        # Create a deep copy of the original simulation
        sim_clone = deepcopy(self.simulation)
        
        # Apply parameter values
        for param_name, value in parameters.items():
            # Parse parameter path (e.g., 'economic.growth_volatility')
            path_parts = param_name.split('.')
            
            # Navigate config structure to set parameter
            config_section = sim_clone.config
            for part in path_parts[:-1]:
                if part not in config_section:
                    config_section[part] = {}
                config_section = config_section[part]
            
            # Set the parameter value
            config_section[path_parts[-1]] = value
        
        # Reinitialize subsystems with updated config
        sim_clone._init_subsystems()
        
        return sim_clone
    
    def _process_results(self):
        """Process simulation results to analyze uncertainty and sensitivity."""
        
        # Create metrics for each run
        metrics = []
        
        for i, run_results in enumerate(self.results['runs']):
            # Extract key metrics from each system
            run_metrics = {}
            
            # Economic metrics
            if 'economic' in run_results and run_results['economic']:
                last_econ = run_results['economic'][-1]
                if 'gdp' in last_econ:
                    run_metrics['final_gdp'] = last_econ['gdp']
                if 'gdp_growth' in last_econ:
                    run_metrics['final_gdp_growth'] = last_econ['gdp_growth']
                if 'inflation_rate' in last_econ:
                    run_metrics['final_inflation'] = last_econ['inflation_rate']
            
            # Demographic metrics
            if 'demographic' in run_results and run_results['demographic']:
                last_demo = run_results['demographic'][-1]
                if 'total_population' in last_demo:
                    run_metrics['final_population'] = last_demo['total_population']
                if 'urbanization_rate' in last_demo:
                    run_metrics['final_urbanization'] = last_demo['urbanization_rate']
            
            # Environmental metrics
            if 'environmental' in run_results and run_results['environmental']:
                last_env = run_results['environmental'][-1]
                if 'forest_cover' in last_env:
                    run_metrics['final_forest_cover'] = last_env['forest_cover']
                if 'annual_emissions' in last_env:
                    run_metrics['final_emissions'] = last_env['annual_emissions']
            
            # Infrastructure metrics
            if 'infrastructure' in run_results and run_results['infrastructure']:
                last_infra = run_results['infrastructure'][-1]
                if 'electricity_coverage' in last_infra:
                    run_metrics['final_electricity_coverage'] = last_infra['electricity_coverage']
                if 'road_density' in last_infra:
                    run_metrics['final_road_density'] = last_infra['road_density']
            
            # Add parameter values for this run
            run_params = self.parameter_samples[i]
            for param_name, value in run_params.items():
                # Simplify parameter name for readability
                simple_name = param_name.split('.')[-1]
                run_metrics[f'param_{simple_name}'] = value
            
            metrics.append(run_metrics)
        
        # Convert to DataFrame for analysis
        self.metrics_df = pd.DataFrame(metrics)
        
        # Save metrics to CSV
        self.metrics_df.to_csv(self.output_dir / 'monte_carlo_metrics.csv', index=False)
        
        # Calculate statistics for all metrics
        self.metric_stats = {}
        for column in self.metrics_df.columns:
            if column.startswith('final_'):
                self.metric_stats[column] = {
                    'mean': self.metrics_df[column].mean(),
                    'std': self.metrics_df[column].std(),
                    'min': self.metrics_df[column].min(),
                    'max': self.metrics_df[column].max(),
                    'median': self.metrics_df[column].median(),
                    '5th_percentile': self.metrics_df[column].quantile(0.05),
                    '95th_percentile': self.metrics_df[column].quantile(0.95)
                }
        
        # Calculate correlations between parameters and outcome metrics
        self.correlations = {}
        param_columns = [col for col in self.metrics_df.columns if col.startswith('param_')]
        outcome_columns = [col for col in self.metrics_df.columns if col.startswith('final_')]
        
        for outcome in outcome_columns:
            self.correlations[outcome] = {}
            for param in param_columns:
                corr = self.metrics_df[[param, outcome]].corr().iloc[0, 1]
                self.correlations[outcome][param] = corr
    
    def get_summary(self):
        """
        Get a summary of Monte Carlo simulation results.
        
        Returns:
            dict: Summary statistics and sensitivity information
        """
        if not hasattr(self, 'metric_stats'):
            raise ValueError("Results must be processed before getting summary")
            
        return {
            'n_runs': self.n_runs,
            'parameters': self.parameters,
            'metric_stats': self.metric_stats,
            'correlations': self.correlations
        }
    
    def generate_plots(self, save_format='png'):
        """
        Generate plots to visualize Monte Carlo simulation results.
        
        Args:
            save_format (str): Format to save plots (png, jpg, pdf, svg)
            
        Returns:
            list: Paths to saved plot files
        """
        if not hasattr(self, 'metrics_df'):
            raise ValueError("Results must be processed before generating plots")
        
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        if not plots_dir.exists():
            os.makedirs(plots_dir, exist_ok=True)
            
        saved_plots = []
        
        # 1. Histograms of key outcome metrics
        outcome_columns = [col for col in self.metrics_df.columns if col.startswith('final_')]
        
        for outcome in outcome_columns:
            plt.figure(figsize=(10, 6))
            plt.hist(self.metrics_df[outcome], bins=20, alpha=0.7, color='blue')
            
            # Add mean and percentile lines
            mean_val = self.metric_stats[outcome]['mean']
            p05_val = self.metric_stats[outcome]['5th_percentile']
            p95_val = self.metric_stats[outcome]['95th_percentile']
            
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            plt.axvline(p05_val, color='green', linestyle='dotted', linewidth=2, label=f'5th %: {p05_val:.2f}')
            plt.axvline(p95_val, color='green', linestyle='dotted', linewidth=2, label=f'95th %: {p95_val:.2f}')
            
            # Format the plot
            title = outcome.replace('final_', '').replace('_', ' ').title()
            plt.title(f"Distribution of {title}")
            plt.xlabel(title)
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plot_path = plots_dir / f"hist_{outcome}.{save_format}"
            plt.savefig(plot_path)
            plt.close()
            
            saved_plots.append(str(plot_path))
        
        # 2. Scatter plots of key parameters vs outcomes
        param_columns = [col for col in self.metrics_df.columns if col.startswith('param_')]
        
        # For each outcome, plot against top 3 most correlated parameters
        for outcome in outcome_columns:
            # Sort parameters by correlation strength
            sorted_params = sorted(self.correlations[outcome].items(), 
                                 key=lambda x: abs(x[1]), reverse=True)
            
            # Take top 3 parameters (or fewer if less available)
            top_params = sorted_params[:min(3, len(sorted_params))]
            
            for param, corr in top_params:
                plt.figure(figsize=(10, 6))
                
                # Create scatter plot
                plt.scatter(self.metrics_df[param], self.metrics_df[outcome], 
                           alpha=0.7, color='blue')
                
                # Add trend line
                z = np.polyfit(self.metrics_df[param], self.metrics_df[outcome], 1)
                p = np.poly1d(z)
                plt.plot(self.metrics_df[param], p(self.metrics_df[param]), 
                        'r--', linewidth=2, label=f'Correlation: {corr:.2f}')
                
                # Format the plot
                param_title = param.replace('param_', '').replace('_', ' ').title()
                outcome_title = outcome.replace('final_', '').replace('_', ' ').title()
                
                plt.title(f"{outcome_title} vs {param_title}")
                plt.xlabel(param_title)
                plt.ylabel(outcome_title)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                plot_path = plots_dir / f"scatter_{outcome}_vs_{param}.{save_format}"
                plt.savefig(plot_path)
                plt.close()
                
                saved_plots.append(str(plot_path))
        
        # 3. Correlation heatmap
        plt.figure(figsize=(12, 10))
        
        # Create correlation matrix
        corr_matrix = pd.DataFrame(index=outcome_columns, columns=param_columns)
        for outcome in outcome_columns:
            for param in param_columns:
                corr_matrix.loc[outcome, param] = self.correlations[outcome][param]
        
        # Plot heatmap
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(label='Correlation')
        
        # Format axis labels
        plt.xticks(range(len(param_columns)), 
                  [col.replace('param_', '').replace('_', ' ').title() for col in param_columns], 
                  rotation=45, ha='right')
        plt.yticks(range(len(outcome_columns)), 
                  [col.replace('final_', '').replace('_', ' ').title() for col in outcome_columns])
        
        # Add correlation values to cells
        for i, outcome in enumerate(outcome_columns):
            for j, param in enumerate(param_columns):
                corr = corr_matrix.loc[outcome, param]
                color = 'white' if abs(corr) > 0.3 else 'black'
                plt.text(j, i, f'{corr:.2f}', ha='center', va='center', color=color)
        
        plt.title('Parameter-Outcome Correlation Heatmap')
        plt.tight_layout()
        
        # Save the heatmap
        heatmap_path = plots_dir / f"correlation_heatmap.{save_format}"
        plt.savefig(heatmap_path)
        plt.close()
        
        saved_plots.append(str(heatmap_path))
        
        return saved_plots
    
    def generate_report(self, output_format='markdown'):
        """
        Generate a report summarizing Monte Carlo simulation results.
        
        Args:
            output_format (str): Format for the report (markdown, html, text)
            
        Returns:
            str: Report content
        """
        if not hasattr(self, 'metric_stats'):
            raise ValueError("Results must be processed before generating a report")
            
        # Start building the report
        report = []
        
        if output_format == 'markdown':
            # Markdown format
            report.append("# Monte Carlo Simulation Results\n")
            report.append(f"**Number of Simulation Runs:** {self.n_runs}\n")
            
            # Parameters varied
            report.append("## Parameters Varied\n")
            for param in self.parameters:
                param_name = param['name'].replace('_', ' ').title()
                report.append(f"- **{param_name}**")
                report.append(f"  - Range: {param['min']} to {param['max']}")
                if 'distribution' in param:
                    report.append(f"  - Distribution: {param['distribution'].title()}")
                report.append("")
            
            # Key outcome metrics
            report.append("## Key Outcome Metrics\n")
            
            outcome_columns = sorted([col for col in self.metric_stats.keys()])
            
            for outcome in outcome_columns:
                outcome_name = outcome.replace('final_', '').replace('_', ' ').title()
                stats = self.metric_stats[outcome]
                
                report.append(f"### {outcome_name}\n")
                report.append(f"- **Mean:** {stats['mean']:.4f}")
                report.append(f"- **Median:** {stats['median']:.4f}")
                report.append(f"- **Standard Deviation:** {stats['std']:.4f}")
                report.append(f"- **Range:** {stats['min']:.4f} to {stats['max']:.4f}")
                report.append(f"- **90% Confidence Interval:** {stats['5th_percentile']:.4f} to {stats['95th_percentile']:.4f}\n")
            
            # Sensitivity analysis
            report.append("## Sensitivity Analysis\n")
            report.append("The table below shows the correlation between input parameters and outcome metrics.\n")
            
            # Create correlation table
            report.append("| Outcome Metric | Most Sensitive Parameters | Correlation |\n")
            report.append("|---------------|---------------------------|-------------|\n")
            
            for outcome in outcome_columns:
                outcome_name = outcome.replace('final_', '').replace('_', ' ').title()
                
                # Sort parameters by absolute correlation value
                sorted_params = sorted(self.correlations[outcome].items(), 
                                     key=lambda x: abs(x[1]), reverse=True)
                
                # Take top 3 parameters (or fewer if less available)
                top_params = sorted_params[:min(3, len(sorted_params))]
                
                for i, (param, corr) in enumerate(top_params):
                    param_name = param.replace('param_', '').replace('_', ' ').title()
                    
                    if i == 0:
                        report.append(f"| {outcome_name} | {param_name} | {corr:.4f} |")
                    else:
                        report.append(f"|  | {param_name} | {corr:.4f} |")
            
            # Conclusions
            report.append("\n## Conclusions\n")
            
            # Find most uncertain outcomes (highest coefficient of variation)
            cv_values = {}
            for outcome, stats in self.metric_stats.items():
                if stats['mean'] != 0:
                    cv_values[outcome] = abs(stats['std'] / stats['mean'])
            
            most_uncertain = sorted(cv_values.items(), key=lambda x: x[1], reverse=True)
            
            if most_uncertain:
                uncertain_outcome = most_uncertain[0][0]
                uncertain_name = uncertain_outcome.replace('final_', '').replace('_', ' ').title()
                
                report.append(f"- **{uncertain_name}** shows the highest relative uncertainty with a coefficient of variation of {most_uncertain[0][1]:.2f}.\n")
            
            # Find most influential parameters
            all_param_influence = {}
            for param in [col for col in self.metrics_df.columns if col.startswith('param_')]:
                # Sum of absolute correlations across all outcomes
                influence = sum(abs(self.correlations[outcome][param]) for outcome in outcome_columns)
                all_param_influence[param] = influence
            
            most_influential = sorted(all_param_influence.items(), key=lambda x: x[1], reverse=True)
            
            if most_influential:
                influential_param = most_influential[0][0]
                influential_name = influential_param.replace('param_', '').replace('_', ' ').title()
                
                report.append(f"- **{influential_name}** is the most influential parameter, significantly affecting multiple outcomes.\n")
            
            # Recommendations
            report.append("### Recommendations\n")
            
            report.append("Based on the sensitivity analysis, the following recommendations are made:\n")
            
            if most_influential:
                report.append(f"1. Focus on more accurately estimating **{influential_name}** as it has the greatest impact on simulation outcomes.\n")
            
            report.append(f"2. Consider policy interventions that target the most sensitive parameters to achieve desired outcomes.\n")
            
            if most_uncertain:
                report.append(f"3. Develop more robust models for predicting **{uncertain_name}** to reduce uncertainty in outcomes.\n")
                
        elif output_format == 'html':
            # HTML format implementation would go here
            pass
            
        elif output_format == 'text':
            # Plain text format implementation would go here
            pass
        
        # Join all parts of the report
        return '\n'.join(report)
