import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Callable, Tuple, Any
from scipy.stats import norm, uniform, lognorm
from SALib.sample import saltelli, latin
from SALib.analyze import sobol, delta, morris

class SensitivityAnalysis:
    """Advanced sensitivity analysis framework for Bangladesh Simulation."""
    
    def __init__(self, model_function: Callable, output_dir: str = 'results/sensitivity'):
        """
        Initialize sensitivity analysis framework.
        
        Args:
            model_function: Function that runs the simulation with given parameters and returns outputs
            output_dir: Directory to save sensitivity analysis results
        """
        self.model_function = model_function
        self.output_dir = output_dir
        self.results = {}
        
    def define_problem(self, parameter_ranges: Dict[str, Tuple[float, float]], 
                      parameter_distributions: Dict[str, str] = None):
        """
        Define the sensitivity analysis problem.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to (min, max) tuples
            parameter_distributions: Dictionary mapping parameter names to distribution types
                                    ('uniform', 'normal', 'lognormal', etc.)
        """
        # Create SALib problem definition
        self.problem = {
            'num_vars': len(parameter_ranges),
            'names': list(parameter_ranges.keys()),
            'bounds': [parameter_ranges[p] for p in self.problem['names']]
        }
        
        # Store distributions for reference
        self.parameter_distributions = parameter_distributions or {
            p: 'uniform' for p in parameter_ranges.keys()
        }
        
    def generate_samples(self, method: str = 'saltelli', 
                        n_samples: int = 1000) -> np.ndarray:
        """
        Generate parameter samples for sensitivity analysis.
        
        Args:
            method: Sampling method ('saltelli', 'latin', 'random')
            n_samples: Number of base samples
            
        Returns:
            Array of parameter samples
        """
        if method == 'saltelli':
            # Saltelli's extension of Sobol sequence (for Sobol indices)
            return saltelli.sample(self.problem, n_samples, calc_second_order=True)
        
        elif method == 'latin':
            # Latin Hypercube Sampling
            return latin.sample(self.problem, n_samples)
        
        elif method == 'random':
            # Simple random sampling
            return np.random.uniform(
                low=[b[0] for b in self.problem['bounds']],
                high=[b[1] for b in self.problem['bounds']],
                size=(n_samples, self.problem['num_vars'])
            )
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def run_model_samples(self, parameter_samples: np.ndarray, 
                         parallel: bool = False) -> Dict[str, np.ndarray]:
        """
        Run the model with the given parameter samples.
        
        Args:
            parameter_samples: Array of parameter samples
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary mapping output names to arrays of output values
        """
        results = {}
        n_samples = parameter_samples.shape[0]
        
        for i in range(n_samples):
            # Create parameter dictionary for this sample
            params = {
                self.problem['names'][j]: parameter_samples[i, j]
                for j in range(self.problem['num_vars'])
            }
            
            # Run model and get outputs
            outputs = self.model_function(params)
            
            # Organize results by output variable
            for output_name, value in outputs.items():
                if output_name not in results:
                    results[output_name] = np.zeros(n_samples)
                results[output_name][i] = value
                
        return results
    
    def calculate_sobol_indices(self, parameter_samples: np.ndarray, 
                              model_outputs: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Calculate Sobol sensitivity indices for each output.
        
        Args:
            parameter_samples: Saltelli samples used for model runs
            model_outputs: Dictionary mapping output names to output values
            
        Returns:
            Dictionary of Sobol indices for each output
        """
        sobol_indices = {}
        
        for output_name, output_values in model_outputs.items():
            # Run Sobol analysis
            sobol_result = sobol.analyze(
                self.problem, output_values, print_to_console=False
            )
            
            # Store results
            sobol_indices[output_name] = {
                'S1': {self.problem['names'][i]: sobol_result['S1'][i] 
                      for i in range(self.problem['num_vars'])},
                'S2': sobol_result['S2'],
                'ST': {self.problem['names'][i]: sobol_result['ST'][i] 
                      for i in range(self.problem['num_vars'])},
            }
            
        return sobol_indices
    
    def calculate_morris_indices(self, parameter_samples: np.ndarray,
                               model_outputs: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Calculate Morris elementary effects for each output.
        
        Args:
            parameter_samples: Samples used for model runs
            model_outputs: Dictionary mapping output names to output values
            
        Returns:
            Dictionary of Morris indices for each output
        """
        morris_indices = {}
        
        for output_name, output_values in model_outputs.items():
            # Run Morris analysis
            morris_result = morris.analyze(
                self.problem, parameter_samples, output_values, 
                print_to_console=False
            )
            
            # Store results
            morris_indices[output_name] = {
                'mu': {self.problem['names'][i]: morris_result['mu'][i] 
                      for i in range(self.problem['num_vars'])},
                'sigma': {self.problem['names'][i]: morris_result['sigma'][i] 
                         for i in range(self.problem['num_vars'])},
                'mu_star': {self.problem['names'][i]: morris_result['mu_star'][i] 
                          for i in range(self.problem['num_vars'])},
            }
            
        return morris_indices
    
    def visualize_sobol_indices(self, sobol_indices: Dict[str, Dict], 
                              output_vars: List[str] = None):
        """
        Visualize Sobol sensitivity indices.
        
        Args:
            sobol_indices: Dictionary of Sobol indices from calculate_sobol_indices
            output_vars: List of output variables to visualize (default: all)
        """
        if output_vars is None:
            output_vars = list(sobol_indices.keys())
            
        for output_var in output_vars:
            if output_var not in sobol_indices:
                continue
                
            indices = sobol_indices[output_var]
            param_names = list(indices['S1'].keys())
            
            # Create figure for this output
            plt.figure(figsize=(12, 8))
            
            # Plot first-order (S1) and total (ST) indices
            width = 0.35
            x = np.arange(len(param_names))
            
            plt.bar(x - width/2, [indices['S1'][p] for p in param_names], 
                   width, label='First-order (S1)')
            plt.bar(x + width/2, [indices['ST'][p] for p in param_names], 
                   width, label='Total (ST)')
            
            plt.xlabel('Parameter', fontsize=12)
            plt.ylabel('Sensitivity Index', fontsize=12)
            plt.title(f'Sobol Sensitivity Indices for {output_var}', fontsize=14)
            plt.xticks(x, param_names, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.output_dir}/sobol_{output_var.replace(' ', '_')}.png", dpi=300)
            plt.close()
            
    def visualize_parameter_interactions(self, sobol_indices: Dict[str, Dict],
                                       output_var: str):
        """
        Visualize parameter interactions using second-order Sobol indices.
        
        Args:
            sobol_indices: Dictionary of Sobol indices from calculate_sobol_indices
            output_var: Output variable to visualize
        """
        if output_var not in sobol_indices:
            return
            
        # Get second-order indices
        S2 = sobol_indices[output_var]['S2']
        param_names = list(sobol_indices[output_var]['S1'].keys())
        n_params = len(param_names)
        
        # Create interaction matrix
        interaction_matrix = np.zeros((n_params, n_params))
        
        # Fill matrix with second-order indices
        for i in range(n_params):
            for j in range(i+1, n_params):
                interaction_matrix[i, j] = S2[i, j]
                interaction_matrix[j, i] = S2[i, j]  # Matrix is symmetric
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(interaction_matrix, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=param_names, yticklabels=param_names)
        
        plt.title(f'Parameter Interactions for {output_var}', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.output_dir}/interactions_{output_var.replace(' ', '_')}.png", dpi=300)
        plt.close()
    
    def tornado_plot(self, sensitivity_data: Dict[str, float], output_var: str):
        """
        Create a tornado plot for sensitivity analysis results.
        
        Args:
            sensitivity_data: Dictionary mapping parameter names to sensitivity values
            output_var: Name of the output variable
        """
        # Sort parameters by absolute sensitivity
        sorted_params = sorted(sensitivity_data.items(), 
                              key=lambda x: abs(x[1]), reverse=True)
        
        param_names = [p[0] for p in sorted_params]
        sensitivities = [p[1] for p in sorted_params]
        
        # Create colors based on positive/negative values
        colors = ['#1f77b4' if s >= 0 else '#d62728' for s in sensitivities]
        
        # Create tornado plot
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(param_names))
        
        plt.barh(y_pos, sensitivities, color=colors)
        plt.yticks(y_pos, param_names)
        plt.xlabel('Sensitivity', fontsize=12)
        plt.title(f'Parameter Sensitivity for {output_var}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.output_dir}/tornado_{output_var.replace(' ', '_')}.png", dpi=300)
        plt.close()
        
    def run_sensitivity_analysis(self, method: str = 'sobol', 
                               n_samples: int = 1000,
                               output_vars: List[str] = None):
        """
        Run a complete sensitivity analysis.
        
        Args:
            method: Sensitivity analysis method ('sobol', 'morris')
            n_samples: Number of base samples to use
            output_vars: List of output variables to analyze (default: all)
            
        Returns:
            Dictionary of sensitivity analysis results
        """
        # Generate samples based on method
        if method == 'sobol':
            samples = self.generate_samples('saltelli', n_samples)
        elif method == 'morris':
            samples = self.generate_samples('latin', n_samples)
        else:
            raise ValueError(f"Unknown sensitivity analysis method: {method}")
            
        # Run model for all samples
        model_outputs = self.run_model_samples(samples)
        
        # Filter output variables if specified
        if output_vars is not None:
            model_outputs = {var: model_outputs[var] for var in output_vars if var in model_outputs}
            
        # Calculate sensitivity indices
        if method == 'sobol':
            sensitivity_indices = self.calculate_sobol_indices(samples, model_outputs)
            
            # Visualize results
            self.visualize_sobol_indices(sensitivity_indices)
            
            # Visualize parameter interactions for each output
            for output_var in model_outputs.keys():
                self.visualize_parameter_interactions(sensitivity_indices, output_var)
                
                # Create tornado plot of total effects
                self.tornado_plot(sensitivity_indices[output_var]['ST'], output_var)
                
        elif method == 'morris':
            sensitivity_indices = self.calculate_morris_indices(samples, model_outputs)
            
            # Visualize results for each output
            for output_var, indices in sensitivity_indices.items():
                # Create tornado plot of mu_star values
                self.tornado_plot(indices['mu_star'], output_var)
                
        # Store results
        self.results[method] = sensitivity_indices
        
        return sensitivity_indices 