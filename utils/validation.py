#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation metrics for the Bangladesh simulation model.
This module provides tools for validating model outputs against historical data
and assessing model performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, List, Tuple


class ValidationMetrics:
    """Enhanced validation framework for simulation models with historical backtesting."""
    
    def __init__(self, historical_data_path: str):
        """
        Initialize validation metrics calculator.
        
        Args:
            historical_data_path: Path to historical data for model validation
        """
        self.historical_data = pd.read_csv(historical_data_path)
        self.validation_results = {}
        
    def calculate_historical_fit(self, model_outputs: Dict[str, List[float]], 
                                 variables: List[str], 
                                 years: List[int]) -> Dict[str, Dict[str, float]]:
        """
        Calculate goodness-of-fit metrics between model outputs and historical data.
        
        Args:
            model_outputs: Dictionary of model outputs by variable
            variables: List of variables to validate
            years: List of years for validation period
            
        Returns:
            Dictionary of validation metrics by variable
        """
        results = {}
        
        for var in variables:
            if var not in model_outputs:
                continue
                
            # Extract historical and modeled data for the variable
            historical = self.historical_data[self.historical_data['Year'].isin(years)][var].values
            modeled = np.array(model_outputs[var])[:len(historical)]
            
            # Calculate various goodness-of-fit metrics
            results[var] = {
                'rmse': np.sqrt(np.mean((historical - modeled) ** 2)),
                'mean_abs_error': np.mean(np.abs(historical - modeled)),
                'r_squared': stats.pearsonr(historical, modeled)[0] ** 2,
                'theil_u': self._calculate_theil_u(historical, modeled),
                'bias_proportion': self._calculate_bias_proportion(historical, modeled),
                'variance_proportion': self._calculate_variance_proportion(historical, modeled)
            }
            
        self.validation_results = results
        return results
    
    def calculate_out_of_sample_performance(self, model_outputs: Dict[str, List[float]],
                                            variables: List[str],
                                            training_years: List[int],
                                            test_years: List[int]) -> Dict[str, Dict[str, float]]:
        """
        Calculate out-of-sample performance metrics.
        
        Args:
            model_outputs: Dictionary of model outputs by variable
            variables: List of variables to validate
            training_years: Years used for model calibration
            test_years: Years used for out-of-sample testing
            
        Returns:
            Dictionary of out-of-sample metrics by variable
        """
        # Implementation for out-of-sample validation
        results = {}
        
        for var in variables:
            if var not in model_outputs:
                continue
                
            # Extract training and testing data
            historical_train = self.historical_data[self.historical_data['Year'].isin(training_years)][var].values
            historical_test = self.historical_data[self.historical_data['Year'].isin(test_years)][var].values
            
            # Get corresponding model outputs
            train_indices = [i for i, y in enumerate(model_outputs['years']) if y in training_years]
            test_indices = [i for i, y in enumerate(model_outputs['years']) if y in test_years]
            
            modeled_train = np.array([model_outputs[var][i] for i in train_indices])
            modeled_test = np.array([model_outputs[var][i] for i in test_indices])
            
            # Calculate out-of-sample metrics
            train_rmse = np.sqrt(np.mean((historical_train - modeled_train) ** 2))
            test_rmse = np.sqrt(np.mean((historical_test - modeled_test) ** 2))
            
            results[var] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'performance_ratio': test_rmse / train_rmse if train_rmse > 0 else float('inf')
            }
            
        return results
    
    def parameter_sensitivity_analysis(self, model_func, base_params: Dict[str, float],
                                      param_ranges: Dict[str, Tuple[float, float]],
                                      num_samples: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Perform local sensitivity analysis on model parameters.
        
        Args:
            model_func: Function that runs model with given parameters
            base_params: Base parameter values
            param_ranges: Dictionary of parameter ranges (min, max)
            num_samples: Number of samples per parameter
            
        Returns:
            Dictionary of sensitivity metrics by parameter
        """
        results = {}
        
        for param, (min_val, max_val) in param_ranges.items():
            # Create parameter values
            param_values = np.linspace(min_val, max_val, num_samples)
            outputs = []
            
            # Run model with each parameter value
            for val in param_values:
                params = base_params.copy()
                params[param] = val
                output = model_func(params)
                outputs.append(output)
                
            # Calculate sensitivity metrics
            results[param] = {
                'elasticity': self._calculate_elasticity(param_values, outputs, base_params[param]),
                'partial_rank_correlation': self._calculate_prcc(param_values, outputs),
                'variance': np.var(outputs),
                'range_ratio': np.max(outputs) / np.min(outputs) if np.min(outputs) > 0 else float('inf')
            }
            
        return results
    
    def generate_validation_report(self, output_path: str) -> None:
        """
        Generate a comprehensive validation report.
        
        Args:
            output_path: Path to save the validation report
        """
        # Implementation for generating validation report
        report = pd.DataFrame()
        
        for var, metrics in self.validation_results.items():
            metrics_df = pd.DataFrame(metrics, index=[var])
            report = pd.concat([report, metrics_df])
            
        report.to_csv(output_path)
        
    def _calculate_theil_u(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Theil's U statistic (inequality coefficient)."""
        # Adjusted to handle edge cases
        n = len(observed)
        if n <= 1:
            return 0.0
            
        # Calculate Theil's U (2nd formula - forecasting accuracy)
        squared_diff = np.sum((predicted - observed) ** 2)
        squared_obs = np.sum(observed ** 2)
        
        if squared_obs == 0:
            return float('inf')
            
        return np.sqrt(squared_diff / squared_obs)
    
    def _calculate_bias_proportion(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate bias proportion of MSE."""
        if len(observed) == 0:
            return 0.0
            
        mean_obs = np.mean(observed)
        mean_pred = np.mean(predicted)
        mse = np.mean((observed - predicted) ** 2)
        
        if mse == 0:
            return 0.0
            
        return ((mean_pred - mean_obs) ** 2) / mse
    
    def _calculate_variance_proportion(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate variance proportion of MSE."""
        if len(observed) == 0:
            return 0.0
            
        std_obs = np.std(observed)
        std_pred = np.std(predicted)
        mse = np.mean((observed - predicted) ** 2)
        
        if mse == 0:
            return 0.0
            
        return ((std_pred - std_obs) ** 2) / mse
    
    def _calculate_elasticity(self, param_values: np.ndarray, 
                             outputs: List[Any], 
                             base_param: float) -> float:
        """Calculate elasticity for sensitivity analysis."""
        if len(param_values) < 2 or len(outputs) < 2:
            return 0.0
            
        # Convert outputs to numeric values if they're not already
        if not isinstance(outputs[0], (int, float)):
            # Extract a numeric value from the output (assuming it has a 'value' attribute or is a dict)
            try:
                numeric_outputs = [o.value if hasattr(o, 'value') else o['value'] for o in outputs]
            except:
                return 0.0
        else:
            numeric_outputs = outputs
            
        # Calculate elasticity
        numeric_outputs = np.array(numeric_outputs)
        
        # Get base output
        base_idx = np.argmin(np.abs(param_values - base_param))
        base_output = numeric_outputs[base_idx]
        
        # Calculate average elasticity across the range
        elasticities = []
        for i in range(1, len(param_values)):
            param_diff = param_values[i] - param_values[i-1]
            output_diff = numeric_outputs[i] - numeric_outputs[i-1]
            
            if param_diff == 0 or base_param == 0 or base_output == 0:
                continue
                
            point_elasticity = (output_diff / param_diff) * (base_param / base_output)
            elasticities.append(point_elasticity)
            
        if not elasticities:
            return 0.0
            
        return np.mean(elasticities)
    
    def _calculate_prcc(self, param_values: np.ndarray, outputs: List[Any]) -> float:
        """Calculate Partial Rank Correlation Coefficient."""
        if len(param_values) < 3 or len(outputs) < 3:
            return 0.0
            
        # Convert outputs to numeric values if they're not already
        if not isinstance(outputs[0], (int, float)):
            # Extract a numeric value from the output
            try:
                numeric_outputs = [o.value if hasattr(o, 'value') else o['value'] for o in outputs]
            except:
                return 0.0
        else:
            numeric_outputs = outputs
            
        # Calculate ranks
        param_ranks = stats.rankdata(param_values)
        output_ranks = stats.rankdata(numeric_outputs)
        
        # Calculate PRCC
        try:
            correlation, p_value = stats.pearsonr(param_ranks, output_ranks)
            return correlation
        except:
            return 0.0
