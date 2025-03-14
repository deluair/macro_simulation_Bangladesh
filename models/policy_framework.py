import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Callable, Tuple
from dataclasses import dataclass
import os
import json
from datetime import datetime

@dataclass
class PolicyIntervention:
    """Class representing a policy intervention in the simulation."""
    
    name: str
    description: str
    target_system: str  # 'economic', 'environmental', 'demographic', 'infrastructure', 'governance'
    parameters: Dict[str, Any]  # Parameter changes to implement the policy
    start_year: int
    end_year: int = None  # None means policy continues to end of simulation
    intensity: float = 1.0  # Scaling factor for intervention intensity (0.0-1.0)
    gradual_implementation: bool = False  # Whether policy is implemented gradually
    implementation_period: int = 1  # Years to reach full implementation if gradual
    regional_targeting: Dict[str, float] = None  # Region-specific implementation intensities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'target_system': self.target_system,
            'parameters': self.parameters,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'intensity': self.intensity,
            'gradual_implementation': self.gradual_implementation,
            'implementation_period': self.implementation_period,
            'regional_targeting': self.regional_targeting
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyIntervention':
        """Create policy from dictionary representation."""
        return cls(
            name=data['name'],
            description=data['description'],
            target_system=data['target_system'],
            parameters=data['parameters'],
            start_year=data['start_year'],
            end_year=data.get('end_year'),
            intensity=data.get('intensity', 1.0),
            gradual_implementation=data.get('gradual_implementation', False),
            implementation_period=data.get('implementation_period', 1),
            regional_targeting=data.get('regional_targeting')
        )


class PolicyScenario:
    """Class representing a collection of policy interventions as a scenario."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a policy scenario.
        
        Args:
            name: Name of the scenario
            description: Description of the scenario
        """
        self.name = name
        self.description = description
        self.interventions: List[PolicyIntervention] = []
        self.baseline_scenario = None
        
    def add_intervention(self, intervention: PolicyIntervention) -> None:
        """Add a policy intervention to the scenario."""
        self.interventions.append(intervention)
        
    def remove_intervention(self, intervention_name: str) -> None:
        """Remove a policy intervention by name."""
        self.interventions = [i for i in self.interventions if i.name != intervention_name]
        
    def set_baseline(self, baseline: 'PolicyScenario') -> None:
        """Set a baseline scenario for comparison."""
        self.baseline_scenario = baseline
        
    def save(self, output_dir: str) -> str:
        """
        Save the policy scenario to a JSON file.
        
        Args:
            output_dir: Directory to save the scenario
            
        Returns:
            Path to the saved file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name.replace(' ', '_')}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data
        data = {
            'name': self.name,
            'description': self.description,
            'interventions': [i.to_dict() for i in self.interventions],
            'baseline': self.baseline_scenario.name if self.baseline_scenario else None
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'PolicyScenario':
        """
        Load a policy scenario from a JSON file.
        
        Args:
            filepath: Path to the scenario file
            
        Returns:
            PolicyScenario object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        scenario = cls(name=data['name'], description=data['description'])
        
        for intervention_data in data['interventions']:
            intervention = PolicyIntervention.from_dict(intervention_data)
            scenario.add_intervention(intervention)
            
        return scenario
        

class PolicyFramework:
    """Framework for defining, analyzing, and comparing policy interventions."""
    
    def __init__(self, simulation_function: Callable, output_dir: str = 'results/policy'):
        """
        Initialize the policy framework.
        
        Args:
            simulation_function: Function that runs a simulation with given policy scenario
            output_dir: Directory for saving policy analysis results
        """
        self.simulation_function = simulation_function
        self.output_dir = output_dir
        self.scenarios: Dict[str, PolicyScenario] = {}
        self.results: Dict[str, Any] = {}
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def add_scenario(self, scenario: PolicyScenario) -> None:
        """Add a policy scenario to the framework."""
        self.scenarios[scenario.name] = scenario
        
    def create_scenario(self, name: str, description: str = "") -> PolicyScenario:
        """
        Create a new policy scenario.
        
        Args:
            name: Name of the scenario
            description: Description of the scenario
            
        Returns:
            The created scenario
        """
        scenario = PolicyScenario(name, description)
        self.add_scenario(scenario)
        return scenario
        
    def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Run a simulation for a specific policy scenario.
        
        Args:
            scenario_name: Name of the scenario to run
            
        Returns:
            Simulation results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
            
        scenario = self.scenarios[scenario_name]
        
        # Run simulation with this scenario
        results = self.simulation_function(scenario)
        
        # Store results
        self.results[scenario_name] = results
        
        return results
    
    def run_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Run simulations for all defined scenarios.
        
        Returns:
            Dictionary mapping scenario names to simulation results
        """
        for scenario_name in self.scenarios:
            self.run_scenario(scenario_name)
            
        return self.results
    
    def compare_scenarios(self, scenario_names: List[str], 
                         output_variables: List[str]) -> pd.DataFrame:
        """
        Compare multiple scenarios based on specified output variables.
        
        Args:
            scenario_names: Names of scenarios to compare
            output_variables: Names of output variables to compare
            
        Returns:
            DataFrame with comparison results
        """
        # Ensure all scenarios have been run
        missing_scenarios = [name for name in scenario_names if name not in self.results]
        if missing_scenarios:
            for name in missing_scenarios:
                if name in self.scenarios:
                    self.run_scenario(name)
                else:
                    raise ValueError(f"Scenario '{name}' not found")
        
        # Prepare comparison dataframe
        comparison = pd.DataFrame()
        
        for var in output_variables:
            for scenario in scenario_names:
                scenario_result = self.results[scenario]
                
                if var in scenario_result:
                    # Get final value or time series depending on result format
                    if isinstance(scenario_result[var], list):
                        # Time series - use final value
                        value = scenario_result[var][-1]
                    else:
                        # Single value
                        value = scenario_result[var]
                        
                    comparison.loc[scenario, var] = value
                else:
                    comparison.loc[scenario, var] = np.nan
                    
        return comparison
    
    def calculate_scenario_impact(self, scenario_name: str, 
                                baseline_name: str,
                                output_variables: List[str]) -> pd.DataFrame:
        """
        Calculate the impact of a policy scenario compared to a baseline.
        
        Args:
            scenario_name: Name of the scenario to evaluate
            baseline_name: Name of the baseline scenario
            output_variables: Names of output variables to compare
            
        Returns:
            DataFrame with impact results
        """
        # Ensure both scenarios have been run
        if scenario_name not in self.results:
            self.run_scenario(scenario_name)
            
        if baseline_name not in self.results:
            self.run_scenario(baseline_name)
            
        # Get results
        scenario_result = self.results[scenario_name]
        baseline_result = self.results[baseline_name]
        
        # Calculate impacts
        impacts = pd.DataFrame(index=output_variables, 
                             columns=['baseline', 'scenario', 'difference', 'percent_change'])
        
        for var in output_variables:
            if var in scenario_result and var in baseline_result:
                # Get final values
                if isinstance(scenario_result[var], list) and isinstance(baseline_result[var], list):
                    scenario_value = scenario_result[var][-1]
                    baseline_value = baseline_result[var][-1]
                else:
                    scenario_value = scenario_result[var]
                    baseline_value = baseline_result[var]
                    
                # Calculate metrics
                difference = scenario_value - baseline_value
                percent_change = (difference / baseline_value) * 100 if baseline_value != 0 else np.inf
                
                # Store in dataframe
                impacts.loc[var, 'baseline'] = baseline_value
                impacts.loc[var, 'scenario'] = scenario_value
                impacts.loc[var, 'difference'] = difference
                impacts.loc[var, 'percent_change'] = percent_change
                
        return impacts
    
    def visualize_scenario_comparison(self, scenario_names: List[str],
                                    output_variable: str,
                                    time_series: bool = True) -> None:
        """
        Create visualization comparing scenarios for a specific output variable.
        
        Args:
            scenario_names: Names of scenarios to compare
            output_variable: Name of the output variable to visualize
            time_series: Whether to show time series (True) or final values (False)
        """
        # Ensure all scenarios have been run
        for name in scenario_names:
            if name not in self.results and name in self.scenarios:
                self.run_scenario(name)
                
        if time_series:
            # Time series comparison
            plt.figure(figsize=(12, 6))
            
            for scenario in scenario_names:
                if scenario in self.results and output_variable in self.results[scenario]:
                    result = self.results[scenario]
                    
                    # Check if we have time series data
                    if isinstance(result[output_variable], list):
                        # Plot time series
                        plt.plot(result['years'], result[output_variable], 
                               label=scenario, linewidth=2)
                        
            plt.title(f'Comparison of {output_variable} across Policy Scenarios', fontsize=14)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel(output_variable, fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.output_dir}/comparison_{output_variable}_timeseries.png", dpi=300)
            plt.close()
            
        else:
            # Bar chart of final values
            values = []
            labels = []
            
            for scenario in scenario_names:
                if scenario in self.results and output_variable in self.results[scenario]:
                    result = self.results[scenario]
                    
                    # Get final value
                    if isinstance(result[output_variable], list):
                        value = result[output_variable][-1]
                    else:
                        value = result[output_variable]
                        
                    values.append(value)
                    labels.append(scenario)
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(labels, values)
            plt.title(f'Comparison of {output_variable} across Policy Scenarios', fontsize=14)
            plt.ylabel(output_variable, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.output_dir}/comparison_{output_variable}_final.png", dpi=300)
            plt.close()
    
    def visualize_policy_impact(self, scenario_name: str, 
                              baseline_name: str,
                              output_variables: List[str],
                              sort_by: str = 'percent_change') -> None:
        """
        Create visualization of policy impacts compared to baseline.
        
        Args:
            scenario_name: Name of the scenario to evaluate
            baseline_name: Name of the baseline scenario
            output_variables: Names of output variables to visualize
            sort_by: Metric to sort by ('percent_change' or 'difference')
        """
        # Calculate impacts
        impacts = self.calculate_scenario_impact(
            scenario_name, baseline_name, output_variables
        )
        
        # Sort by specified metric
        impacts = impacts.sort_values(by=sort_by, ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot percent changes
        bars = plt.barh(impacts.index, impacts['percent_change'])
        
        # Color bars based on positive/negative impact
        for i, bar in enumerate(bars):
            if bar.get_width() < 0:
                bar.set_color('#d62728')  # Red for negative
            else:
                bar.set_color('#1f77b4')  # Blue for positive
        
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.title(f'Impact of {scenario_name} vs {baseline_name}', fontsize=14)
        plt.xlabel('Percent Change (%)', fontsize=12)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.output_dir}/impact_{scenario_name}_vs_{baseline_name}.png", dpi=300)
        plt.close()
    
    def generate_policy_report(self, scenario_names: List[str], 
                             baseline_name: str,
                             output_variables: List[str],
                             output_file: str = None) -> str:
        """
        Generate a comprehensive policy analysis report.
        
        Args:
            scenario_names: Names of scenarios to include in report
            baseline_name: Name of the baseline scenario
            output_variables: Names of output variables to analyze
            output_file: Path to save the report (default: auto-generated)
            
        Returns:
            Path to the generated report
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Default output file if not specified
        if output_file is None:
            output_file = f"{self.output_dir}/policy_report_{timestamp}.html"
            
        # Ensure all scenarios have been run
        for name in scenario_names + [baseline_name]:
            if name not in self.results and name in self.scenarios:
                self.run_scenario(name)
                
        # Create report content
        report = []
        report.append("<html><head>")
        report.append("<style>")
        report.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        report.append("h1 { color: #333366; }")
        report.append("h2 { color: #333366; margin-top: 30px; }")
        report.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        report.append("th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
        report.append("th { background-color: #f2f2f2; }")
        report.append("tr:hover { background-color: #f5f5f5; }")
        report.append(".positive { color: green; }")
        report.append(".negative { color: red; }")
        report.append("</style>")
        report.append("<title>Policy Analysis Report</title>")
        report.append("</head><body>")
        
        # Header
        report.append(f"<h1>Bangladesh Development Policy Analysis Report</h1>")
        report.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Scenarios overview
        report.append("<h2>Policy Scenarios</h2>")
        report.append("<table>")
        report.append("<tr><th>Scenario</th><th>Description</th><th>Number of Interventions</th></tr>")
        
        for name in scenario_names + [baseline_name]:
            if name in self.scenarios:
                scenario = self.scenarios[name]
                report.append(f"<tr>")
                report.append(f"<td>{scenario.name}</td>")
                report.append(f"<td>{scenario.description}</td>")
                report.append(f"<td>{len(scenario.interventions)}</td>")
                report.append(f"</tr>")
                
        report.append("</table>")
        
        # Comparison of key indicators
        report.append("<h2>Key Indicators Comparison</h2>")
        
        # Create comparison table
        comparison = self.compare_scenarios(scenario_names + [baseline_name], output_variables)
        
        report.append("<table>")
        
        # Header row
        report.append("<tr><th>Indicator</th>")
        for scenario in comparison.index:
            report.append(f"<th>{scenario}</th>")
        report.append("</tr>")
        
        # Data rows
        for var in comparison.columns:
            report.append(f"<tr><td>{var}</td>")
            
            baseline_value = comparison.loc[baseline_name, var] if baseline_name in comparison.index else None
            
            for scenario in comparison.index:
                value = comparison.loc[scenario, var]
                
                # Add color coding for non-baseline scenarios
                if scenario != baseline_name and baseline_value is not None:
                    percent_change = ((value - baseline_value) / baseline_value) * 100 if baseline_value != 0 else 0
                    
                    if percent_change > 0:
                        report.append(f"<td class='positive'>{value:.2f} (+{percent_change:.1f}%)</td>")
                    elif percent_change < 0:
                        report.append(f"<td class='negative'>{value:.2f} ({percent_change:.1f}%)</td>")
                    else:
                        report.append(f"<td>{value:.2f}</td>")
                else:
                    report.append(f"<td>{value:.2f}</td>")
                    
            report.append("</tr>")
            
        report.append("</table>")
        
        # Detailed impact analysis for each scenario
        report.append("<h2>Detailed Impact Analysis</h2>")
        
        for scenario_name in scenario_names:
            if scenario_name == baseline_name:
                continue
                
            report.append(f"<h3>Impact of {scenario_name} vs {baseline_name}</h3>")
            
            # Calculate impacts
            try:
                impacts = self.calculate_scenario_impact(
                    scenario_name, baseline_name, output_variables
                )
                
                # Create impact table
                report.append("<table>")
                report.append("<tr><th>Indicator</th><th>Baseline</th><th>Scenario</th><th>Difference</th><th>% Change</th></tr>")
                
                # Sort by percent change
                impacts = impacts.sort_values(by='percent_change', ascending=False)
                
                for var in impacts.index:
                    baseline = impacts.loc[var, 'baseline']
                    scenario = impacts.loc[var, 'scenario']
                    difference = impacts.loc[var, 'difference']
                    percent_change = impacts.loc[var, 'percent_change']
                    
                    # Apply color coding based on direction of change
                    if percent_change > 0:
                        class_name = 'positive'
                        percent_str = f"+{percent_change:.1f}%"
                    elif percent_change < 0:
                        class_name = 'negative'
                        percent_str = f"{percent_change:.1f}%"
                    else:
                        class_name = ''
                        percent_str = "0.0%"
                        
                    report.append(f"<tr>")
                    report.append(f"<td>{var}</td>")
                    report.append(f"<td>{baseline:.2f}</td>")
                    report.append(f"<td>{scenario:.2f}</td>")
                    report.append(f"<td class='{class_name}'>{difference:.2f}</td>")
                    report.append(f"<td class='{class_name}'>{percent_str}</td>")
                    report.append(f"</tr>")
                    
                report.append("</table>")
                
            except Exception as e:
                report.append(f"<p>Error calculating impacts: {e}</p>")
        
        # Conclusions
        report.append("<h2>Conclusions and Recommendations</h2>")
        
        # Simple automatic analysis
        try:
            # Find best performing scenario for each indicator
            best_scenarios = {}
            
            for var in output_variables:
                var_comparison = {}
                
                for scenario in scenario_names:
                    if scenario == baseline_name:
                        continue
                        
                    # Get impact data
                    impacts = self.calculate_scenario_impact(
                        scenario, baseline_name, [var]
                    )
                    
                    if var in impacts.index:
                        var_comparison[scenario] = impacts.loc[var, 'percent_change']
                
                # Find best scenario for this variable
                if var_comparison:
                    best_scenario = max(var_comparison.items(), key=lambda x: x[1])
                    best_scenarios[var] = best_scenario
            
            # Generate conclusions
            report.append("<p>Based on the analysis of different policy scenarios, the following recommendations are made:</p>")
            report.append("<ul>")
            
            for var, (scenario, change) in best_scenarios.items():
                report.append(f"<li>For improving <strong>{var}</strong>, the <strong>{scenario}</strong> scenario shows the best results with a <span class='positive'>{change:.1f}%</span> improvement.</li>")
                
            report.append("</ul>")
            
        except Exception as e:
            report.append(f"<p>Error generating conclusions: {e}</p>")
        
        # Close HTML
        report.append("</body></html>")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
            
        return output_file


# Define some common policy interventions for Bangladesh

def create_education_investment_policy(start_year: int, 
                                      intensity: float = 1.0) -> PolicyIntervention:
    """
    Create a policy for increased education investment.
    
    Args:
        start_year: Year to start the policy
        intensity: Policy intensity (0.0-1.0)
        
    Returns:
        PolicyIntervention object
    """
    return PolicyIntervention(
        name="Education Investment Policy",
        description="Increase investment in education to improve human capital development",
        target_system="demographic",
        parameters={
            "education_spending_gdp_share": 0.05 * intensity,  # Percentage point increase
            "teacher_student_ratio_improvement": 0.2 * intensity,
            "educational_technology_investment": 0.02 * intensity,
            "rural_education_access_factor": 0.15 * intensity
        },
        start_year=start_year,
        gradual_implementation=True,
        implementation_period=3
    )

def create_renewable_energy_policy(start_year: int,
                                 intensity: float = 1.0) -> PolicyIntervention:
    """
    Create a policy for renewable energy investment.
    
    Args:
        start_year: Year to start the policy
        intensity: Policy intensity (0.0-1.0)
        
    Returns:
        PolicyIntervention object
    """
    return PolicyIntervention(
        name="Renewable Energy Transition",
        description="Accelerate transition to renewable energy sources",
        target_system="infrastructure",
        parameters={
            "renewable_energy_investment": 0.03 * intensity,  # % of GDP
            "solar_capacity_target": 10000 * intensity,  # MW
            "fossil_fuel_subsidy_reduction": 0.4 * intensity,  # 40% reduction
            "carbon_pricing": 25 * intensity  # USD per ton CO2
        },
        start_year=start_year,
        gradual_implementation=True,
        implementation_period=5
    )

def create_agricultural_modernization_policy(start_year: int,
                                           intensity: float = 1.0) -> PolicyIntervention:
    """
    Create a policy for agricultural modernization.
    
    Args:
        start_year: Year to start the policy
        intensity: Policy intensity (0.0-1.0)
        
    Returns:
        PolicyIntervention object
    """
    return PolicyIntervention(
        name="Agricultural Modernization",
        description="Modernize agricultural practices to improve productivity and climate resilience",
        target_system="economic",
        parameters={
            "agricultural_technology_adoption": 0.25 * intensity,
            "irrigation_improvement": 0.3 * intensity,
            "crop_diversification": 0.4 * intensity,
            "agricultural_extension_services": 0.2 * intensity
        },
        start_year=start_year,
        regional_targeting={
            "rural_north": 1.2,  # Higher intensity in northern regions
            "rural_central": 1.0,
            "rural_south": 0.8
        }
    )

def create_export_diversification_policy(start_year: int,
                                       intensity: float = 1.0) -> PolicyIntervention:
    """
    Create a policy for export diversification.
    
    Args:
        start_year: Year to start the policy
        intensity: Policy intensity (0.0-1.0)
        
    Returns:
        PolicyIntervention object
    """
    return PolicyIntervention(
        name="Export Diversification",
        description="Diversify exports beyond garments to higher value-added sectors",
        target_system="economic",
        parameters={
            "export_incentives_non_garment": 0.04 * intensity,  # % of GDP
            "skills_development_program": 0.02 * intensity,  # % of GDP
            "technology_park_investment": 0.015 * intensity,  # % of GDP
            "trade_facilitation_improvement": 0.2 * intensity
        },
        start_year=start_year,
        gradual_implementation=True,
        implementation_period=4
    ) 