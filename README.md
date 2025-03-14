# Bangladesh Development Simulation

A comprehensive simulation model for analyzing and projecting development trends in Bangladesh across economic, demographic, environmental, infrastructure, and governance dimensions.

## Overview

This project implements a multi-system simulation model for Bangladesh's development trajectory. The model integrates five interconnected systems to capture the complex interactions between different aspects of development.

## Key Features

- **Integrated Systems Model**: Combines economic, demographic, environmental, infrastructure, and governance systems
- **Data-driven Approach**: Based on historical trends and empirical relationships
- **Advanced Sensitivity Analysis**: Robust sensitivity analysis with Sobol indices and variance decomposition
- **Flexible Time Steps**: Support for variable time steps (days, months, quarters, years) with system-specific execution schedules
- **Policy Intervention Framework**: Comprehensive policy scenario testing and comparative analysis
- **Monte Carlo Simulation**: Uncertainty analysis through Monte Carlo methods
- **Model Validation**: Historical backtesting and out-of-sample performance metrics
- **Spatial Analysis**: Geographic visualizations to highlight regional development patterns
- **Comprehensive Reporting**: Generates detailed HTML reports with visualizations and key metrics

## Repository Structure

- `models/`: Core simulation model components
  - `economic/`: Economic system models
  - `demographic/`: Demographic system models
  - `environmental/`: Environmental system models
  - `infrastructure/`: Infrastructure system models
  - `governance/`: Governance system models
  - `policy_framework.py`: Policy intervention framework
- `utils/`: Utility functions
  - `validation.py`: Model validation framework
  - `sensitivity_analysis.py`: Sensitivity analysis tools 
  - `data_loader.py`: Data loading utilities
  - `html_report_generator.py`: Report generation tools
- `visualization/`: Visualization tools and geographic mapping functionality
- `config/`: Configuration settings for simulations
- `data/`: Input data for the simulation models
- `results/`: Output directory for simulation results and reports

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run a basic simulation:
   ```
   python run_simulation.py
   ```

3. Run a simulation with policy interventions:
   ```
   python run_simulation.py --policy-scenario=education_investment
   ```

4. Run a sensitivity analysis:
   ```
   python run_simulation.py --sensitivity-analysis
   ```

5. View the generated HTML report in the `results/[simulation_id]/` directory

## Policy Scenarios

The simulation includes several pre-defined policy scenarios that can be tested:

- **Education Investment**: Increased investment in education infrastructure and quality
- **Renewable Energy Transition**: Accelerating transition to renewable energy sources
- **Agricultural Modernization**: Technology and efficiency improvements in agriculture
- **Export Diversification**: Policies to diversify exports beyond garments

Custom policy scenarios can be created by combining different interventions.

## Advanced Usage

### Variable Time Steps

The simulation supports different time step granularities:

```python
# Run with quarterly time steps
python run_simulation.py --time-step=quarter
```

### Sensitivity Analysis

```python
# Run Sobol sensitivity analysis
python run_simulation.py --sensitivity-method=sobol --n-samples=1000
```

### Model Validation

```python
# Run historical validation
python run_simulation.py --validate --validation-period=2000-2020
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- University of Tennessee Research Team
