# Bangladesh Development Simulation

A comprehensive simulation model for analyzing and projecting development trends in Bangladesh across economic, demographic, environmental, infrastructure, and governance dimensions.

## Overview

This project implements a multi-system simulation model for Bangladesh's development trajectory. The model integrates five interconnected systems to capture the complex interactions between different aspects of development.

## Recent Updates

- **Enhanced Cross-System Integration**: Improved the `SystemIntegrator` class to facilitate robust interactions between different subsystems
- **Flexible System Execution**: Added support for dynamic system scheduling and step parameters
- **Robust Parameter Validation**: Implemented comprehensive validation framework for configuration parameters
- **Improved Error Handling**: Enhanced error reporting and graceful degradation when components fail
- **Memory Optimization**: Reduced memory usage for large simulations through efficient data structures

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
  - `system_integrator.py`: Cross-system integration utilities
  - `result_processor.py`: Simulation result processing
- `visualization/`: Visualization tools and geographic mapping functionality
- `config/`: Configuration settings for simulations
- `tests/`: Unit and integration tests
- `docs/`: Documentation and model specifications
- `data/`: Input data for the simulation models
- `results/`: Output directory for simulation results and reports

## Data Files

**Note:** Large data files are not included in this repository due to GitHub size limitations. These include:

- Geographic shapefiles (*.shp)
- GIS database files (*.gdb.zip)
- Large datasets (*.zip)

### Obtaining Data Files

To run the simulation with complete data, you'll need to download these files separately:

1. **Bangladesh Admin Boundaries**: 
   - Download from [Humanitarian Data Exchange](https://data.humdata.org/dataset/bangladesh-administrative-boundaries)
   - Place .shp files in `data/shapefiles/bgd_adm_bbs_20201113_SHP/`

2. **Bangladesh GIS Database**:
   - Contact the research team at [contact@example.com](mailto:contact@example.com)
   - Place downloaded files in `data/` directory

3. **Economic and Demographic Data**:
   - Download from [Bangladesh Bureau of Statistics](https://bbs.gov.bd/site/page/db69b9c4-7687-4c96-bdb6-1d71d8b414ef/-)
   - Extract to `data/` directory

The simulation will look for these files in their respective directories. If not found, it will use built-in fallback data that's less detailed but sufficient for basic simulations.

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the data files as described in the [Data Files](#data-files) section.

3. Run a basic simulation:
   ```
   python run_simulation.py
   ```

4. Run a simulation with policy interventions:
   ```
   python run_simulation.py --policy-scenario=education_investment
   ```

5. Run a sensitivity analysis:
   ```
   python run_simulation.py --sensitivity-analysis
   ```

6. View the generated HTML report in the `results/[simulation_id]/` directory

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
