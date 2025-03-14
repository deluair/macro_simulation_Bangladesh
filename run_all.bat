@echo off
echo ========================================================================
echo BANGLADESH INTEGRATED SOCIOECONOMIC AND ENVIRONMENTAL SIMULATION SYSTEM
echo ========================================================================
echo.

:menu
echo Choose an option:
echo 1. Run model validation (tests against historical data)
echo 2. Run future projection simulation
echo 3. Run sensitivity analysis (Monte Carlo simulation)
echo 4. Generate documentation
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto validation
if "%choice%"=="2" goto simulation
if "%choice%"=="3" goto monte_carlo
if "%choice%"=="4" goto documentation
if "%choice%"=="5" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:validation
echo.
echo Running model validation...
python validation_test.py
echo.
pause
goto menu

:simulation
echo.
echo Running future projection simulation...
python run_simulation.py
echo.
pause
goto menu

:monte_carlo
echo.
echo Preparing Monte Carlo simulation...
echo This will run multiple simulations with varying parameters.
echo.
set /p runs="Enter number of simulation runs (default: 100): "
if "%runs%"=="" set runs=100

echo Updating configuration for Monte Carlo with %runs% runs...
echo Running Monte Carlo simulation...
python -c "import yaml; config = yaml.safe_load(open('config/config.yaml')); config['simulation']['monte_carlo']['enabled'] = True; config['simulation']['monte_carlo']['n_runs'] = %runs%; yaml.dump(config, open('config/temp_config.yaml', 'w'))"
move /y config\temp_config.yaml config\config.yaml >nul
python run_simulation.py
echo.
pause
goto menu

:documentation
echo.
echo Generating documentation...
echo.
echo Documentation will be available in the docs directory.
if not exist docs mkdir docs
python -c "import os; import markdown; from pathlib import Path; readme = Path('README.md').read_text(); html = markdown.markdown(readme); Path('docs/index.html').write_text(f'<!DOCTYPE html><html><head><title>Bangladesh Simulation Model</title><style>body{{font-family:Arial;max-width:900px;margin:0 auto;padding:20px;}}</style></head><body>{html}</body></html>')"
echo Documentation generated successfully.
echo.
pause
goto menu

:end
echo.
echo Thank you for using the Bangladesh Simulation System!
echo.
