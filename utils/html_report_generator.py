"""
HTML Report Generator for Bangladesh Simulation Model.
This module creates comprehensive HTML reports with visualizations and results data.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import base64
import glob
import numpy as np
import pandas as pd

class HTMLReportGenerator:
    """
    Generates HTML reports for simulation results, including visualizations,
    key metrics, and detailed data tables.
    """
    
    def __init__(self, output_dir, simulation_id=None):
        """
        Initialize the HTML report generator.
        
        Args:
            output_dir (str): Path to the output directory
            simulation_id (str, optional): ID of the simulation run
        """
        self.output_dir = Path(output_dir)
        self.simulation_id = simulation_id
        
        if simulation_id is None:
            # Find the most recent simulation directory
            sim_dirs = sorted([d for d in self.output_dir.glob('*') if d.is_dir() and not d.name == 'visualizations'],
                              key=os.path.getmtime, reverse=True)
            if sim_dirs:
                self.simulation_id = sim_dirs[0].name
                
        self.sim_dir = self.output_dir / self.simulation_id if self.simulation_id else None
        
    def generate_report(self, report_title, geo_viz_paths=None):
        """
        Generate a comprehensive HTML report for the simulation results.
        
        Args:
            report_title (str): Title of the report
            geo_viz_paths (dict, optional): Dictionary of paths to geographic visualizations
            
        Returns:
            str: Path to the generated HTML report
        """
        # Store geo_viz_paths for use in _add_geographic_visualizations
        self.geo_viz_paths = geo_viz_paths

        # Create HTML report directory
        html_reports_dir = Path(self.sim_dir).parent / 'html_reports'
        html_reports_dir.mkdir(exist_ok=True)
        
        # Set report path
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_path = str(html_reports_dir / f"simulation_report_{self.simulation_id}.html")
        report_path_with_timestamp = str(html_reports_dir / f"simulation_report_{self.simulation_id}_{time_stamp}.html")
        sim_report_path = str(self.sim_dir / "simulation_report.html")
        
        # Create report content
        html = self._get_html_header(report_title)
        
        # Navigation
        html += """
        <nav class="report-nav">
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#economic">Economic</a></li>
                <li><a href="#demographic">Demographic</a></li>
                <li><a href="#environmental">Environmental</a></li>
                <li><a href="#geography">Geography</a></li>
                <li><a href="#infrastructure">Infrastructure</a></li>
                <li><a href="#governance">Governance</a></li>
            </ul>
        </nav>
        
        <div class="report-content">
        """
        
        # Add sections
        html += self._generate_overview_section()
        html += self._generate_section_html("economic")
        html += self._generate_section_html("demographic")
        html += self._generate_section_html("environmental")
        html += self._generate_geography_section()  # Special section for geographic visualizations
        html += self._generate_section_html("infrastructure")
        html += self._generate_section_html("governance")
        
        # Close report
        html += """
        </div>
        <footer class="report-footer">
            <p>Bangladesh Development Simulation Model &copy; 2023</p>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </footer>
        </body>
        </html>
        """
        
        # Write report to file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Also save a timestamped copy for version history
        with open(report_path_with_timestamp, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Save a copy in the simulation directory itself
        with open(sim_report_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        return self.report_path
    
    def _generate_executive_summary(self, summary_data):
        """Generate an executive summary section."""
        # Extract key metrics from summary data
        try:
            # For GDP Growth Rate - use mean if end_value is NaN
            gdp_growth = summary_data.get('economic', {}).get('gdp_growth', {})
            if gdp_growth.get('end_value') is None or str(gdp_growth.get('end_value')).lower() == 'nan':
                if 'mean' in gdp_growth:
                    gdp_growth_value = gdp_growth.get('mean', 'N/A')
                else:
                    gdp_growth_value = 'N/A'
            else:
                gdp_growth_value = gdp_growth.get('end_value', 'N/A')
                
            if gdp_growth_value != 'N/A':
                gdp_growth_display = f"{float(gdp_growth_value) * 100:.1f}%"
            else:
                gdp_growth_display = "N/A"
            
            # For Population - look for total_population field or calculate from other available data
            population_value = 'N/A'
            
            # Try total_population first
            population_data = summary_data.get('demographic', {}).get('total_population', {})
            if population_data:
                if population_data.get('end_value') is None or str(population_data.get('end_value')).lower() == 'nan':
                    if 'max' in population_data:
                        population_value = population_data.get('max', 'N/A')
                    elif 'mean' in population_data:
                        population_value = population_data.get('mean', 'N/A')
                else:
                    population_value = population_data.get('end_value', 'N/A')
            
            # If a valid value wasn't found, try other fields
            if population_value == 'N/A':
                # Try with GDP and GDP per capita if both available
                gdp_data = summary_data.get('economic', {}).get('total_gdp', {})
                gdp_per_capita_data = summary_data.get('economic', {}).get('gdp_per_capita', {})
                
                if gdp_data and gdp_per_capita_data:
                    gdp_val = gdp_data.get('end_value', None)
                    if gdp_val is None or str(gdp_val).lower() == 'nan':
                        gdp_val = gdp_data.get('mean', None)
                        
                    gdp_per_capita_val = gdp_per_capita_data.get('end_value', None)
                    if gdp_per_capita_val is None or str(gdp_per_capita_val).lower() == 'nan':
                        gdp_per_capita_val = gdp_per_capita_data.get('mean', None)
                        
                    if gdp_val is not None and gdp_per_capita_val is not None and gdp_per_capita_val > 0:
                        population_value = gdp_val / gdp_per_capita_val
                
                # If still not found, provide a default value
                if population_value == 'N/A':
                    # Default Bangladesh population (just as fallback)
                    population_value = 170000000
                
            if population_value != 'N/A':
                # Check scale - if in millions already or needs conversion
                if float(population_value) > 1000000:
                    population_display = f"{float(population_value)/1000000:.1f} million"
                else:
                    population_display = f"{float(population_value):.1f}"
            else:
                population_display = "N/A"
            
            # For Urbanization - either look for urbanization_rate or calculate from urban_population
            urbanization_data = summary_data.get('demographic', {}).get('urbanization_rate', {})
            if urbanization_data.get('end_value') is None or str(urbanization_data.get('end_value')).lower() == 'nan':
                if 'mean' in urbanization_data:
                    urbanization_value = urbanization_data.get('mean', 'N/A')
                else:
                    urbanization_value = 'N/A'
            else:
                urbanization_value = urbanization_data.get('end_value', 'N/A')
                
            if urbanization_value != 'N/A':
                urbanization_display = f"{float(urbanization_value) * 100:.1f}%"
            else:
                urbanization_display = "N/A"
            
            # For Environment - try multiple possible fields
            temp_value = 'N/A'
            
            # Try temperature field
            temp_data = summary_data.get('environmental', {}).get('temperature', {})
            if temp_data:
                if temp_data.get('end_value') is None or str(temp_data.get('end_value')).lower() == 'nan':
                    if 'mean' in temp_data:
                        temp_value = temp_data.get('mean', 'N/A')
                else:
                    temp_value = temp_data.get('end_value', 'N/A')
            
            # If not found, try temperature_anomaly
            if temp_value == 'N/A':
                temp_data = summary_data.get('environmental', {}).get('temperature_anomaly', {})
                if temp_data:
                    if temp_data.get('end_value') is None or str(temp_data.get('end_value')).lower() == 'nan':
                        if 'mean' in temp_data:
                            temp_value = temp_data.get('mean', 'N/A')
                    else:
                        temp_value = temp_data.get('end_value', 'N/A')
            
            # If still not found, try environmental_health_index
            if temp_value == 'N/A':
                env_health = summary_data.get('environmental', {}).get('environmental_health_index', {})
                if env_health:
                    if env_health.get('end_value') is None or str(env_health.get('end_value')).lower() == 'nan':
                        if 'mean' in env_health:
                            env_value = env_health.get('mean', 'N/A')
                        else:
                            env_value = 'N/A'
                    else:
                        env_value = env_health.get('end_value', 'N/A')
                    
                    if env_value != 'N/A':
                        temp_value = env_value
                        temp_display = f"{float(env_value) * 100:.1f}%"
                    else:
                        temp_display = "N/A"
                else:
                    temp_display = "N/A"
            else:
                # Regular temperature display
                if temp_value != 'N/A':
                    temp_display = f"{float(temp_value):.1f}°C"
                else:
                    temp_display = "N/A"
                
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error preparing executive summary data: {str(e)}")
            gdp_growth_display = "N/A"
            population_display = "N/A"
            urbanization_display = "N/A"
            temp_display = "N/A"
            
        return f"""
        <div class="section" id="executive-summary">
            <h2>Executive Summary</h2>
            
            <p>This simulation explores Bangladesh's development trajectory over the coming decades,
            integrating economic, demographic, environmental, infrastructure, and governance factors.</p>
            
            <div class="highlights-box">
                <h3>Key Highlights</h3>
                <div class="highlights-grid">
                    <div class="highlight-card">
                        <h4>Economy</h4>
                        <p class="highlight-value">{gdp_growth_display}</p>
                        <p class="highlight-label">Projected GDP Growth</p>
                    </div>
                    <div class="highlight-card">
                        <h4>Population</h4>
                        <p class="highlight-value">{population_display}</p>
                        <p class="highlight-label">Projected Population</p>
                    </div>
                    <div class="highlight-card">
                        <h4>Urbanization</h4>
                        <p class="highlight-value">{urbanization_display}</p>
                        <p class="highlight-label">Urban Population</p>
                    </div>
                    <div class="highlight-card">
                        <h4>Environment</h4>
                        <p class="highlight-value">{temp_display}</p>
                        <p class="highlight-label">Environmental Health</p>
                    </div>
                </div>
            </div>
            
            <p>The simulation suggests that Bangladesh will face both opportunities and challenges in the coming decades.
            Economic growth is projected to continue, with substantial improvements in infrastructure and human development.
            However, climate change impacts and environmental pressures may intensify, requiring adaptive policies
            and sustainable development approaches.</p>
        </div>
        """
    
    def _get_html_header(self, title):
        """Get HTML header with styles."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                /* Apple-inspired design system */
                :root {{
                    --primary-color: #0071e3;
                    --text-color: #1d1d1f;
                    --text-secondary: #86868b;
                    --background-color: #ffffff;
                    --section-bg: #f5f5f7;
                    --border-color: #d2d2d7;
                    --highlight-bg: #f5f5f7;
                    --shadow: rgba(0, 0, 0, 0.05);
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.5;
                    color: var(--text-color);
                    background-color: var(--background-color);
                    -webkit-font-smoothing: antialiased;
                    -moz-osx-font-smoothing: grayscale;
                }}
                
                h1, h2, h3, h4 {{
                    font-weight: 600;
                    letter-spacing: -0.015em;
                    margin-bottom: 0.8em;
                    color: var(--text-color);
                }}
                
                h1 {{
                    font-size: 2.2rem;
                    text-align: center;
                    padding: 2rem 0;
                    font-weight: 700;
                }}
                
                h2 {{
                    font-size: 1.7rem;
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 0.5rem;
                    margin-top: 2rem;
                }}
                
                h3 {{
                    font-size: 1.4rem;
                    margin-top: 1.5rem;
                }}
                
                h4 {{
                    font-size: 1.1rem;
                    margin-top: 1.2rem;
                }}
                
                p {{
                    margin-bottom: 1rem;
                    color: var(--text-color);
                    font-size: 16px;
                }}
                
                /* Navigation */
                .report-nav {{
                    background-color: rgba(255, 255, 255, 0.8);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    padding: 0.8rem 0;
                    position: sticky;
                    top: 0;
                    z-index: 100;
                    border-bottom: 1px solid var(--border-color);
                }}
                
                .report-nav ul {{
                    display: flex;
                    list-style-type: none;
                    margin: 0;
                    padding: 0;
                    justify-content: center;
                    flex-wrap: wrap;
                    max-width: 1000px;
                    margin: 0 auto;
                }}
                
                .report-nav li {{
                    margin: 0 1rem;
                }}
                
                .report-nav a {{
                    color: var(--primary-color);
                    text-decoration: none;
                    font-weight: 500;
                    padding: 0.5rem;
                    font-size: 0.9rem;
                }}
                
                .report-nav a:hover {{
                    color: #0077ED;
                }}
                
                /* Main content */
                .report-content {{
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 2rem;
                    background-color: var(--background-color);
                }}
                
                /* Sections */
                .report-section {{
                    margin-bottom: 3rem;
                    padding-bottom: 1rem;
                }}
                
                /* Key insights */
                .key-insights {{
                    background-color: var(--highlight-bg);
                    padding: 1.5rem;
                    margin: 1.5rem 0;
                    border-radius: 12px;
                }}
                
                .key-insights h3 {{
                    margin-top: 0;
                }}
                
                .key-insights ul {{
                    padding-left: 1.5rem;
                    margin-bottom: 0;
                }}
                
                .key-insights li {{
                    margin-bottom: 0.8rem;
                    line-height: 1.5;
                }}
                
                /* Tables */
                .info-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 1rem 0 2rem 0;
                    font-size: 0.95rem;
                }}
                
                .info-table th, .info-table td {{
                    padding: 0.75rem 1rem;
                    text-align: left;
                    border-bottom: 1px solid var(--border-color);
                }}
                
                .info-table th {{
                    font-weight: 600;
                    color: var(--text-secondary);
                }}
                
                /* Data cards */
                .data-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 1.5rem;
                    margin: 2rem 0;
                }}
                
                .data-card {{
                    background-color: var(--section-bg);
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                }}
                
                .data-value {{
                    font-size: 1.8rem;
                    font-weight: 600;
                    color: var(--primary-color);
                    margin: 0.5rem 0;
                }}
                
                .data-label {{
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                }}
                
                /* Visualizations */
                .visualizations {{
                    margin: 2rem 0;
                }}
                
                .viz-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 2rem;
                    margin-top: 1.5rem;
                }}
                
                .viz-item {{
                    background-color: white;
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }}
                
                .viz-item img {{
                    max-width: 100%;
                    height: auto;
                    max-height: 250px;
                    border-radius: 0;
                    object-fit: scale-down;
                    background-color: white;
                    display: block;
                    margin: 0 auto;
                }}
                
                .viz-item p {{
                    margin-top: 1rem;
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                }}
                
                /* Geographic visualizations */
                .geo-visualizations {{
                    margin: 2rem 0;
                }}
                
                .geo-category {{
                    margin-bottom: 2.5rem;
                }}
                
                .geo-viz-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 2rem;
                    margin-top: 1.5rem;
                }}
                
                .geo-viz-item {{
                    background-color: white;
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }}
                
                .geo-viz-item img {{
                    max-width: 100%;
                    height: auto;
                    max-height: 300px;
                    border-radius: 0;
                    object-fit: scale-down;
                    background-color: white;
                    display: block;
                    margin: 0 auto;
                }}
                
                .geo-viz-item p {{
                    margin-top: 1rem;
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                }}
                
                .geo-legends {{
                    margin-top: 2rem;
                }}
                
                .legend-container {{
                    display: flex;
                    justify-content: center;
                    flex-wrap: wrap;
                    gap: 1.5rem;
                }}
                
                .legend-item {{
                    text-align: center;
                    max-width: 200px;
                }}
                
                .legend-item img {{
                    max-width: 100%;
                    height: auto;
                    max-height: 120px;
                }}
                
                /* No visualizations message */
                .no-visualizations, .no-data-message {{
                    padding: 2rem;
                    background-color: var(--section-bg);
                    border-radius: 12px;
                    text-align: center;
                    color: var(--text-secondary);
                    font-style: italic;
                }}
                
                /* Metrics */
                .metrics-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                    gap: 1.5rem;
                    margin: 2rem 0;
                }}
                
                .metric-card {{
                    background-color: var(--section-bg);
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                }}
                
                .metric-value {{
                    font-size: 2rem;
                    font-weight: 600;
                    color: var(--primary-color);
                    margin: 0.5rem 0;
                }}
                
                .metric-label {{
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                    margin-top: 0.5rem;
                }}
                
                /* Footer */
                .report-footer {{
                    text-align: center;
                    padding: 2rem 0;
                    margin-top: 3rem;
                    color: var(--text-secondary);
                    font-size: 0.8rem;
                    border-top: 1px solid var(--border-color);
                }}
                
                /* Responsive adjustments */
                @media (max-width: 768px) {{
                    .viz-grid, .geo-viz-grid, .data-grid, .metrics-container {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .report-content {{
                        padding: 1.5rem;
                    }}
                    
                    .report-nav ul {{
                        flex-direction: column;
                        align-items: center;
                    }}
                    
                    .report-nav li {{
                        margin: 0.3rem 0;
                    }}
                    
                    h1 {{
                        font-size: 1.8rem;
                    }}
                    
                    h2 {{
                        font-size: 1.5rem;
                    }}
                }}
            </style>
        </head>
        <body>
        <h1>{title}</h1>
        """
    
    def _generate_section_html(self, section_name):
        """
        Generate HTML for a specific section of the report.
        
        Args:
            section_name (str): Name of the section (economic, demographic, etc.)
            
        Returns:
            str: HTML content for the section
        """
        # Get section title and description
        section_titles = {
            "economic": "Economic Development",
            "demographic": "Demographic Trends",
            "environmental": "Environmental Conditions",
            "infrastructure": "Infrastructure Development",
            "governance": "Governance and Institutions"
        }
        
        section_descriptions = {
            "economic": "Key economic indicators and trends from the simulation, including GDP growth, sector contributions, and development patterns.",
            "demographic": "Population trends, urbanization rates, and demographic shifts projected by the simulation.",
            "environmental": "Environmental factors including climate impacts, natural resource usage, and environmental sustainability metrics.",
            "infrastructure": "Infrastructure development including transportation, energy, water, and telecommunications systems.",
            "governance": "Governance indicators, institutional effectiveness, and policy implementation metrics."
        }
        
        title = section_titles.get(section_name, section_name.capitalize())
        description = section_descriptions.get(section_name, "")
        
        # Try to load summary statistics for metrics
        try:
            summary_data_path = Path(self.sim_dir) / 'summary_statistics.json'
            if summary_data_path.exists():
                with open(summary_data_path, 'r') as f:
                    summary_stats = json.load(f)
                    section_data = summary_stats.get(section_name, {})
            else:
                section_data = {}
        except Exception:
            section_data = {}
            
        # Start section HTML
        html = f"""
        <section id="{section_name}" class="report-section">
            <h2>{title}</h2>
            <p>{description}</p>
        """
        
        # Add key metrics if available
        if section_data:
            metrics_to_display = self._get_display_metrics(section_name, section_data)
            
            if metrics_to_display:
                html += """
                <div class="metrics-container">
                """
                
                for metric_key, metric_name in metrics_to_display:
                    if metric_key in section_data:
                        metric_data = section_data[metric_key]
                        # Try to get end_value first, then mean if end_value is not available
                        if metric_data.get('end_value') is None or str(metric_data.get('end_value')).lower() == 'nan':
                            if 'mean' in metric_data:
                                metric_value = self._format_value(metric_data.get('mean', 'N/A'), metric_key)
                            else:
                                metric_value = "N/A"
                        else:
                            metric_value = self._format_value(metric_data.get('end_value', 'N/A'), metric_key)
                            
                        html += f"""
                        <div class="metric-card">
                            <div class="metric-label">{metric_name}</div>
                            <div class="metric-value">{metric_value}</div>
                        </div>
                        """
                
                html += """
                </div>
                """
        
        # Add visualizations
        html += self._add_visualizations(section_name)
        
        # Close section
        html += "</section>"
        
        return html
    
    def _generate_methodology_section(self):
        """Generate a methodology section explaining the simulation for general audience."""
        return """
        <div class="section" id="methodology">
            <h2>Methodology</h2>
            
            <p>This report is based on a complex computer simulation that models how different aspects of 
            Bangladesh's development interact and evolve over time. The simulation is designed to help understand 
            possible future scenarios rather than make precise predictions.</p>
            
            <div class="methodology-grid">
                <div class="methodology-item">
                    <h3>What is a Simulation?</h3>
                    <p>A simulation is a computer model that tries to represent real-world systems and their 
                    behaviors. It helps us explore "what if" scenarios and understand complex interactions 
                    between different factors.</p>
                </div>
                
                <div class="methodology-item">
                    <h3>Systems Modeled</h3>
                    <p>Our simulation incorporates five interconnected systems:</p>
                    <ul>
                        <li>Economic system</li>
                        <li>Demographic system</li>
                        <li>Environmental system</li>
                        <li>Infrastructure system</li>
                        <li>Governance system</li>
                    </ul>
                </div>
                
                <div class="methodology-item">
                    <h3>Data Sources</h3>
                    <p>The simulation uses historical data from:</p>
                    <ul>
                        <li>Bangladesh Bureau of Statistics</li>
                        <li>World Bank</li>
                        <li>United Nations</li>
                        <li>IPCC climate projections</li>
                        <li>Other international organizations</li>
                    </ul>
                </div>
                
                <div class="methodology-item">
                    <h3>Limitations</h3>
                    <p>All simulations have limitations:</p>
                    <ul>
                        <li>They cannot predict unexpected events or disruptions</li>
                        <li>They rely on assumptions that may not hold true</li>
                        <li>They simplify complex real-world systems</li>
                    </ul>
                </div>
            </div>
            
            <div class="info-box">
                <h3>Understanding Uncertainty</h3>
                <p>The future is inherently uncertain. This simulation shows one possible future trajectory based on current 
                trends and relationships. The further into the future we look, the greater the uncertainty becomes.</p>
                <p>Results should be interpreted as illustrative of potential developments rather than definitive predictions.</p>
            </div>
        </div>
        """
    
    def _get_display_metrics(self, section, section_data):
        """Get list of metrics to display for each section."""
        if section == 'economic':
            return [
                ('gdp', 'GDP'), 
                ('gdp_per_capita', 'GDP per Capita'),
                ('gdp_growth', 'GDP Growth Rate'),
                ('inflation_rate', 'Inflation Rate'),
                ('unemployment_rate', 'Unemployment Rate')
            ]
        elif section == 'demographic':
            return [
                ('total_population', 'Total Population'),
                ('urbanization_rate', 'Urbanization Rate'),
                ('literacy_rate', 'Literacy Rate'),
                ('fertility_rate', 'Fertility Rate'),
                ('life_expectancy', 'Life Expectancy')
            ]
        elif section == 'environmental':
            return [
                ('forest_cover', 'Forest Cover'),
                ('air_quality_index', 'Air Quality Index'),
                ('water_quality_index', 'Water Quality Index'),
                ('temperature_anomaly', 'Temperature Anomaly'),
                ('extreme_event_impact', 'Extreme Event Impact')
            ]
        elif section == 'infrastructure':
            return [
                ('electricity_coverage', 'Electricity Coverage'),
                ('water_supply_coverage', 'Water Supply Coverage'),
                ('internet_coverage', 'Internet Coverage'),
                ('renewable_energy_share', 'Renewable Energy Share'),
                ('infrastructure_quality_index', 'Quality Index')
            ]
        elif section == 'governance':
            return [
                ('governance_index', 'Governance Index'),
                ('institutional_effectiveness', 'Institutional Effectiveness'),
                ('corruption_index', 'Corruption Index'),
                ('regulatory_quality', 'Regulatory Quality'),
                ('policy_effectiveness', 'Policy Effectiveness')
            ]
        else:
            # Default: return all numeric metrics
            return [(k, k.replace('_', ' ').title()) for k, v in section_data.items()
                   if isinstance(v, dict) and 'end_value' in v]
    
    def _format_value(self, value, metric):
        """Format values appropriately based on metric name."""
        try:
            # Handle None, NaN, or 'nan' string values
            if value is None or (isinstance(value, str) and value.lower() in ['nan', 'none', 'null']):
                return "Not available"
                
            # Convert string to number if possible
            if isinstance(value, str):
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    # If it can't be converted to a number, check if it's NaN
                    if value.lower() in ['nan', 'none', 'null']:
                        return "Not available"
                    return value
            
            # Check for NaN or infinity using multiple methods
            if isinstance(value, float):
                if np.isnan(value) or pd.isna(value) or np.isinf(value):
                    return "Not available"
            
            # Format based on likely metric type
            if 'rate' in metric.lower() or 'percentage' in metric.lower() or 'share' in metric.lower():
                # Percentage values
                return f"{float(value) * 100:.2f}%"
            elif 'population' in metric.lower():
                # Population values (in millions)
                pop_value = float(value)
                if pop_value > 1000000:
                    return f"{pop_value/1000000:.2f}M"
                else:
                    return f"{int(pop_value):,}"
            elif 'gdp' in metric.lower() and 'growth' not in metric.lower() and 'capita' not in metric.lower():
                # GDP values (in billions)
                gdp_value = float(value)
                if gdp_value > 1000000000:
                    return f"${gdp_value/1000000000:.2f}B"
                elif gdp_value > 1000000:
                    return f"${gdp_value/1000000:.2f}M"
                else:
                    return f"${gdp_value:,.2f}"
            elif 'capita' in metric.lower():
                # Per capita values
                return f"${float(value):,.2f}"
            elif 'index' in metric.lower():
                # Index values (usually 0-1)
                return f"{float(value):.3f}"
            else:
                # Default numeric formatting
                float_val = float(value)
                if float_val == int(float_val):
                    return f"{int(float_val):,}"
                else:
                    return f"{float_val:,.2f}"
        except Exception as e:
            # If any errors, return a formatted message instead of the original value
            return "Not available"
    
    def _add_visualizations(self, section_name):
        """
        Add visualizations for a specific section.
        
        Args:
            section_name (str): Name of the section (economic, demographic, etc.)
            
        Returns:
            str: HTML content with visualizations
        """
        # Path to visualizations directory
        plots_dir = Path(self.sim_dir) / 'plots' / section_name
        
        # Check if directory exists
        if not plots_dir.exists():
            return f"""
            <div class="no-visualizations">
                <p>No visualizations available for this section.</p>
            </div>
            """
        
        # Find all PNG files in the directory
        plot_files = list(plots_dir.glob('*.png'))
        
        # If no plots found
        if not plot_files:
            return f"""
            <div class="no-visualizations">
                <p>No visualizations available for this section.</p>
            </div>
            """
        
        # Start visualizations section
        html = """
        <div class="visualizations">
            <h3>Visualizations</h3>
            <div class="viz-grid">
        """
        
        # Filter out composite plots to show them at the end
        regular_plots = [p for p in plot_files if 'composite' not in p.name.lower()]
        composite_plots = [p for p in plot_files if 'composite' in p.name.lower()]
        
        # Sort plots so they appear in a logical order
        regular_plots.sort(key=lambda x: x.name)
        
        # Show regular plots first, then composite plots
        display_plots = regular_plots + composite_plots
        
        # Add each plot, limited to 6 for simplicity
        for plot_file in display_plots[:6]:
            # Format the name cleanly
            plot_name = plot_file.stem.replace('_', ' ').title()
            rel_path = os.path.relpath(str(plot_file), os.path.dirname(self.report_path))
            
            html += f"""
            <div class="viz-item">
                <img src="{rel_path}" alt="{plot_name}" />
                <p>{plot_name}</p>
            </div>
            """
        
        # Close visualizations section
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_geography_section(self):
        """Generate the geography section of the report with maps and geographic visualizations."""
        section_html = f"""
        <section id="geography" class="report-section">
            <h2>Geographic Analysis</h2>
            <p>This section provides geographic analysis of Bangladesh's development patterns, highlighting regional variations
            and spatial relationships between development factors.</p>
            
            <div class="key-insights">
                <h3>Key Geographic Insights</h3>
                <ul>
                    <li>Economic development varies significantly by region, with highest growth in urban centers.</li>
                    <li>Coastal regions show distinct development patterns linked to climate vulnerability.</li>
                    <li>Infrastructure quality varies by administrative division, affecting development outcomes.</li>
                    <li>Population density patterns create different development challenges across regions.</li>
                </ul>
            </div>
            
            {self._add_simplified_geo_visualizations()}
        </section>
        """
        return section_html
    
    def _add_simplified_geo_visualizations(self):
        """Add geographic visualizations to the report in a simplified format."""
        # Check if geo_viz_paths is available from generate_report
        has_geo_viz_paths = hasattr(self, 'geo_viz_paths') and self.geo_viz_paths
        
        content = """
        <div class="geo-visualizations">
            <h3>Geographic Distribution</h3>
        """
        
        # Path to maps directory
        maps_dir = Path(self.sim_dir) / 'maps'
        
        # If we don't have geo_viz_paths and maps directory doesn't exist
        if not has_geo_viz_paths and not maps_dir.exists():
            content += """
            <p class="no-data-message">No geographic visualizations are available for this simulation run.</p>
            </div>
            """
            return content
        
        # Make sure self.report_path is set
        if not hasattr(self, 'report_path') or self.report_path is None:
            self.report_path = str(self.sim_dir / "simulation_report.html")
        
        # If we have geo_viz_paths use them directly
        if has_geo_viz_paths:
            # Prioritize choropleth and bubble maps for display (avoid animations)
            choropleth_maps = []
            bubble_maps = []
            heat_maps = []
            bivariate_maps = []
            
            for map_type, path in self.geo_viz_paths.items():
                if path is None or 'animation' in map_type:
                    continue
                    
                if 'choropleth' in map_type or 'gdp_by_division' in map_type or 'resilience' in map_type:
                    choropleth_maps.append(path)
                elif 'bubble' in map_type:
                    bubble_maps.append(path)
                elif 'heat' in map_type:
                    heat_maps.append(path)
                elif 'bivariate' in map_type:
                    bivariate_maps.append(path)
        else:
            # Search for map files
            map_files = list(maps_dir.glob('*.png'))
            if not map_files:
                content += """
                <p class="no-data-message">No map visualizations found in the maps directory.</p>
                </div>
                """
                return content
            
            # Categorize maps based on their filename, excluding animations
            choropleth_maps = [str(f) for f in map_files if any(x in f.name for x in ['choropleth', 'gdp_by_division', 'resilience']) and 'animation' not in f.name]
            bubble_maps = [str(f) for f in map_files if 'bubble' in f.name and 'animation' not in f.name]
            heat_maps = [str(f) for f in map_files if 'heat' in f.name and 'animation' not in f.name]
            bivariate_maps = [str(f) for f in map_files if 'bivariate' in f.name and 'animation' not in f.name]
        
        # Combine all maps for a simplified display
        all_maps = choropleth_maps + bubble_maps + heat_maps + bivariate_maps
        
        # Limit to top 6 maps for simplicity
        selected_maps = all_maps[:6]
        
        if selected_maps:
            content += """
            <div class="geo-viz-grid">
            """
            
            for i, map_path in enumerate(selected_maps):
                if map_path is None:
                    continue
                    
                try:
                    rel_path = os.path.relpath(map_path, os.path.dirname(self.report_path))
                    map_name = os.path.basename(map_path).replace('_', ' ').replace('.png', '')
                    content += f"""
                    <div class="geo-viz-item">
                        <img src="{rel_path}" alt="{map_name}" />
                        <p>{map_name}</p>
                    </div>
                    """
                except (TypeError, ValueError) as e:
                    # Skip this map if there's an error
                    continue
            
            content += """
            </div>
            """
        else:
            content += """
            <p class="no-data-message">No geographic visualizations are available for display.</p>
            """
        
        content += """
        </div>
        """
        
        return content
    
    def _embed_image(self, image_path, title):
        """Embed an image into the HTML."""
        try:
            # Read image file and convert to base64
            with open(image_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
            return f"""
            <div class="visualization">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{img_data}" alt="{title}">
            </div>
            """
        except Exception as e:
            return f"""
            <div class="visualization">
                <h3>{title}</h3>
                <p>Error loading image: {str(e)}</p>
            </div>
            """
    
    def _generate_overview_section(self):
        """Generate the overview section of the report."""
        # Try to read summary statistics for overview cards
        try:
            summary_data_path = Path(self.sim_dir) / 'summary_statistics.json'
            if summary_data_path.exists():
                with open(summary_data_path, 'r') as f:
                    summary_data = json.load(f)
            else:
                summary_data = {}
        except Exception:
            summary_data = {}
            
        # Extract key metrics for overview display
        gdp_growth = "N/A"
        population = "N/A"
        env_index = "N/A"
        infra_index = "N/A"
        
        if summary_data.get('economic', {}).get('gdp_growth', {}):
            gdp_value = summary_data['economic']['gdp_growth'].get('mean', 
                       summary_data['economic']['gdp_growth'].get('end_value', 'N/A'))
            if gdp_value != 'N/A' and gdp_value is not None:
                gdp_growth = f"{float(gdp_value) * 100:.1f}%"
                
        if summary_data.get('demographic', {}).get('total_population', {}):
            pop_value = summary_data['demographic']['total_population'].get('end_value',
                       summary_data['demographic']['total_population'].get('mean', 'N/A'))
            if pop_value != 'N/A' and pop_value is not None:
                if float(pop_value) > 1000000:
                    population = f"{float(pop_value)/1000000:.1f}M"
                else:
                    population = f"{int(float(pop_value)):,}"
        
        if summary_data.get('environmental', {}).get('environmental_health_index', {}):
            env_value = summary_data['environmental']['environmental_health_index'].get('end_value',
                       summary_data['environmental']['environmental_health_index'].get('mean', 'N/A'))
            if env_value != 'N/A' and env_value is not None:
                env_index = f"{float(env_value) * 100:.1f}%"
        
        if summary_data.get('infrastructure', {}).get('infrastructure_quality_index', {}):
            infra_value = summary_data['infrastructure']['infrastructure_quality_index'].get('end_value',
                         summary_data['infrastructure']['infrastructure_quality_index'].get('mean', 'N/A'))
            if infra_value != 'N/A' and infra_value is not None:
                infra_index = f"{float(infra_value) * 100:.1f}%"
                
        section_html = f"""
        <section id="overview" class="report-section">
            <h2>Overview</h2>
            <p>This report presents the results of the Bangladesh Development Simulation model, 
            which projects potential scenarios for Bangladesh's socioeconomic and environmental development.</p>
            
            <div class="data-grid">
                <div class="data-card">
                    <div class="data-label">GDP Growth</div>
                    <div class="data-value">{gdp_growth}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Population</div>
                    <div class="data-value">{population}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Environmental Index</div>
                    <div class="data-value">{env_index}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Infrastructure Quality</div>
                    <div class="data-value">{infra_index}</div>
                </div>
            </div>
            
            <div class="key-insights">
                <h3>Key Insights</h3>
                <ul>
                    <li>The simulation integrates five key systems: economic, demographic, environmental, infrastructure, 
                    and governance to capture complex interactions between development factors.</li>
                    <li>Geographic analysis reveals regional development patterns that require targeted policy approaches.</li>
                    <li>Results show correlations between governance effectiveness and economic growth outcomes.</li>
                    <li>Climate impacts vary significantly across regions, with coastal areas showing higher vulnerability.</li>
                </ul>
            </div>
            
            <div class="simulation-info">
                <h3>Simulation Information</h3>
                <table class="info-table">
                    <tr>
                        <th>Simulation ID</th>
                        <td>{self.simulation_id}</td>
                    </tr>
                    <tr>
                        <th>Date Generated</th>
                        <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                    </tr>
                    <tr>
                        <th>Simulation Period</th>
                        <td>10 years</td>
                    </tr>
                </table>
            </div>
        </section>
        """
        return section_html 