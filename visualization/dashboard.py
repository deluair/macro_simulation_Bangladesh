"""
Interactive dashboard for the Bangladesh Development Simulation Model.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os

class Dashboard:
    """Class for creating interactive dashboards."""
    
    def __init__(self, history: List[Dict[str, Any]], output_dir: str = 'results/dashboard'):
        """
        Initialize the dashboard.
        
        Args:
            history: List of simulation states
            output_dir: Directory to save dashboard files
        """
        self.history = history
        self.df = pd.DataFrame(history)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1('Bangladesh Development Simulation Dashboard'),
            
            # Development Trajectory
            html.Div([
                html.H2('Development Trajectory'),
                dcc.Graph(id='development-trajectory')
            ]),
            
            # Economic Indicators
            html.Div([
                html.H2('Economic Indicators'),
                dcc.Dropdown(
                    id='economic-indicator',
                    options=[
                        {'label': 'GDP by Sector', 'value': 'sector_gdp'},
                        {'label': 'Employment Rate', 'value': 'employment_rate'},
                        {'label': 'Trade Balance', 'value': 'trade_balance'}
                    ],
                    value='sector_gdp'
                ),
                dcc.Graph(id='economic-plot')
            ]),
            
            # Environmental Indicators
            html.Div([
                html.H2('Environmental Indicators'),
                dcc.Dropdown(
                    id='environmental-indicator',
                    options=[
                        {'label': 'Climate Indicators', 'value': 'climate'},
                        {'label': 'Environmental Health', 'value': 'health'},
                        {'label': 'Resource Use', 'value': 'resources'}
                    ],
                    value='climate'
                ),
                dcc.Graph(id='environmental-plot')
            ]),
            
            # Demographic Indicators
            html.Div([
                html.H2('Demographic Indicators'),
                dcc.Dropdown(
                    id='demographic-indicator',
                    options=[
                        {'label': 'Population Structure', 'value': 'structure'},
                        {'label': 'Social Indicators', 'value': 'social'},
                        {'label': 'Education', 'value': 'education'}
                    ],
                    value='structure'
                ),
                dcc.Graph(id='demographic-plot')
            ]),
            
            # Infrastructure Indicators
            html.Div([
                html.H2('Infrastructure Indicators'),
                dcc.Dropdown(
                    id='infrastructure-indicator',
                    options=[
                        {'label': 'Physical Infrastructure', 'value': 'physical'},
                        {'label': 'Quality Indicators', 'value': 'quality'},
                        {'label': 'Access Indicators', 'value': 'access'}
                    ],
                    value='physical'
                ),
                dcc.Graph(id='infrastructure-plot')
            ]),
            
            # Governance Indicators
            html.Div([
                html.H2('Governance Indicators'),
                dcc.Dropdown(
                    id='governance-indicator',
                    options=[
                        {'label': 'Institutional Quality', 'value': 'institutional'},
                        {'label': 'Policy Effectiveness', 'value': 'policy'},
                        {'label': 'Social Progress', 'value': 'social'}
                    ],
                    value='institutional'
                ),
                dcc.Graph(id='governance-plot')
            ]),
            
            # Correlation Matrix
            html.Div([
                html.H2('Correlation Matrix'),
                dcc.Graph(id='correlation-matrix')
            ])
        ])
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            Output('development-trajectory', 'figure'),
            Input('development-trajectory', 'id')
        )
        def update_development_trajectory(_):
            fig = go.Figure()
            
            for indicator in ['development_index', 'sustainability_index', 
                            'resilience_index', 'wellbeing_index']:
                fig.add_trace(go.Scatter(
                    x=self.df.index,
                    y=self.df[indicator],
                    name=indicator.replace('_', ' ').title()
                ))
            
            fig.update_layout(
                title='Development Trajectory Over Time',
                xaxis_title='Time Step',
                yaxis_title='Index Value',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('economic-plot', 'figure'),
            Input('economic-indicator', 'value')
        )
        def update_economic_plot(indicator):
            if indicator == 'sector_gdp':
                fig = go.Figure()
                for sector in ['garment_gdp', 'agriculture_gdp', 'remittances_gdp', 
                              'tech_gdp', 'informal_gdp']:
                    fig.add_trace(go.Scatter(
                        x=self.df.index,
                        y=self.df[sector],
                        name=sector.replace('_gdp', '').title()
                    ))
                fig.update_layout(
                    title='GDP by Sector',
                    xaxis_title='Time Step',
                    yaxis_title='GDP (USD)',
                    hovermode='x unified'
                )
            elif indicator == 'employment_rate':
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=self.df.index,
                    y=self.df['unemployment_rate'],
                    name='Unemployment Rate'
                ))
                fig.update_layout(
                    title='Unemployment Rate Over Time',
                    xaxis_title='Time Step',
                    yaxis_title='Rate',
                    hovermode='x unified'
                )
            else:  # trade_balance
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=self.df.index,
                    y=self.df['trade_balance'],
                    name='Trade Balance'
                ))
                fig.update_layout(
                    title='Trade Balance Over Time',
                    xaxis_title='Time Step',
                    yaxis_title='Balance (USD)',
                    hovermode='x unified'
                )
            
            return fig
        
        # Add similar callbacks for other indicators...
    
    def run(self, debug: bool = True, port: int = 8050):
        """
        Run the dashboard.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        self.app.run_server(debug=debug, port=port)
    
    def save(self):
        """Save the dashboard as a static HTML file."""
        self.app.to_html(os.path.join(self.output_dir, 'dashboard.html')) 