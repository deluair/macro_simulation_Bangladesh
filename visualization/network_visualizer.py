"""
Network visualization module for the Bangladesh Development Simulation Model.
Contains tools for creating network visualizations of system interconnections and flows.
"""

import os
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class NetworkVisualizer:
    """Class for creating network visualizations of system interactions."""
    
    def __init__(self, output_dir: str = 'results/networks'):
        """
        Initialize the network visualizer.
        
        Args:
            output_dir: Directory to save network visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_system_interaction_graph(self, 
                                      interactions: Dict[str, List[str]], 
                                      title: str,
                                      filename: str,
                                      edge_weights: Dict[Tuple[str, str], float] = None,
                                      node_sizes: Dict[str, float] = None) -> str:
        """
        Create a network visualization showing interactions between systems.
        
        Args:
            interactions: Dictionary mapping system names to lists of systems they interact with
            title: Chart title
            filename: Filename to save the visualization
            edge_weights: Optional dictionary mapping (source, target) tuples to interaction strengths
            node_sizes: Optional dictionary mapping node names to sizes
            
        Returns:
            Path to the saved visualization file
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for source, targets in interactions.items():
            if source not in G:
                G.add_node(source)
            
            for target in targets:
                if target not in G:
                    G.add_node(target)
                
                # Add edge with weight if provided
                if edge_weights and (source, target) in edge_weights:
                    G.add_edge(source, target, weight=edge_weights[(source, target)])
                else:
                    G.add_edge(source, target, weight=1.0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Network layout
        pos = nx.spring_layout(G, seed=42)
        
        # Node sizes
        if node_sizes:
            node_size = [node_sizes.get(node, 300) for node in G.nodes()]
        else:
            node_size = 300
        
        # Edge weights for width
        if edge_weights:
            edge_width = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        else:
            edge_width = 1.0
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, edge_color='gray', 
                               connectionstyle='arc3,rad=0.1', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Remove axis
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_flow_diagram(self, 
                          flows: Dict[str, Dict[str, float]], 
                          title: str,
                          filename: str,
                          node_colors: Dict[str, str] = None) -> str:
        """
        Create a Sankey or flow diagram showing resource, information, or value flows.
        
        Args:
            flows: Dictionary mapping source nodes to dictionaries mapping target nodes to flow values
            title: Chart title
            filename: Filename to save the visualization
            node_colors: Optional dictionary mapping node names to colors
            
        Returns:
            Path to the saved visualization file
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges with weights
        for source, targets in flows.items():
            if source not in G:
                G.add_node(source)
            
            for target, value in targets.items():
                if target not in G:
                    G.add_node(target)
                G.add_edge(source, target, weight=value)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Network layout - hierarchical for flow diagrams
        pos = nx.multipartite_layout(G, subset_key=lambda x: 0 if x in flows else 1)
        
        # Customize node colors if provided
        if node_colors:
            colors = [node_colors.get(node, 'skyblue') for node in G.nodes()]
        else:
            colors = 'skyblue'
        
        # Get edge weights for width
        edge_width = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        max_width = max(edge_width) if edge_width else 1
        edge_width = [max(1, w * 10 / max_width) for w in edge_width]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=colors, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7, edge_color='gray', 
                               connectionstyle='arc3,rad=0.1', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Add edge labels (flow values)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        # Remove axis
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
