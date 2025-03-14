"""
Geographic visualization module for the Bangladesh Development Simulation Model.
Contains tools for creating maps and spatial visualizations of simulation results.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

class GeoVisualizer:
    """Class for creating geographic visualizations."""
    
    def __init__(self, output_dir: str = 'results/maps'):
        """
        Initialize the geographic visualizer.
        
        Args:
            output_dir: Directory to save map visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Bangladesh shapefile paths
        self.bd_shapefile_paths = {
            'country': 'data/shapefiles/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm0_bbs_20201113.shp',
            'division': 'data/shapefiles/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm1_bbs_20201113.shp',
            'district': 'data/shapefiles/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm2_bbs_20201113.shp',
            'upazila': 'data/shapefiles/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm3_bbs_20201113.shp',
            'union': 'data/shapefiles/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm4_bbs_20201113.shp',
            'all_admin': 'data/shapefiles/bgd_adm_bbs_20201113_SHP/bgd_admbndl_admALL_bbs_itos_20201113.shp'
        }
        
        # Load shapefiles for different administrative levels
        self.bd_shapefiles = {}
        for level, path in self.bd_shapefile_paths.items():
            shapefile_path = Path(path)
            if shapefile_path.exists():
                try:
                    self.bd_shapefiles[level] = gpd.read_file(shapefile_path)
                    print(f"Loaded Bangladesh {level} shapefile from {shapefile_path}")
                except Exception as e:
                    print(f"Warning: Could not load Bangladesh {level} shapefile: {e}")
        
        # If we have at least one shapefile, use it as the default
        if self.bd_shapefiles:
            # Prefer division level for most visualizations, or use whatever is available
            if 'division' in self.bd_shapefiles:
                self.bd_shapefile = self.bd_shapefiles['division']
            else:
                level, gdf = next(iter(self.bd_shapefiles.items()))
                self.bd_shapefile = gdf
                print(f"Using {level} shapefile as default")
        else:
            # Fallback to the existing GeoJSON if no shapefiles are loaded
            self.bd_shapefile = None
            geojson_path = Path('data/gis/bangladesh_detailed.geojson')
            if geojson_path.exists():
                try:
                    self.bd_shapefile = gpd.read_file(geojson_path)
                    print(f"Loaded Bangladesh GeoJSON from {geojson_path}")
                except Exception as e:
                    print(f"Warning: Could not load Bangladesh GeoJSON: {e}")
            
            if self.bd_shapefile is None:
                # Final fallback to bd_divisions.geojson
                geojson_path = Path('data/gis/bd_divisions.geojson')
                if geojson_path.exists():
                    try:
                        self.bd_shapefile = gpd.read_file(geojson_path)
                        print(f"Loaded Bangladesh GeoJSON from {geojson_path}")
                    except Exception as e:
                        print(f"Warning: Could not load Bangladesh GeoJSON: {e}")
        
        # Load major cities data if available
        self.cities_data = None
        cities_path = Path('data/gis/bd_major_cities.csv')
        if cities_path.exists():
            try:
                self.cities_data = pd.read_csv(cities_path)
            except Exception as e:
                print(f"Warning: Could not load Bangladesh cities data: {e}")
                
        # Create sample cities data if not available
        if self.cities_data is None:
            # Create sample city data for Bangladesh
            self.cities_data = pd.DataFrame({
                'city': ['Dhaka', 'Chittagong', 'Khulna', 'Rajshahi', 'Sylhet', 'Barisal', 'Rangpur'],
                'lat': [23.8103, 22.3569, 22.8456, 24.3745, 24.8949, 22.7010, 25.7439],
                'lon': [90.4125, 91.7832, 89.5403, 88.6042, 91.8687, 90.3535, 89.2752],
                'population': [8906039, 2581643, 663342, 449756, 526412, 328278, 294265]
            })
    
    def create_choropleth(self, 
                         data: Dict[str, float], 
                         title: str,
                         filename: str,
                         cmap: str = 'viridis',
                         admin_level: str = 'division',
                         legend_title: str = None) -> str:
        """
        Create a choropleth map of Bangladesh showing data by administrative units.
        
        Args:
            data: Dictionary mapping admin unit names to values
            title: Map title
            filename: Filename to save the map
            cmap: Matplotlib colormap name
            admin_level: Administrative level ('division', 'district', 'upazila', 'union')
            legend_title: Title for the legend
            
        Returns:
            Path to the saved map file
        """
        # Use the specified admin level shapefile if available
        if admin_level in self.bd_shapefiles:
            gdf = self.bd_shapefiles[admin_level].copy()
            print(f"Using {admin_level} shapefile for choropleth map")
        elif self.bd_shapefile is not None:
            gdf = self.bd_shapefile.copy()
            print(f"Using default shapefile for choropleth map")
        else:
            print("Warning: Bangladesh shapefile not available. Skipping choropleth map creation.")
            return None
        
        # Find the name column for this admin level
        admin_col = None
        possible_name_columns = [
            'ADM1_EN', 'ADM1_PCODE', 'ADM1_REF',  # Division
            'ADM2_EN', 'ADM2_PCODE', 'ADM2_REF',  # District
            'ADM3_EN', 'ADM3_PCODE', 'ADM3_REF',  # Upazila
            'ADM4_EN', 'ADM4_PCODE', 'ADM4_REF',  # Union
            'division', 'district', 'upazila', 'union',
            'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4',
            'NAME', 'Division', 'DIVISION', 'name'
        ]
        
        for col in possible_name_columns:
            if col in gdf.columns:
                admin_col = col
                print(f"Using column '{admin_col}' for admin unit names")
                break
        
        if admin_col is None:
            print(f"Warning: No suitable admin name column found in shapefile. Using first column.")
            admin_col = gdf.columns[0]  # Use first column as fallback
        
        # Convert dictionary to Series for joining
        data_series = pd.Series(data)
        
        # Check if any admin names from data match those in the shapefile
        if admin_col in gdf.columns:
            matching_names = set(data.keys()) & set(gdf[admin_col])
            if not matching_names:
                print(f"Warning: No matching admin unit names found between data and shapefile.")
                # Print available names in shapefile for debugging
                print(f"Available names in shapefile: {gdf[admin_col].unique()[:5]}...")
                print(f"Names in data: {list(data.keys())}")
                
                # Try to normalize names for better matching
                data_normalized = {k.lower().replace(' ', '_'): v for k, v in data.items()}
                if isinstance(gdf[admin_col].iloc[0], str):
                    gdf_normalized = gdf[admin_col].str.lower().str.replace(' ', '_')
                    matching_normalized = set(data_normalized.keys()) & set(gdf_normalized)
                    
                    if matching_normalized:
                        print(f"Found {len(matching_normalized)} matches after normalization.")
                        # Create mapping from normalized to original names
                        norm_to_orig = {k.lower().replace(' ', '_'): k for k in data.keys()}
                        # Update GDF with normalized names for joining
                        gdf['normalized'] = gdf_normalized
                        admin_col = 'normalized'
                        # Update data to use normalized keys
                        data_series = pd.Series({k: data[norm_to_orig[k]] for k in matching_normalized})
        
        # Join data to geodataframe
        gdf = gdf.set_index(admin_col)
        gdf['value'] = data_series
        gdf = gdf.reset_index()
        
        # Create the map
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        
        # Plot the map with improved styling
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        gdf.plot(column='value', 
                ax=ax, 
                legend=True,
                cmap=cmap, 
                missing_kwds={'color': 'lightgrey'},
                legend_kwds={'label': legend_title if legend_title else "Value"},
                cax=cax)
        
        # Add title and adjust layout
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        
        # Add country borders with higher weight
        if 'geometry' in gdf.columns:
            gdf.boundary.plot(ax=ax, linewidth=0.5, color='black')
        
        # Save the map
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def get_admin_shapefile(self, admin_level='division'):
        """
        Get the shapefile for a specific administrative level.
        
        Args:
            admin_level: Administrative level ('country', 'division', 'district', 'upazila', 'union')
            
        Returns:
            GeoDataFrame for the specified admin level
        """
        if admin_level in self.bd_shapefiles:
            return self.bd_shapefiles[admin_level].copy()
        elif self.bd_shapefile is not None:
            return self.bd_shapefile.copy()
        return None
    
    def create_regional_comparison(self, 
                                 data: Dict[str, Dict[str, float]], 
                                 title: str,
                                 filename: str,
                                 regions: List[str] = None,
                                 metrics: List[str] = None) -> str:
        """
        Create a comparative visualization of multiple metrics across regions.
        
        Args:
            data: Nested dict mapping regions to metrics to values
            title: Chart title
            filename: Filename to save the chart
            regions: List of regions to include (if None, use all)
            metrics: List of metrics to include (if None, use all)
            
        Returns:
            Path to the saved chart file
        """
        # Prepare data
        if regions is None:
            regions = list(data.keys())
        
        if metrics is None:
            # Get all unique metrics across all regions
            metrics = set()
            for region_data in data.values():
                metrics.update(region_data.keys())
            metrics = sorted(metrics)
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(index=regions, columns=metrics)
        for region in regions:
            if region in data:
                for metric in metrics:
                    if metric in data[region]:
                        df.loc[region, metric] = data[region][metric]
        
        # Replace NaN with 0
        df = df.fillna(0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a bar chart
        x = np.arange(len(regions))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            offset = i - len(metrics) / 2 + 0.5
            ax.bar(x + offset * width, df[metric], width, label=metric)
        
        # Add labels and legend
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right')
        ax.legend()
        
        # Save the chart
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    def create_bubble_map(self,
                       location_data: List[Dict[str, Any]],
                       title: str,
                       filename: str,
                       size_field: str = 'value',
                       color_field: str = None,
                       label_field: str = 'name',
                       size_scale: float = 100.0,
                       cmap: str = 'viridis',
                       add_labels: bool = True,
                       admin_level: str = 'country') -> str:
        """
        Create a bubble map showing point data with variable size and color.
        
        Args:
            location_data: List of dicts with 'lat', 'lon', size_field, and optionally color_field and label_field
            title: Map title
            filename: Filename to save the map
            size_field: Field to use for bubble size
            color_field: Field to use for bubble color (optional)
            label_field: Field to use for labels
            size_scale: Scaling factor for bubble sizes
            cmap: Colormap to use for color_field
            add_labels: Whether to add labels for each bubble
            admin_level: Administrative level to use for the background map
            
        Returns:
            Path to the saved map file
        """
        # Get the appropriate shapefile for the background
        if admin_level in self.bd_shapefiles:
            bg_shapefile = self.bd_shapefiles[admin_level]
            print(f"Using {admin_level} shapefile for bubble map background")
        elif 'country' in self.bd_shapefiles:
            bg_shapefile = self.bd_shapefiles['country']
            print(f"Using country shapefile for bubble map background")
        elif self.bd_shapefile is not None:
            bg_shapefile = self.bd_shapefile
            print(f"Using default shapefile for bubble map background")
        else:
            bg_shapefile = None
            print("Warning: Bangladesh shapefile not available. Using basic background for bubble map.")
            
        # Create the map figure
        fig, ax = plt.subplots(figsize=(10, 12))
        
        if bg_shapefile is not None:
            # Create map with Bangladesh shapefile as background
            bg_shapefile.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
        else:
            # Create a simple background map of Bangladesh with approximate bounds
            ax.set_xlim([88.0, 92.7])  # Approximate longitude bounds of Bangladesh
            ax.set_ylim([20.5, 26.7])  # Approximate latitude bounds of Bangladesh
        
        # Extract data for plotting
        lats = [loc['lat'] for loc in location_data]
        lons = [loc['lon'] for loc in location_data]
        sizes = [loc[size_field] for loc in location_data]
        labels = [loc.get(label_field, '') for loc in location_data]
        
        # Normalize sizes for better visualization
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 1
        norm_sizes = [((s - min_size) / (max_size - min_size) * 0.8 + 0.2) * size_scale 
                     for s in sizes]
        
        # Handle coloring
        if color_field:
            colors = [loc.get(color_field, 0) for loc in location_data]
            # Create colormap
            norm = Normalize(vmin=min(colors), vmax=max(colors))
            cmap_obj = cm.get_cmap(cmap)
            
            # Plot with colors
            scatter = ax.scatter(lons, lats, s=norm_sizes, c=colors, cmap=cmap_obj, 
                               alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(color_field.capitalize())
        else:
            # Plot with default color
            scatter = ax.scatter(lons, lats, s=norm_sizes, alpha=0.7, 
                               edgecolor='black', linewidth=0.5)
        
        # Add labels if requested
        if add_labels:
            for i, (x, y, label, size) in enumerate(zip(lons, lats, labels, norm_sizes)):
                # Only label larger bubbles to avoid clutter
                if size > size_scale * 0.4:
                    ax.annotate(label, (x, y), xytext=(0, 5), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        # Add title and adjust layout
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        
        # Create legend for bubble sizes
        handles = []
        if len(sizes) > 0:
            size_categories = [min_size, (min_size + max_size) / 2, max_size]
            size_labels = [f"{s:.1f}" for s in size_categories]
            norm_size_categories = [((s - min_size) / (max_size - min_size) * 0.8 + 0.2) * size_scale 
                                  for s in size_categories]
            
            for size, label in zip(norm_size_categories, size_labels):
                handles.append(plt.scatter([], [], s=size, color='blue', alpha=0.7, 
                                         edgecolor='black', linewidth=0.5, label=label))
            
            ax.legend(handles=handles, labels=size_labels, title=size_field.capitalize(),
                    loc='lower right', frameon=True, framealpha=0.8)
        
        # Save the map
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_heat_map(self,
                     grid_data: np.ndarray,
                     bounds: Tuple[float, float, float, float],
                     title: str,
                     filename: str,
                     cmap: str = 'hot_r',
                     alpha: float = 0.7,
                     overlay_shapefile: bool = True,
                     admin_level: str = 'country') -> str:
        """
        Create a heat map showing intensity over the geographical area.
        
        Args:
            grid_data: 2D numpy array representing intensity values
            bounds: (min_lon, min_lat, max_lon, max_lat) - geographical bounds of the grid
            title: Map title
            filename: Filename to save the map
            cmap: Colormap to use
            alpha: Transparency level for the heat map
            overlay_shapefile: Whether to overlay country boundaries
            admin_level: Administrative level to use for the country boundaries
            
        Returns:
            Path to the saved map file
        """
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Display the heat map
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Plot the heat map
        im = ax.imshow(grid_data, 
                      extent=[min_lon, max_lon, min_lat, max_lat],
                      cmap=cmap, 
                      alpha=alpha,
                      origin='lower')
        
        # Overlay country boundaries if requested and shapefile is available
        if overlay_shapefile:
            if admin_level in self.bd_shapefiles:
                self.bd_shapefiles[admin_level].boundary.plot(ax=ax, linewidth=1, color='black')
                print(f"Using {admin_level} shapefile for heat map boundaries")
            elif 'country' in self.bd_shapefiles:
                self.bd_shapefiles['country'].boundary.plot(ax=ax, linewidth=1, color='black')
                print(f"Using country shapefile for heat map boundaries")
            elif self.bd_shapefile is not None:
                self.bd_shapefile.boundary.plot(ax=ax, linewidth=1, color='black')
                print(f"Using default shapefile for heat map boundaries")
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Intensity")
        
        # Add title and adjust layout
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Save the map
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_bivariate_map(self,
                         var1_data: Dict[str, float],
                         var2_data: Dict[str, float],
                         title: str,
                         filename: str,
                         var1_name: str = "Variable 1",
                         var2_name: str = "Variable 2",
                         admin_level: str = "division",
                         n_classes: int = 3) -> str:
        """
        Create a bivariate choropleth map showing the relationship between two variables.
        
        Args:
            var1_data: Dictionary mapping admin unit names to values for first variable
            var2_data: Dictionary mapping admin unit names to values for second variable
            title: Map title
            filename: Filename to save the map
            var1_name: Name of first variable for legend
            var2_name: Name of second variable for legend
            admin_level: Administrative level ('division', 'district', 'upazila', 'union')
            n_classes: Number of classes for each variable (typically 3)
            
        Returns:
            Path to the saved map file
        """
        # Use the specified admin level shapefile if available
        if admin_level in self.bd_shapefiles:
            gdf = self.bd_shapefiles[admin_level].copy()
            print(f"Using {admin_level} shapefile for bivariate map")
        elif self.bd_shapefile is not None:
            gdf = self.bd_shapefile.copy()
            print(f"Using default shapefile for bivariate map")
        else:
            print("Warning: Bangladesh shapefile not available. Skipping bivariate map creation.")
            return None
        
        # Find the name column for this admin level
        admin_col = None
        possible_name_columns = [
            'ADM1_EN', 'ADM1_PCODE', 'ADM1_REF',  # Division
            'ADM2_EN', 'ADM2_PCODE', 'ADM2_REF',  # District
            'ADM3_EN', 'ADM3_PCODE', 'ADM3_REF',  # Upazila
            'ADM4_EN', 'ADM4_PCODE', 'ADM4_REF',  # Union
            'division', 'district', 'upazila', 'union',
            'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4',
            'NAME', 'Division', 'DIVISION', 'name'
        ]
        
        for col in possible_name_columns:
            if col in gdf.columns:
                admin_col = col
                print(f"Using column '{admin_col}' for admin unit names")
                break
        
        if admin_col is None:
            print(f"Warning: No suitable admin name column found in shapefile. Using first column.")
            admin_col = gdf.columns[0]  # Use first column as fallback
        
        # Convert dictionaries to Series for joining
        var1_series = pd.Series(var1_data)
        var2_series = pd.Series(var2_data)
        
        # Check if any admin names from data match those in the shapefile
        if admin_col in gdf.columns:
            matching_names_var1 = set(var1_data.keys()) & set(gdf[admin_col])
            matching_names_var2 = set(var2_data.keys()) & set(gdf[admin_col])
            
            if not matching_names_var1 or not matching_names_var2:
                print(f"Warning: No matching admin unit names found between data and shapefile.")
                # Print available names in shapefile for debugging
                print(f"Available names in shapefile: {gdf[admin_col].unique()[:5]}...")
                print(f"Names in var1 data: {list(var1_data.keys())}")
                print(f"Names in var2 data: {list(var2_data.keys())}")
                
                # Try to normalize names for better matching
                var1_normalized = {k.lower().replace(' ', '_'): v for k, v in var1_data.items()}
                var2_normalized = {k.lower().replace(' ', '_'): v for k, v in var2_data.items()}
                
                if isinstance(gdf[admin_col].iloc[0], str):
                    gdf_normalized = gdf[admin_col].str.lower().str.replace(' ', '_')
                    matching_normalized_var1 = set(var1_normalized.keys()) & set(gdf_normalized)
                    matching_normalized_var2 = set(var2_normalized.keys()) & set(gdf_normalized)
                    
                    if matching_normalized_var1 and matching_normalized_var2:
                        print(f"Found matches after normalization: {len(matching_normalized_var1)} for var1, {len(matching_normalized_var2)} for var2")
                        # Create mapping from normalized to original names
                        norm_to_orig_var1 = {k.lower().replace(' ', '_'): k for k in var1_data.keys()}
                        norm_to_orig_var2 = {k.lower().replace(' ', '_'): k for k in var2_data.keys()}
                        
                        # Update GDF with normalized names for joining
                        gdf['normalized'] = gdf_normalized
                        admin_col = 'normalized'
                        
                        # Update data to use normalized keys
                        var1_series = pd.Series({k: var1_data[norm_to_orig_var1[k]] for k in matching_normalized_var1})
                        var2_series = pd.Series({k: var2_data[norm_to_orig_var2[k]] for k in matching_normalized_var2})
        
        # Join data to geodataframe
        gdf = gdf.set_index(admin_col)
        gdf['var1'] = var1_series
        gdf['var2'] = var2_series
        gdf = gdf.reset_index()
        
        # Filter out rows with missing values
        gdf = gdf.dropna(subset=['var1', 'var2'])
        
        if len(gdf) == 0:
            print("Warning: No data matched with shapefile after joining. Skipping bivariate map creation.")
            return None
        
        # Create classification for each variable
        gdf['var1_class'] = pd.qcut(gdf['var1'].rank(method='first'), n_classes, labels=False)
        gdf['var2_class'] = pd.qcut(gdf['var2'].rank(method='first'), n_classes, labels=False)
        
        # Create bivariate classification
        gdf['bivariate_class'] = gdf['var1_class'] * n_classes + gdf['var2_class']
        
        # Create bivariate color scheme
        # We'll use a 3x3 grid with varying shades of two colors
        if n_classes == 3:
            colors = [
                '#e8e8e8', '#e4acac', '#c85a5a',  # bottom row
                '#b0d5df', '#ad9ea5', '#985356',  # middle row
                '#64acbe', '#627f8c', '#574249'   # top row
            ]
        else:
            # Simple fallback for other n_classes values
            colors = plt.cm.viridis(np.linspace(0, 1, n_classes * n_classes))
            colors = [mcolors.rgb2hex(c) for c in colors]
        
        # Create the map
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        
        # Create a color dictionary for the bivariate classes
        color_dict = {i: colors[i] for i in range(n_classes * n_classes)}
        
        # Plot the map
        gdf.plot(column='bivariate_class', 
                ax=ax, 
                categorical=True,
                legend=False,
                color=[color_dict.get(b, '#ffffff') for b in gdf['bivariate_class']],
                edgecolor='black',
                linewidth=0.5)
        
        # Add title and adjust layout
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        
        # Create a custom legend for the bivariate map
        legend_elements = []
        for i in range(n_classes):
            for j in range(n_classes):
                bivariate_class = i * n_classes + j
                color = color_dict.get(bivariate_class, '#ffffff')
                legend_elements.append(
                    mpatches.Patch(facecolor=color, edgecolor='black',
                                 label=f"{var1_name} {i+1}, {var2_name} {j+1}")
                )
        
        # Create a legend grid
        legend_fig, legend_ax = plt.subplots(figsize=(4, 4))
        legend_ax.set_axis_off()
        
        # Create grid of colored squares
        for i in range(n_classes):
            for j in range(n_classes):
                bivariate_class = i * n_classes + j
                color = color_dict.get(bivariate_class, '#ffffff')
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black')
                legend_ax.add_patch(rect)
        
        # Add text labels
        legend_ax.text(-0.5, n_classes/2, var2_name, rotation=90, va='center', ha='center', fontsize=10)
        legend_ax.text(n_classes/2, -0.5, var1_name, va='center', ha='center', fontsize=10)
        
        # Add "Low" and "High" labels
        legend_ax.text(-0.3, 0, "Low", rotation=90, va='center', ha='center', fontsize=8)
        legend_ax.text(-0.3, n_classes-1, "High", rotation=90, va='center', ha='center', fontsize=8)
        legend_ax.text(0, -0.3, "Low", va='center', ha='center', fontsize=8)
        legend_ax.text(n_classes-1, -0.3, "High", va='center', ha='center', fontsize=8)
        
        # Set axis limits
        legend_ax.set_xlim(-1, n_classes)
        legend_ax.set_ylim(-1, n_classes)
        
        # Save the legend
        legend_path = os.path.join(self.output_dir, f"{filename}_legend.png")
        legend_fig.tight_layout()
        legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
        plt.close(legend_fig)
        
        # Save the main map
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    def create_time_series_map_animation(self,
                                      time_series_data: Dict[str, Dict[int, float]],
                                      title: str,
                                      filename: str,
                                      admin_level: str = "division",
                                      cmap: str = "viridis",
                                      fps: int = 2) -> str:
        """
        Create an animated choropleth map showing how values change over time.
        
        Args:
            time_series_data: Dict mapping admin unit names to dict of year:value
            title: Animation title
            filename: Filename to save the animation
            admin_level: Administrative level ('division', 'district', 'upazila', 'union')
            cmap: Colormap to use
            fps: Frames per second for the animation
            
        Returns:
            Path to the saved animation file
        """
        # Use the specified admin level shapefile if available
        if admin_level in self.bd_shapefiles:
            gdf = self.bd_shapefiles[admin_level].copy()
            print(f"Using {admin_level} shapefile for animation")
        elif self.bd_shapefile is not None:
            gdf = self.bd_shapefile.copy()
            print(f"Using default shapefile for animation")
        else:
            print("Warning: Bangladesh shapefile not available. Skipping animation creation.")
            return None
        
        # Extract years from time_series_data
        all_years = set()
        for region_data in time_series_data.values():
            all_years.update(region_data.keys())
        years = sorted(all_years)
        
        if not years:
            print("Warning: No time series data available for animation.")
            return None
        
        # Find the name column for this admin level
        admin_col = None
        possible_name_columns = [
            'ADM1_EN', 'ADM1_PCODE', 'ADM1_REF',  # Division
            'ADM2_EN', 'ADM2_PCODE', 'ADM2_REF',  # District
            'ADM3_EN', 'ADM3_PCODE', 'ADM3_REF',  # Upazila
            'ADM4_EN', 'ADM4_PCODE', 'ADM4_REF',  # Union
            'division', 'district', 'upazila', 'union',
            'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4',
            'NAME', 'Division', 'DIVISION', 'name'
        ]
        
        for col in possible_name_columns:
            if col in gdf.columns:
                admin_col = col
                print(f"Using column '{admin_col}' for admin unit names")
                break
        
        if admin_col is None:
            print(f"Warning: No suitable admin name column found in shapefile. Using first column.")
            admin_col = gdf.columns[0]  # Use first column as fallback
        
        # Check for name matching issues
        matching_names = set(time_series_data.keys()) & set(gdf[admin_col])
        if not matching_names:
            print(f"Warning: No matching admin unit names found between data and shapefile.")
            # Print available names in shapefile for debugging
            print(f"Available names in shapefile: {gdf[admin_col].unique()[:5]}...")
            print(f"Names in data: {list(time_series_data.keys())[:5]}...")
            
            # Try to normalize names for better matching
            data_normalized = {}
            for k, v in time_series_data.items():
                data_normalized[k.lower().replace(' ', '_')] = v
            
            if isinstance(gdf[admin_col].iloc[0], str):
                gdf_normalized = gdf[admin_col].str.lower().str.replace(' ', '_')
                matching_normalized = set(data_normalized.keys()) & set(gdf_normalized)
                
                if matching_normalized:
                    print(f"Found {len(matching_normalized)} matches after normalization.")
                    # Create mapping from normalized to original names
                    norm_to_orig = {k.lower().replace(' ', '_'): k for k in time_series_data.keys()}
                    # Update GDF with normalized names
                    gdf['normalized'] = gdf_normalized
                    admin_col = 'normalized'
                    # Update data to use normalized keys
                    time_series_data = {k: time_series_data[norm_to_orig.get(k, k)] for k in matching_normalized}
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Find min and max values for consistent color scale
        all_values = []
        for region_data in time_series_data.values():
            all_values.extend(region_data.values())
        
        vmin = min(all_values) if all_values else 0
        vmax = max(all_values) if all_values else 1
        
        # Function to update the plot for each frame
        def update(frame):
            ax.clear()
            year = years[frame]
            
            # Create data for this year
            data = {region: region_data.get(year, np.nan) 
                   for region, region_data in time_series_data.items()}
            
            # Convert data to DataFrame for joining
            data_series = pd.Series(data)
            
            # Copy geodataframe for this frame
            frame_gdf = gdf.copy()
            
            # Join data to geodataframe
            frame_gdf = frame_gdf.set_index(admin_col)
            frame_gdf['value'] = data_series
            frame_gdf = frame_gdf.reset_index()
            
            # Plot the map
            frame_gdf.plot(column='value', 
                         ax=ax, 
                         cmap=cmap,
                         vmin=vmin,
                         vmax=vmax,
                         missing_kwds={'color': 'lightgrey'},
                         legend=True,
                         legend_kwds={'label': "Value"})
            
            # Add title and year
            ax.set_title(f"{title} - {year}", fontsize=14)
            ax.set_axis_off()
            
            # Add country borders
            if 'geometry' in frame_gdf.columns:
                frame_gdf.boundary.plot(ax=ax, linewidth=0.5, color='black')
            
            return ax,
        
        # Create the animation
        animation = FuncAnimation(fig, update, frames=len(years), blit=False)
        
        # Save the animation
        output_path = os.path.join(self.output_dir, f"{filename}.gif")
        animation.save(output_path, writer='pillow', fps=fps, dpi=100)
        plt.close()
        
        return output_path
