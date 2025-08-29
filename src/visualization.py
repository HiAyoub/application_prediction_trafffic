"""
Visualization Module
==================

Handles geospatial visualization and interactive charts for the traffic prediction platform.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import branca.colormap as cm
from shapely.geometry import Point, LineString
logger = logging.getLogger(__name__)

class GeospatialVisualizer:
    """
    Handles geospatial visualization using Folium and interactive maps.
    """
    
    def __init__(self):
        """Initialize the geospatial visualizer."""
        self.map_styles = {
            'OpenStreetMap': 'OpenStreetMap',
            'CartoDB positron': 'CartoDB positron',
            'CartoDB dark_matter': 'CartoDB dark_matter',
            'Stamen Terrain': 'Stamen Terrain'
        }
        logger.info("GeospatialVisualizer initialized")
    
    

    def linestring_to_point(self,geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == 'LineString':
            return Point(geom.coords[0])  # Premier point de la ligne
        elif geom.geom_type == 'Point':
            return geom
        else:
            return None
    def create_traffic_map(self, df: pd.DataFrame, geometry_col: str = 'geometry',
                          color_col: Optional[str] = None, size_col: Optional[str] = None,
                          tile_style: str = 'OpenStreetMap',lat_col='latitude',      # <-- Ajoute cette ligne
                          lon_col='longitude', popup_func=None) -> folium.Map:
        """
        Create an interactive traffic map using geometry column for coordinates.
        
        Args:
            df: DataFrame containing the data
            geometry_col: Name of geometry column
            color_col: Column to use for color coding (optional)
            size_col: Column to use for sizing markers (optional)
            tile_style: Map tile style
            lat_col: Latitude column name
            lon_col: Longitude column name
            popup_func: Optional function(row) -> str for custom popup HTML
            
        Returns:
            Folium map object
        """
        logger.info(f"Creating traffic map with {len(df)} points")
        df['geometry'] = df['geometry'].apply(self.linestring_to_point)
        df = df[df['geometry'].notnull()]
        try:
            # Utiliser geometry pour calculer le centre
            coords = df[geometry_col].apply(lambda g: (g.y, g.x) if hasattr(g, 'x') and hasattr(g, 'y') else (np.nan, np.nan))
            center_lat = coords.apply(lambda c: c[0]).mean()
            center_lon = coords.apply(lambda c: c[1]).mean()

            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles=self.map_styles.get(tile_style, 'OpenStreetMap')
            )
            
            # Prepare color mapping if color column is specified
            color_map = None
            if color_col and color_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[color_col]):
                    # Numeric color mapping
                    min_val = df[color_col].min()
                    max_val = df[color_col].max()
                    color_map = cm.LinearColormap(
                        colors=['blue', 'green', 'yellow', 'red'],
                        index=[min_val, min_val + (max_val - min_val) * 0.33,
                               min_val + (max_val - min_val) * 0.66, max_val],
                        vmin=min_val,
                        vmax=max_val,
                        caption=color_col
                    )
                    color_map.add_to(m)
                else:
                    # Categorical color mapping
                    unique_values = df[color_col].unique()
                    colors = px.colors.qualitative.Set1[:len(unique_values)]
                    color_mapping = dict(zip(unique_values, colors))
            
            # Prepare size mapping if size column is specified
            size_mapping = None
            if size_col and size_col in df.columns and pd.api.types.is_numeric_dtype(df[size_col]):
                min_size, max_size = 5, 20
                min_val = df[size_col].min()
                max_val = df[size_col].max()
                size_range = max_val - min_val
                
                def get_marker_size(value):
                    if size_range == 0:
                        return min_size
                    normalized = (value - min_val) / size_range
                    return min_size + (max_size - min_size) * normalized
                
                size_mapping = get_marker_size
            
            # Add markers
            for idx, row in df.iterrows():
                geom = row[geometry_col]
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    lat, lon = geom.y, geom.x
                else:
                    continue  # skip if geometry is invalid

                # Determine marker color
                if color_col and color_col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[color_col]):
                        marker_color = color_map(row[color_col]) if color_map else 'blue'
                    else:
                        marker_color = color_mapping.get(row[color_col], 'blue')
                else:
                    marker_color = 'blue'
                
                # Determine marker size
                marker_radius = 8
                if size_mapping and size_col in df.columns:
                    marker_radius = size_mapping(row[size_col])
                
                # Custom popup if provided
                if popup_func is not None:
                    popup_content = popup_func(row)
                else:
                    # Default popup
                    popup_content = f"""
                    <div style="font-family: Arial, sans-serif; width: 200px;">
                        <h4 style="margin: 0 0 10px 0; color: #333;">Point de Données Trafic</h4>
                        <table style="width: 100%; font-size: 12px;">
                            <tr><td><b>Latitude :</b></td><td>{lat:.6f}</td></tr>
                            <tr><td><b>Longitude :</b></td><td>{lon:.6f}</td></tr>
                    """
                    # Add other relevant columns (French translation)
                    display_cols = [col for col in df.columns 
                                  if col not in [geometry_col] and not pd.isna(row[col])][:5]
                    
                    for col in display_cols:
                        value = row[col]
                        if isinstance(value, (int, float)):
                            if abs(value) >= 1000:
                                value = f"{value:,.0f}"
                            else:
                                value = f"{value:.3f}" if isinstance(value, float) else str(value)
                        popup_content += f"<tr><td><b>{col} :</b></td><td>{value}</td></tr>"
                    popup_content += "</table></div>"
                
                # Add marker to map
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=marker_radius,
                    popup=folium.Popup(popup_content, max_width=250),
                    color='black',
                    weight=1,
                    fillColor=marker_color,
                    fillOpacity=0.7,
                    tooltip=f"Point {idx}"
                ).add_to(m)
            
            # Add layer control if multiple layers would be useful
            
            if len(df) > 1000:
                # Add heatmap layer for large datasets
                heat_data = [[row[lat_col], row[lon_col]] for idx, row in df.iterrows()]
                
                heatmap = plugins.HeatMap(
                    heat_data,
                    name='Traffic Density Heatmap',
                    show=False
                )
                heatmap.add_to(m)
                
                folium.LayerControl().add_to(m)
            
            # Add fullscreen button
            plugins.Fullscreen().add_to(m)
            
            # Add measure control
            plugins.MeasureControl().add_to(m)
            
            logger.info("Traffic map created successfully")
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating traffic map: {str(e)}")
            raise
    
    def create_prediction_map(self, df: pd.DataFrame, geometry_col: str = 'geometry',
                            prediction_col: str = 'prediction', actual_col: Optional[str] = None) -> folium.Map:
        """
        Create a map showing model predictions vs actual values using geometry column.
        
        Args:
            df: DataFrame containing predictions and coordinates
            geometry_col: Name of geometry column
            prediction_col: Name of prediction column
            actual_col: Name of actual values column (optional)
            
        Returns:
            Folium map object
        """
        logger.info("Creating prediction comparison map")
        
        try:
            coords = df[geometry_col].apply(lambda g: (g.y, g.x) if hasattr(g, 'x') and hasattr(g, 'y') else (np.nan, np.nan))
            center_lat = coords.apply(lambda c: c[0]).mean()
            center_lon = coords.apply(lambda c: c[1]).mean()

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='CartoDB positron'
            )
            
            # Create color mapping for predictions
            pred_min = df[prediction_col].min()
            pred_max = df[prediction_col].max()
            
            pred_colormap = cm.LinearColormap(
                colors=['blue', 'green', 'yellow', 'red'],
                index=[pred_min, pred_min + (pred_max - pred_min) * 0.33,
                       pred_min + (pred_max - pred_min) * 0.66, pred_max],
                vmin=pred_min,
                vmax=pred_max,
                caption='Predicted Values'
            )
            pred_colormap.add_to(m)
            
            # Add prediction markers
            for idx, row in df.iterrows():
                geom = row[geometry_col]
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    lat, lon = geom.y, geom.x
                else:
                    continue

                pred_value = row[prediction_col]
                
                # Create popup content
                popup_content = f"""
                <div style="font-family: Arial, sans-serif; width: 220px;">
                    <h4 style="margin: 0 0 10px 0; color: #333;">Prediction Details</h4>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>Predicted:</b></td><td>{pred_value:.3f}</td></tr>
                """
                
                if actual_col and actual_col in df.columns and not pd.isna(row[actual_col]):
                    actual_value = row[actual_col]
                    error = abs(pred_value - actual_value)
                    popup_content += f"""
                        <tr><td><b>Actual:</b></td><td>{actual_value:.3f}</td></tr>
                        <tr><td><b>Error:</b></td><td>{error:.3f}</td></tr>
                    """
                
                popup_content += f"""
                        <tr><td><b>Latitude:</b></td><td>{lat:.6f}</td></tr>
                        <tr><td><b>Longitude:</b></td><td>{lon:.6f}</td></tr>
                    </table>
                </div>
                """
                
                # Determine marker color based on prediction
                marker_color = pred_colormap(pred_value)
                
                # Determine marker size based on error (if actual values available)
                marker_radius = 8
                if actual_col and actual_col in df.columns and not pd.isna(row[actual_col]):
                    error = abs(pred_value - row[actual_col])
                    max_error = abs(df[prediction_col] - df[actual_col]).max()
                    if max_error > 0:
                        error_ratio = error / max_error
                        marker_radius = 5 + error_ratio * 10  # Size based on error
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=marker_radius,
                    popup=folium.Popup(popup_content, max_width=250),
                    color='black',
                    weight=1,
                    fillColor=marker_color,
                    fillOpacity=0.7,
                    tooltip=f"Prediction: {pred_value:.3f}"
                ).add_to(m)
            
            # Add plugins
            plugins.Fullscreen().add_to(m)
            plugins.MeasureControl().add_to(m)
            
            logger.info("Prediction map created successfully")
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating prediction map: {str(e)}")
            raise
    
    def create_cluster_map(self, df: pd.DataFrame, geometry_col: str = 'geometry',
                          cluster_col: str = 'cluster') -> folium.Map:
        """
        Create a map showing data clusters using geometry column.
        
        Args:
            df: DataFrame containing cluster assignments
            geometry_col: Name of geometry column
            cluster_col: Name of cluster assignment column
            
        Returns:
            Folium map object
        """
        logger.info("Creating cluster visualization map")
        
        try:
            coords = df[geometry_col].apply(lambda g: (g.y, g.x) if hasattr(g, 'x') and hasattr(g, 'y') else (np.nan, np.nan))
            center_lat = coords.apply(lambda c: c[0]).mean()
            center_lon = coords.apply(lambda c: c[1]).mean()

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Get unique clusters and assign colors
            unique_clusters = sorted(df[cluster_col].unique())
            colors = px.colors.qualitative.Set1[:len(unique_clusters)]
            cluster_colors = dict(zip(unique_clusters, colors))
            
            # Add markers for each cluster
            for cluster in unique_clusters:
                cluster_data = df[df[cluster_col] == cluster]
                
                for idx, row in cluster_data.iterrows():
                    geom = row[geometry_col]
                    if hasattr(geom, 'x') and hasattr(geom, 'y'):
                        lat, lon = geom.y, geom.x
                    else:
                        continue
                    
                    popup_content = f"""
                    <div style="font-family: Arial, sans-serif; width: 200px;">
                        <h4 style="margin: 0 0 10px 0; color: #333;">Cluster {cluster}</h4>
                        <table style="width: 100%; font-size: 12px;">
                            <tr><td><b>Cluster:</b></td><td>{cluster}</td></tr>
                            <tr><td><b>Latitude:</b></td><td>{lat:.6f}</td></tr>
                            <tr><td><b>Longitude:</b></td><td>{lon:.6f}</td></tr>
                        </table>
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=8,
                        popup=folium.Popup(popup_content, max_width=250),
                        color='black',
                        weight=1,
                        fillColor=cluster_colors[cluster],
                        fillOpacity=0.7,
                        tooltip=f"Cluster {cluster}"
                    ).add_to(m)
            
            # Create legend
            legend_html = """
            <div style="position: fixed; 
                        top: 10px; right: 10px; width: 150px; height: auto; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
                <h4 style="margin: 0 0 10px 0;">Clusters</h4>
            """
            
            for cluster, color in cluster_colors.items():
                legend_html += f"""
                <p style="margin: 5px 0;"><i class="fa fa-circle" 
                   style="color:{color}"></i> Cluster {cluster}</p>
                """
            
            legend_html += "</div>"
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add plugins
            plugins.Fullscreen().add_to(m)
            
            logger.info("Cluster map created successfully")
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating cluster map: {str(e)}")
            raise

class MetricsVisualizer:
    """
    Handles creation of interactive charts and visualizations for model metrics and data analysis.
    """
    
    def __init__(self):
        """Initialize the metrics visualizer."""
        logger.info("MetricsVisualizer initialized")
    
    def create_model_comparison_chart(self, results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Create a comparison chart for multiple models.
        
        Args:
            results: Dictionary containing model results
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating model comparison chart")
        
        try:
            models = list(results.keys())
            metrics = ['mae', 'rmse', 'r2']
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Mean Absolute Error', 'Root Mean Square Error', 'R² Score'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            mae_values  = [results[model]['metrics']['mae'] for model in models]
            rmse_values = [results[model]['metrics']['rmse'] for model in models]
            r2_values   = [results[model]['metrics']['r2'] for model in models]
            
            fig.add_trace(go.Bar(x=models, y=mae_values, name='MAE', showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=models, y=rmse_values, name='RMSE', showlegend=False), row=1, col=2)
            fig.add_trace(go.Bar(x=models, y=r2_values, name='R²', showlegend=False), row=1, col=3)
            
            fig.update_layout(
                height=400,
                title_text="Model Performance Comparison",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model comparison chart: {str(e)}")
            raise
    
    def create_prediction_scatter(self, y_true: List[float], y_pred: List[float], 
                                model_name: str = "Model") -> go.Figure:
        """
        Create a scatter plot of predictions vs actual values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating prediction scatter plot")
        
        try:
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions',
                opacity=0.6,
                marker=dict(size=8, color='blue')
            ))
            
            # Add perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"{model_name} - Predictions vs Actual Values",
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating prediction scatter plot: {str(e)}")
            raise
    
    def create_residuals_plot(self, y_true: List[float], y_pred: List[float], 
                            model_name: str = "Model") -> go.Figure:
        """
        Create a residuals plot for model evaluation.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating residuals plot")
        
        try:
            residuals = np.array(y_true) - np.array(y_pred)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                opacity=0.6,
                marker=dict(size=8, color='green')
            ))
            
            # Add horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title=f"{model_name} - Residual Analysis",
                xaxis_title="Predicted Values",
                yaxis_title="Residuals (Actual - Predicted)",
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating residuals plot: {str(e)}")
            raise
    
    def create_feature_importance_chart(self, importance_dict: Dict[str, float], 
                                      model_name: str = "Model", top_n: int = 15) -> go.Figure:
        """
        Create a feature importance chart.
        
        Args:
            importance_dict: Dictionary of feature names and their importance scores
            model_name: Name of the model
            top_n: Number of top features to display
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating feature importance chart")
        
        try:
            # Sort features by importance and take top N
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            
            fig = go.Figure(go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color='skyblue'
            ))
            
            fig.update_layout(
                title=f"{model_name} - Feature Importance (Top {top_n})",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(features) * 25),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {str(e)}")
            raise
