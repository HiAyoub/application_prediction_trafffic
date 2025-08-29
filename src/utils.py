"""
Utility Functions
================

Common utility functions for data processing, validation, and file handling.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
import re
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

def detect_coordinate_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Automatically detect latitude and longitude columns in a dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with 'lat' and 'lon' keys containing column names or None
    """
    logger.info("Detecting coordinate columns...")

    # 1. Check for geometry column with shapely Point objects
    if 'geometry' in df.columns:
        try:
            from shapely.geometry import Point
            if df['geometry'].apply(lambda g: isinstance(g, Point)).all():
                df['latitude'] = df['geometry'].apply(lambda g: g.y)
                df['longitude'] = df['geometry'].apply(lambda g: g.x)
                logger.info("Extracted latitude/longitude from geometry column.")
                return {'lat': 'latitude', 'lon': 'longitude'}
        except Exception as e:
            logger.warning(f"Could not extract coordinates from geometry: {str(e)}")
    
    # Common patterns for latitude columns
    lat_patterns = [
        r'lat(itude)?', r'y', r'coord_y', r'northing', 
        r'geo_lat', r'position_lat', r'lat_deg'
    ]
    
    # Common patterns for longitude columns
    lon_patterns = [
        r'lon(gitude)?', r'lng', r'x', r'coord_x', r'easting',
        r'geo_lon', r'geo_lng', r'position_lon', r'lon_deg'
    ]
    
    lat_col = None
    lon_col = None
    
    # Check column names against patterns
    columns = df.columns.tolist()
    
    for col in columns:
        col_lower = col.lower().strip()
        
        # Check for latitude patterns
        if not lat_col:
            for pattern in lat_patterns:
                if re.search(pattern, col_lower, re.IGNORECASE):
                    # Verify it's numeric and has reasonable values
                    if pd.api.types.is_numeric_dtype(df[col]):
                        lat_values = df[col].dropna()
                        if len(lat_values) > 0 and -90 <= lat_values.min() <= lat_values.max() <= 90:
                            lat_col = col
                            break
        
        # Check for longitude patterns
        if not lon_col:
            for pattern in lon_patterns:
                if re.search(pattern, col_lower, re.IGNORECASE):
                    # Verify it's numeric and has reasonable values
                    if pd.api.types.is_numeric_dtype(df[col]):
                        lon_values = df[col].dropna()
                        if len(lon_values) > 0 and -180 <= lon_values.min() <= lon_values.max() <= 180:
                            lon_col = col
                            break
    
    # If not found by pattern, check for numeric columns with coordinate-like values
    if not lat_col or not lon_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_values = df[col].dropna()
            if len(col_values) == 0:
                continue
                
            # Check if values are in latitude range
            if not lat_col and -90 <= col_values.min() <= col_values.max() <= 90:
                # Additional check: latitude values should have reasonable distribution
                value_range = col_values.max() - col_values.min()
                if 0 < value_range < 180:  # Reasonable range for latitude
                    lat_col = col
                    continue
            
            # Check if values are in longitude range
            if not lon_col and -180 <= col_values.min() <= col_values.max() <= 180:
                # Additional check: longitude values should have reasonable distribution
                value_range = col_values.max() - col_values.min()
                if 0 < value_range < 360:  # Reasonable range for longitude
                    lon_col = col
    
    result = {'lat': lat_col, 'lon': lon_col}
    
    if lat_col and lon_col:
        logger.info(f"Coordinate columns detected: latitude='{lat_col}', longitude='{lon_col}'")
    else:
        logger.warning("Could not detect coordinate columns automatically")
    
    return result

def validate_file(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate uploaded file against configuration requirements.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, message)
    """
    logger.info("Validating uploaded file...")
    
    try:
        # Check if dataframe is empty
        if df.empty:
            return False, "File is empty"
        
        # Check minimum size requirements
        if len(df) < 10:
            return False, "File contains too few rows (minimum 10 required)"
        
        # Check for required columns if specified in config
        required_columns = config.get('required_columns', [])
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Check data types for known columns
        numeric_columns = config.get('numeric_columns', [])
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    pd.to_numeric(df[col], errors='coerce')
                except:
                    return False, f"Column '{col}' should be numeric but contains non-numeric values"
        
        # Check for excessive missing values
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 50:
            return False, f"File contains too many missing values ({missing_percentage:.1f}%)"
        
        # Check for coordinate columns if geospatial analysis is expected
        coord_info = detect_coordinate_columns(df)
        if not coord_info['lat'] or not coord_info['lon']:
            logger.warning("No coordinate columns detected - geospatial features will be limited")
        
        logger.info("File validation passed")
        return True, "File validation successful"
        
    except Exception as e:
        logger.error(f"Error during file validation: {str(e)}")
        return False, f"Validation error: {str(e)}"

def save_uploaded_file(uploaded_file, upload_dir: str = "data/uploads") -> str:
    """
    Save an uploaded file to the specified directory.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        upload_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    logger.info(f"Saving uploaded file: {uploaded_file.name}")
    
    try:
        # Create upload directory if it doesn't exist
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = upload_path / filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"File saved to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing special characters and standardizing format.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with cleaned column names
    """
    logger.info("Cleaning column names...")
    
    try:
        df_cleaned = df.copy()
        
        # Clean column names
        new_columns = []
        for col in df_cleaned.columns:
            # Convert to string and strip whitespace
            clean_col = str(col).strip()
            
            # Replace special characters with underscores
            clean_col = re.sub(r'[^\w\s]', '_', clean_col)
            
            # Replace spaces with underscores
            clean_col = re.sub(r'\s+', '_', clean_col)
            
            # Remove multiple consecutive underscores
            clean_col = re.sub(r'_+', '_', clean_col)
            
            # Remove leading/trailing underscores
            clean_col = clean_col.strip('_')
            
            # Convert to lowercase
            clean_col = clean_col.lower()
            
            # Ensure column name is not empty
            if not clean_col:
                clean_col = f'column_{len(new_columns)}'
            
            new_columns.append(clean_col)
        
        # Handle duplicate column names
        seen = set()
        final_columns = []
        for col in new_columns:
            original_col = col
            counter = 1
            while col in seen:
                col = f"{original_col}_{counter}"
                counter += 1
            seen.add(col)
            final_columns.append(col)
        
        df_cleaned.columns = final_columns
        
        logger.info(f"Column names cleaned. Renamed {sum(1 for old, new in zip(df.columns, final_columns) if old != new)} columns")
        
        return df_cleaned
        
    except Exception as e:
        logger.error(f"Error cleaning column names: {str(e)}")
        return df

def calculate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics for a dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary containing summary statistics
    """
    logger.info("Calculating data summary...")
    
    try:
        summary = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'duplicated_rows': df.duplicated().sum()
            },
            'column_types': {
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns)
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'columns_with_missing': df.columns[df.isnull().any()].tolist()
            },
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()
            summary['numeric_summary'] = {
                'columns': numeric_cols.tolist(),
                'mean_values': numeric_stats.loc['mean'].to_dict(),
                'std_values': numeric_stats.loc['std'].to_dict(),
                'min_values': numeric_stats.loc['min'].to_dict(),
                'max_values': numeric_stats.loc['max'].to_dict()
            }
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols:
                unique_count = df[col].nunique()
                most_common = df[col].mode()
                cat_summary[col] = {
                    'unique_values': unique_count,
                    'most_common': most_common.iloc[0] if len(most_common) > 0 else None,
                    'most_common_count': df[col].value_counts().iloc[0] if unique_count > 0 else 0
                }
            summary['categorical_summary'] = cat_summary
        
        logger.info("Data summary calculated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error calculating data summary: {str(e)}")
        return {}

def format_number(number: float, precision: int = 3) -> str:
    """
    Format numbers for display with appropriate precision and units.
    
    Args:
        number: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if pd.isna(number):
        return "N/A"
    
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.{precision}f}M"
    elif abs(number) >= 1_000:
        return f"{number / 1_000:.{precision}f}K"
    elif abs(number) < 0.001 and number != 0:
        return f"{number:.2e}"
    else:
        return f"{number:.{precision}f}"

def create_download_link(df: pd.DataFrame, filename: str = "processed_data.csv") -> str:
    """
    Create a download link for a dataframe.
    
    Args:
        df: Dataframe to download
        filename: Name for the downloaded file
        
    Returns:
        CSV string for download
    """
    return df.to_csv(index=False)

def validate_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str) -> Tuple[bool, str, pd.DataFrame]:
    """
    Validate and clean coordinate data.
    
    Args:
        df: Input dataframe
        lat_col: Latitude column name
        lon_col: Longitude column name
        
    Returns:
        Tuple of (is_valid, message, cleaned_dataframe)
    """
    logger.info("Validating coordinate data...")
    
    try:
        df_clean = df.copy()
        
        # Check if columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            return False, f"Coordinate columns not found: {lat_col}, {lon_col}", df
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[lat_col]) or not pd.api.types.is_numeric_dtype(df[lon_col]):
            return False, "Coordinate columns must be numeric", df
        
        # Remove rows with missing coordinates
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=[lat_col, lon_col])
        removed_missing = initial_len - len(df_clean)
        
        if len(df_clean) == 0:
            return False, "No valid coordinate data found", df
        
        # Validate coordinate ranges
        lat_values = df_clean[lat_col]
        lon_values = df_clean[lon_col]
        
        # Remove invalid latitude values (outside -90 to 90)
        valid_lat = (lat_values >= -90) & (lat_values <= 90)
        df_clean = df_clean[valid_lat]
        removed_lat = len(df_clean) - valid_lat.sum()
        
        # Remove invalid longitude values (outside -180 to 180)
        lon_values = df_clean[lon_col]
        valid_lon = (lon_values >= -180) & (lon_values <= 180)
        df_clean = df_clean[valid_lon]
        removed_lon = len(df_clean) - valid_lon.sum()
        
        if len(df_clean) == 0:
            return False, "No coordinates within valid ranges found", df
        
        # Create validation message
        total_removed = removed_missing + removed_lat + removed_lon
        message = f"Coordinate validation completed. "
        if total_removed > 0:
            message += f"Removed {total_removed} invalid records "
            message += f"({removed_missing} missing, {removed_lat} invalid lat, {removed_lon} invalid lon). "
        message += f"Valid coordinates: {len(df_clean)} records"
        
        logger.info(message)
        return True, message, df_clean
        
    except Exception as e:
        logger.error(f"Error validating coordinates: {str(e)}")
        return False, f"Coordinate validation error: {str(e)}", df
