"""
Data Processing Module
=====================

Comprehensive data processing pipeline integrating the existing preprocessing logic
with MLOps best practices.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any, Union
from pathlib import Path
import yaml
from datetime import datetime
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Comprehensive data processor integrating existing preprocessing pipeline
    with additional MLOps capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor with configuration.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
        self.numeric_columns = config.get('numeric_columns', [])
        self.categorical_columns = config.get('categorical_columns', [])
        self.required_columns = config.get('required_columns', [])
        self.target_column = config.get('target_column', 'debit')
        self.geometry_columns = config.get('geometry_columns', [])
        
        # Initialize scalers and encoders
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        logger.info("DataProcessor initialized successfully")
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate input data structure and content.
        
        Args:
            df: Input dataframe to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check if dataframe is empty
            if df.empty:
                return False, "Dataset is empty"
            
            # Check required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Check data types
            for col in self.numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        return False, f"Column {col} cannot be converted to numeric"
            
            # Check for minimum data requirements
            if len(df) < 10:
                return False, "Dataset too small (minimum 10 rows required)"
            
            # Vérification colonne geometry
            for col in self.geometry_columns:
                if col not in df.columns:
                    return False, f"Missing required geometry column: {col}"
                # Vérification du type (object ou geometry)
                if not (pd.api.types.is_object_dtype(df[col]) or str(df[col].dtype).startswith('geometry')):
                    return False, f"Column {col} must be of geometry type"
                # Vérification du taux de valeurs manquantes
                missing_rate = df[col].isnull().mean()
                if missing_rate > 0:
                    return False, f"Column {col} has missing values rate {missing_rate:.2%}, must be 0"
            
            return True, "Data validation passed"
            
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def run_quality_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive data quality tests.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing quality test results
        """
        logger.info("Running data quality tests...")
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
        
        try:
            tests_passed = 0
            tests_failed = 0
            
            # Test 1: Missing values analysis
            missing_test = self._test_missing_values(df)
            quality_report['tests']['missing_values'] = missing_test
            if missing_test['passed']:
                tests_passed += 1
            else:
                tests_failed += 1

            # Test geometry missing rate
            for col in self.geometry_columns:
                missing_rate = df[col].isnull().mean() if col in df.columns else 1
                geometry_test = {
                    'passed': missing_rate == 0,
                    'missing_rate': missing_rate,
                    'message': f"Geometry column '{col}' missing rate: {missing_rate:.2%}"
                }
                quality_report['tests'][f'geometry_missing_{col}'] = geometry_test
                if geometry_test['passed']:
                    tests_passed += 1
                else:
                    tests_failed += 1
            
            # Test 2: Data type consistency
            dtype_test = self._test_data_types(df)
            quality_report['tests']['data_types'] = dtype_test
            if dtype_test['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
            
            # Test 3: Value range validation
            range_test = self._test_value_ranges(df)
            quality_report['tests']['value_ranges'] = range_test
            if range_test['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
            
            # Test 4: Duplicate records
            duplicate_test = self._test_duplicates(df)
            quality_report['tests']['duplicates'] = duplicate_test
            if duplicate_test['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
            
            # Test 5: Statistical outliers
            outlier_test = self._test_outliers(df)
            quality_report['tests']['outliers'] = outlier_test
            if outlier_test['passed']:
                tests_passed += 1
            else:
                tests_failed += 1
            
            # Summary
            quality_report['summary'] = {
                'tests_passed': tests_passed,
                'tests_failed': tests_failed,
                'total_tests': tests_passed + tests_failed,
                'quality_score': tests_passed / (tests_passed + tests_failed) if (tests_passed + tests_failed) > 0 else 0
            }
            
            logger.info(f"Quality tests completed: {tests_passed} passed, {tests_failed} failed")
            
        except Exception as e:
            logger.error(f"Error during quality testing: {str(e)}")
            quality_report['error'] = str(e)
        
        return quality_report
    
    def _test_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test for missing values"""
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
        # Vérification stricte pour geometry
        geometry_missing = {}
        for col in self.geometry_columns:
            if col in df.columns:
                rate = df[col].isnull().mean()
                geometry_missing[col] = rate
        passed = missing_percentage < 10 and all(rate == 0 for rate in geometry_missing.values())
        return {
            'passed': passed,
            'total_missing': int(total_missing),
            'missing_percentage': round(missing_percentage, 2),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'geometry_missing_rate': geometry_missing,
            'message': f"Missing values: {missing_percentage:.2f}% of total data"
        }
    
    def _test_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test data type consistency"""
        type_issues = []
        
        for col in self.numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                type_issues.append(f"{col} should be numeric")
        
        for col in self.categorical_columns:
            if col in df.columns and not pd.api.types.is_object_dtype(df[col]):
                type_issues.append(f"{col} should be categorical")
        
        return {
            'passed': len(type_issues) == 0,
            'issues': type_issues,
            'message': f"Data type issues: {len(type_issues)}"
        }
    
    def _test_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test value ranges against configured limits"""
        range_violations = []
        
        # Check minimum values
        min_values = self.config.get('data_quality', {}).get('min_value', {})
        for col, min_val in min_values.items():
            if col in df.columns:
                violations = (df[col] < min_val).sum()
                if violations > 0:
                    range_violations.append(f"{col}: {violations} values below minimum {min_val}")
        
        # Check maximum values
        max_values = self.config.get('data_quality', {}).get('max_value', {})
        for col, max_val in max_values.items():
            if col in df.columns:
                violations = (df[col] > max_val).sum()
                if violations > 0:
                    range_violations.append(f"{col}: {violations} values above maximum {max_val}")
        
        return {
            'passed': len(range_violations) == 0,
            'violations': range_violations,
            'message': f"Range violations: {len(range_violations)}"
        }
    
    def _test_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test for duplicate records"""
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        return {
            'passed': duplicate_percentage < 5,  # Pass if less than 5% duplicates
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': round(duplicate_percentage, 2),
            'message': f"Duplicate records: {duplicate_percentage:.2f}%"
        }
    
    def _test_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test for statistical outliers using IQR method"""
        outlier_summary = {}
        total_outliers = 0
        
        for col in self.numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_summary[col] = int(outliers)
                total_outliers += outliers
        
        outlier_percentage = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
        
        return {
            'passed': outlier_percentage < 20,  # Pass if less than 20% outliers
            'total_outliers': int(total_outliers),
            'outlier_percentage': round(outlier_percentage, 2),
            'outliers_by_column': outlier_summary,
            'message': f"Statistical outliers: {outlier_percentage:.2f}%"
        }
    
    def detect_drift(self, df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            df: Current dataset
            reference_df: Reference dataset for comparison
            
        Returns:
            Dictionary containing drift analysis results
        """
        logger.info("Running data drift detection...")
        
        if reference_df is None:
            # Create a synthetic reference or use a subset
            reference_df = df.sample(frac=0.5, random_state=42)
            df = df.drop(reference_df.index)
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
        
        try:
            drift_detected = 0
            stable_features = 0
            threshold = self.config.get('drift', {}).get('threshold', 0.05)
            
            # Test numeric columns using Kolmogorov-Smirnov test
            for col in self.numeric_columns:
                if col in df.columns and col in reference_df.columns:
                    try:
                        statistic, p_value = stats.ks_2samp(
                            reference_df[col].dropna(), 
                            df[col].dropna()
                        )
                        
                        drift_report['tests'][f'ks_{col}'] = {
                            'test': 'Kolmogorov-Smirnov',
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'drift_detected': p_value < threshold,
                            'threshold': threshold
                        }
                        
                        if p_value < threshold:
                            drift_detected += 1
                        else:
                            stable_features += 1
                            
                    except Exception as e:
                        logger.warning(f"Could not test drift for {col}: {str(e)}")
            
            # Test categorical columns using Chi-square test
            for col in self.categorical_columns:
                if col in df.columns and col in reference_df.columns:
                    try:
                        # Create contingency table
                        ref_counts = reference_df[col].value_counts()
                        curr_counts = df[col].value_counts()
                        
                        # Align indices
                        all_categories = set(ref_counts.index) | set(curr_counts.index)
                        ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
                        curr_aligned = curr_counts.reindex(all_categories, fill_value=0)
                        
                        # Perform chi-square test
                        statistic, p_value, _, _ = stats.chi2_contingency([ref_aligned, curr_aligned])
                        
                        drift_report['tests'][f'chi2_{col}'] = {
                            'test': 'Chi-square',
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'drift_detected': p_value < threshold,
                            'threshold': threshold
                        }
                        
                        if p_value < threshold:
                            drift_detected += 1
                        else:
                            stable_features += 1
                            
                    except Exception as e:
                        logger.warning(f"Could not test drift for {col}: {str(e)}")
            
            # Summary
            drift_report['summary'] = {
                'drift_detected': drift_detected,
                'stable_features': stable_features,
                'total_features_tested': drift_detected + stable_features,
                'drift_percentage': (drift_detected / (drift_detected + stable_features)) * 100 if (drift_detected + stable_features) > 0 else 0
            }
            
            logger.info(f"Drift detection completed: {drift_detected} features with drift detected")
            
        except Exception as e:
            logger.error(f"Error during drift detection: {str(e)}")
            drift_report['error'] = str(e)
        
        return drift_report
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on domain knowledge.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        logger.info("Engineering features...")
        
        try:
            df_processed = df.copy()
            
            # Create speed-related features if speed columns exist
            speed_cols = ['AvgSp', 'MedSp', 'P5sp', 'P95sp', 'SdSp', 'HvgSp']
            if all(col in df_processed.columns for col in speed_cols):
                logger.info("Creating speed-related features...")
                
                # Speed variability features
                df_processed['speed_variance'] = df_processed['AvgSp'] - df_processed['MedSp']
                df_processed['speed_range'] = df_processed['P95sp'] - df_processed['P5sp']
                df_processed['speed_cv'] = df_processed['SdSp'] / (df_processed['AvgSp'] + 1e-6)  # Coefficient of variation
                
                # Traffic efficiency indicators
                if 'HvgSp' in df_processed.columns:
                    df_processed['harmonic_ratio'] = df_processed['AvgSp'] / (df_processed['HvgSp'] + 1e-6)
            
            # Create traffic density and efficiency features
            if 'debit' in df_processed.columns and 'AvgSp' in df_processed.columns:
                logger.info("Creating traffic flow features...")
                
                # Traffic density estimation: k = q / v
                df_processed['traffic_density'] = df_processed['debit'] / (df_processed['AvgSp'] + 1e-6)
                
                # Traffic efficiency: higher speed with higher flow is more efficient
                df_processed['traffic_efficiency'] = (df_processed['debit'] * df_processed['AvgSp']) / 1000
            
            # Create time-based features if time column exists
            if 'heure' in df_processed.columns:
                logger.info("Creating time-based features...")
                
                # Hour categories
                df_processed['hour_category'] = pd.cut(
                    df_processed['heure'], 
                    bins=[-1, 6, 12, 18, 24], 
                    labels=['Night', 'Morning', 'Afternoon', 'Evening']
                )
                
                # Rush hour indicators
                df_processed['is_rush_hour'] = df_processed['heure'].isin([7, 8, 9, 17, 18, 19]).astype(int)
                
                # Peak/off-peak
                df_processed['is_peak'] = ((df_processed['heure'] >= 7) & (df_processed['heure'] <= 9) | 
                                         (df_processed['heure'] >= 17) & (df_processed['heure'] <= 19)).astype(int)
            
            # Create zone-based features if zone column exists
            if 'zone' in df_processed.columns:
                logger.info("Creating zone-based features...")
                
                # Zone traffic statistics
                zone_stats = df_processed.groupby('zone')['debit'].agg(['mean', 'std']).add_prefix('zone_')
                df_processed = df_processed.merge(zone_stats, left_on='zone', right_index=True, how='left')
                
                # Relative traffic compared to zone average
                if 'debit' in df_processed.columns:
                    df_processed['traffic_vs_zone_avg'] = df_processed['debit'] / (df_processed['zone_mean'] + 1e-6)
            
            # Create sensor-based features if capteur column exists
            if 'capteur' in df_processed.columns:
                logger.info("Creating sensor-based features...")
                
                # Sensor reliability indicators
                sensor_counts = df_processed['capteur'].value_counts()
                df_processed['sensor_frequency'] = df_processed['capteur'].map(sensor_counts)
                
                # Binary encoding for common sensors
                top_sensors = sensor_counts.head(10).index
                for sensor in top_sensors:
                    df_processed[f'is_sensor_{sensor}'] = (df_processed['capteur'] == sensor).astype(int)
            
            # Create ratio features
            ratio_features = [
                ('AvgTt', 'MedTt', 'tt_ratio'),
                ('Hits', 'AvgTt', 'hits_per_time'),
                ('P95sp', 'P5sp', 'speed_percentile_ratio')
            ]
            
            for col1, col2, new_name in ratio_features:
                if col1 in df_processed.columns and col2 in df_processed.columns:
                    df_processed[new_name] = df_processed[col1] / (df_processed[col2] + 1e-6)
            
            logger.info(f"Feature engineering completed. Added {len(df_processed.columns) - len(df.columns)} new features")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values using specified strategy.
        
        Args:
            df: Input dataframe
            strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'mode')
            
        Returns:
            Dataframe with missing values handled
        """
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        try:
            df_processed = df.copy()
            
            if strategy == 'drop':
                df_processed = df_processed.dropna()
            
            elif strategy == 'mean':
                for col in self.numeric_columns:
                    if col in df_processed.columns:
                        df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                
                for col in self.categorical_columns:
                    if col in df_processed.columns:
                        df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
            
            elif strategy == 'median':
                for col in self.numeric_columns:
                    if col in df_processed.columns:
                        df_processed[col].fillna(df_processed[col].median(), inplace=True)
                
                for col in self.categorical_columns:
                    if col in df_processed.columns:
                        df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
            
            elif strategy == 'mode':
                for col in df_processed.columns:
                    if df_processed[col].dtype == 'object':
                        mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                        df_processed[col].fillna(mode_val, inplace=True)
                    else:
                        df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else df_processed[col].mean(), inplace=True)
            
            missing_after = df_processed.isnull().sum().sum()
            logger.info(f"Missing values handled. Remaining missing values: {missing_after}")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using specified method.
        
        Args:
            df: Input dataframe
            method: Method for outlier detection ('iqr', 'zscore')
            factor: Factor for outlier detection threshold
            
        Returns:
            Dataframe with outliers removed
        """
        logger.info(f"Removing outliers using {method} method")
        
        try:
            df_processed = df.copy()
            original_len = len(df_processed)
            
            for col in self.numeric_columns:
                if col in df_processed.columns:
                    if method == 'iqr':
                        Q1 = df_processed[col].quantile(0.25)
                        Q3 = df_processed[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - factor * IQR
                        upper_bound = Q3 + factor * IQR
                        
                        df_processed = df_processed[
                            (df_processed[col] >= lower_bound) & 
                            (df_processed[col] <= upper_bound)
                        ]
                    
                    elif method == 'zscore':
                        z_scores = np.abs(stats.zscore(df_processed[col].dropna()))
                        df_processed = df_processed[z_scores < factor]
            
            removed_count = original_len - len(df_processed)
            logger.info(f"Removed {removed_count} outlier records ({removed_count/original_len*100:.2f}%)")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            return df
    
    def select_features(self, df: pd.DataFrame, target_col: Optional[str] = None, method: str = 'correlation') -> pd.DataFrame:
        """
        Select relevant features for modeling.
        
        Args:
            df: Input dataframe
            target_col: Target column for supervised selection
            method: Feature selection method ('correlation', 'mutual_info', 'rfe')
            
        Returns:
            Dataframe with selected features
        """
        logger.info(f"Selecting features using {method} method")
        
        try:
            if target_col is None:
                target_col = self.target_column
            
            if target_col not in df.columns:
                logger.warning(f"Target column {target_col} not found. Returning original dataframe.")
                return df
            
            df_processed = df.copy()
            
            # Get numeric features for selection
            numeric_features = [col for col in df_processed.columns 
                              if pd.api.types.is_numeric_dtype(df_processed[col]) and col != target_col]
            
            if len(numeric_features) == 0:
                logger.warning("No numeric features found for selection.")
                return df_processed
            
            X = df_processed[numeric_features]
            y = df_processed[target_col]
            
            if method == 'correlation':
                # Remove highly correlated features
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.95)]
                
                selected_features = [col for col in numeric_features if col not in to_drop]
                
            elif method == 'mutual_info':
                from sklearn.feature_selection import mutual_info_regression, SelectKBest
                
                selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(numeric_features)))
                X_selected = selector.fit_transform(X, y)
                selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
                
            elif method == 'rfe':
                # Recursive Feature Elimination
                estimator = RandomForestRegressor(n_estimators=10, random_state=42)
                selector = RFECV(estimator, step=1, cv=3, scoring='neg_mean_squared_error')
                selector.fit(X, y)
                
                selected_features = [numeric_features[i] for i in range(len(numeric_features)) 
                                   if selector.support_[i]]
            
            # Keep target column and selected features
            all_selected = selected_features + [target_col]
            categorical_cols = [col for col in self.categorical_columns if col in df_processed.columns]
            all_selected.extend(categorical_cols)
            
            df_selected = df_processed[all_selected]
            
            logger.info(f"Feature selection completed. Selected {len(selected_features)} features from {len(numeric_features)}")
            
            return df_selected
            
        except Exception as e:
            logger.error(f"Error during feature selection: {str(e)}")
            return df
