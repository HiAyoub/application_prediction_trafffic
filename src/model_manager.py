"""
Model Management Module
======================

Handles model training, evaluation, and management with MLOps best practices.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
import joblib
from pathlib import Path
import time
from datetime import datetime
import json

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Try to import optional libraries with fallbacks
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - XGBoost models will be disabled")

# LightGBM disabled due to system library dependencies in Replit environment
LGBMRegressor = None
LIGHTGBM_AVAILABLE = False

# Feature selection and preprocessing
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Comprehensive model manager for training, evaluation, and deployment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager with configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.models = {}
        self.trained_models = {}
        self.model_metrics = {}
        self.preprocessors = {}
        
        # Model registry - conditionally include models based on availability
        self.model_registry = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_registry['xgboost'] = XGBRegressor
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.model_registry['lightgbm'] = LGBMRegressor
        
        logger.info("ModelManager initialized successfully")
    
    def prepare_features(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Preparing features for modeling...")
        
        try:
            # Separate features and target
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            # Handle categorical variables with improved encoding
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                logger.info(f"Encoding categorical columns: {categorical_cols}")
                
                # Store encoders for future use
                if not hasattr(self, 'encoders'):
                    self.encoders = {}
                
                X_encoded = X.copy()
                
                for col in categorical_cols:
                    # Handle missing values first - convert to string and then fill
                    X_encoded[col] = X_encoded[col].astype(str).fillna('missing')
                    
                    # Check number of unique values
                    n_unique = X_encoded[col].nunique()
                    
                    if n_unique <= 10:  # Use one-hot encoding for low cardinality
                        # One-hot encode with proper handling
                        dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
                        X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)
                        logger.info(f"One-hot encoded '{col}' with {n_unique} categories")
                    else:  # Use label encoding for high cardinality
                        le = LabelEncoder()
                        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                        self.encoders[col] = le
                        logger.info(f"Label encoded '{col}' with {n_unique} categories")
            else:
                X_encoded = X.copy()
            
            # Handle any remaining non-numeric columns
            for col in X_encoded.columns:
                if not pd.api.types.is_numeric_dtype(X_encoded[col]):
                    try:
                        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
                        logger.info(f"Converted column '{col}' to numeric")
                    except:
                        logger.warning(f"Could not convert column '{col}' to numeric. Dropping.")
                        X_encoded = X_encoded.drop(columns=[col])
            
            # Remove any rows with NaN values
            initial_len = len(X_encoded)
            X_encoded = X_encoded.dropna()
            y = y.loc[X_encoded.index]
            
            if len(X_encoded) < initial_len:
                logger.info(f"Removed {initial_len - len(X_encoded)} rows with missing values")
            
            logger.info(f"Feature preparation completed. Shape: {X_encoded.shape}")
            
            return X_encoded, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, cv_folds: int = 5, 
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Train a single model with comprehensive evaluation.
        
        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Target vector
            test_size: Test set size ratio
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing model results and metrics
        """
        logger.info(f"Training model: {model_name}")
        
        try:
            start_time = time.time()
            
            # Get model class and parameters
            if model_name not in self.model_registry:
                raise ValueError(f"Unknown model: {model_name}")
            
            model_class = self.model_registry[model_name]
            model_params = self.config.get('training', {}).get('models', {}).get(model_name, {}).get('params', {})
            
            # Initialize model
            model = model_class(**model_params)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features for certain models
            scaler = None
            if model_name in ['linear_regression', 'ridge', 'lasso']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'train_r2': r2_score(y_train, y_pred_train)
            }
            
            # Cross-validation
            cv_scores = None
            try:
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=cv_folds, scoring='neg_mean_squared_error'
                )
                cv_scores = np.sqrt(-cv_scores)  # Convert to RMSE
                metrics['cv_rmse_mean'] = np.mean(cv_scores)
                metrics['cv_rmse_std'] = np.std(cv_scores)
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model_name}: {str(e)}")
            
            # Feature importance (if available)
            feature_importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    importance_values = model.feature_importances_
                    feature_names = X.columns.tolist()
                    feature_importance = dict(zip(feature_names, importance_values))
                    # Sort by importance
                    feature_importance = dict(sorted(feature_importance.items(), 
                                                   key=lambda x: x[1], reverse=True))
                elif hasattr(model, 'coef_'):
                    # For linear models, use absolute coefficients
                    coef_values = np.abs(model.coef_)
                    feature_names = X.columns.tolist()
                    feature_importance = dict(zip(feature_names, coef_values))
                    feature_importance = dict(sorted(feature_importance.items(), 
                                                   key=lambda x: x[1], reverse=True))
            except Exception as e:
                logger.warning(f"Could not extract feature importance for {model_name}: {str(e)}")
            
            training_time = time.time() - start_time
            
            # Prepare results
            results = {
                'model_name': model_name,
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'predictions': y_pred_test.tolist(),
                'y_test': y_test.tolist(),
                'X_test': X_test,
                'feature_importance': feature_importance,
                'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
                'training_time': training_time,
                'timestamp': datetime.now().isoformat(),
                'data_shape': X.shape,
                'test_size': test_size,
                'cv_folds': cv_folds,
                'random_state': random_state
            }
            
            # Store trained model
            self.trained_models[model_name] = results
            
            logger.info(f"Model {model_name} training completed in {training_time:.2f}s")
            logger.info(f"Test RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            raise
    
    def transform_new_data(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the same encoders fitted during training.
        
        Args:
            X_new: New data to transform
            
        Returns:
            Transformed data ready for prediction
        """
        if not hasattr(self, 'encoders'):
            logger.warning("No encoders found. Make sure to train a model first.")
            return X_new
        
        X_transformed = X_new.copy()
        
        # Apply the same categorical encoding as during training
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                # Handle missing values
                X_transformed[col] = X_transformed[col].fillna('missing')
                
                # Transform using the fitted encoder
                try:
                    X_transformed[col] = encoder.transform(X_transformed[col].astype(str))
                    logger.info(f"Applied label encoding to column '{col}'")
                except ValueError as e:
                    logger.warning(f"Unknown categories in column '{col}': {str(e)}")
                    # Handle unknown categories by using a default value
                    X_transformed[col] = 0
        
        return X_transformed
    
    def train_multiple_models(self, X: pd.DataFrame, y: pd.Series, 
                            model_names: List[str] = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models and compare their performance.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_names: List of model names to train
            **kwargs: Additional arguments for train_model
            
        Returns:
            Dictionary containing results for all trained models
        """
        if model_names is None:
            model_names = list(self.config.get('training', {}).get('models', {}).keys())
        
        logger.info(f"Training {len(model_names)} models: {model_names}")
        
        all_results = {}
        
        for model_name in model_names:
            try:
                results = self.train_model(model_name, X, y, **kwargs)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        if all_results:
            # Find best model based on test RMSE
            best_model = min(all_results.keys(), 
                           key=lambda x: all_results[x]['metrics']['rmse'])
            logger.info(f"Best performing model: {best_model}")
        
        return all_results
    
    def save_model(self, model_name: str, output_dir: str = "models") -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            output_dir: Directory to save the model
            
        Returns:
            Path to saved model file
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_data = self.trained_models[model_name]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model object
        model_file = output_path / f"{model_name}_{timestamp}.joblib"
        joblib.dump({
            'model': model_data['model'],
            'scaler': model_data['scaler'],
            'feature_names': model_data['X_test'].columns.tolist(),
            'metadata': {
                'model_name': model_name,
                'timestamp': model_data['timestamp'],
                'metrics': model_data['metrics'],
                'data_shape': model_data['data_shape']
            }
        }, model_file)
        
        # Save metrics separately
        metrics_file = output_path / f"{model_name}_{timestamp}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'metrics': model_data['metrics'],
                'feature_importance': model_data['feature_importance'],
                'training_time': model_data['training_time'],
                'timestamp': model_data['timestamp'],
                'data_shape': model_data['data_shape']
            }, f, indent=2)
        
        logger.info(f"Model {model_name} saved to {model_file}")
        
        return str(model_file)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Dictionary containing loaded model and metadata
        """
        logger.info(f"Loading model from {model_path}")
        
        try:
            loaded_data = joblib.load(model_path)
            
            logger.info(f"Model loaded successfully: {loaded_data['metadata']['model_name']}")
            
            return loaded_data
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the trained model
            X: Feature matrix for prediction
            
        Returns:
            Array of predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model_data = self.trained_models[model_name]
        if isinstance(model_data, dict):
            model = model_data['model']
        else:
            model = model_data
        scaler = model_data['scaler']
        
        # Apply scaling if used during training
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison table of all trained models.
        
        Returns:
            DataFrame containing model comparison metrics
        """
        if not self.trained_models:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, model_data in self.trained_models.items():
            metrics = model_data['metrics']
            comparison_data.append({
                'Model': model_name,
                'Test_MAE': round(metrics['mae'], 4),
                'Test_RMSE': round(metrics['rmse'], 4),
                'Test_R2': round(metrics['r2'], 4),
                'Train_R2': round(metrics['train_r2'], 4),
                'CV_RMSE_Mean': round(metrics.get('cv_rmse_mean', 0), 4),
                'CV_RMSE_Std': round(metrics.get('cv_rmse_std', 0), 4),
                'Training_Time': round(model_data['training_time'], 2),
                'Timestamp': model_data['timestamp']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for selection ('rmse', 'mae', 'r2')
            
        Returns:
            Tuple of (best_model_name, best_model_data)
        """
        if not self.trained_models:
            raise ValueError("No trained models available")
        
        if metric in ['rmse', 'mae']:
            # Lower is better
            best_model_name = min(self.trained_models.keys(), 
                                key=lambda x: self.trained_models[x]['metrics'][metric])
        elif metric == 'r2':
            # Higher is better
            best_model_name = max(self.trained_models.keys(), 
                                key=lambda x: self.trained_models[x]['metrics'][metric])
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return best_model_name, self.trained_models[best_model_name]
