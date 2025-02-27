import numpy as np
import xgboost as xgb
import pandas as pd
import joblib
import logging
import os
from typing import Dict, Any, Optional, Tuple

class XGBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_names = None
        self.logger = logging.getLogger('XGBoostModel')
        
        # Check for GPU acceleration
        self._configure_gpu()
        
        # Number of boosting rounds
        self.num_rounds = config.xgb_rounds if hasattr(config, 'xgb_rounds') else 100
    
    def _configure_gpu(self):
        """Configure XGBoost to use GPU if available"""
        try:
            # Default parameters with CPU
            self.params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'rmse',
                'tree_method': 'hist',  # Default CPU method
                'n_jobs': -1  # Use all CPU cores
            }
            
            # Check if GPU is enabled in config
            if hasattr(self.config, 'use_gpu') and self.config.use_gpu:
                # Try to use GPU acceleration
                # First try with CUDA
                try:
                    # Create a small test model with GPU
                    test_params = {'tree_method': 'gpu_hist'}
                    test_data = xgb.DMatrix(np.array([[1, 2, 3]]), label=np.array([1]))
                    test_model = xgb.train(test_params, test_data, num_boost_round=1)
                    
                    # If successful, update parameters for GPU
                    self.params.update({
                        'tree_method': 'gpu_hist',  # Use GPU for histogram calculation
                        'predictor': 'gpu_predictor',  # Use GPU for prediction
                        'gpu_id': getattr(self.config, 'gpu_id', 0)  # Default to first GPU
                    })
                    
                    self.logger.info(f"XGBoost GPU acceleration enabled with CUDA (gpu_hist)")
                    
                except Exception as e:
                    self.logger.warning(f"CUDA GPU acceleration failed: {str(e)}")
                    self.logger.info("Falling back to CPU acceleration")
            else:
                self.logger.info("Using CPU for XGBoost (GPU not enabled in config)")
        
        except Exception as e:
            self.logger.error(f"Error configuring GPU for XGBoost: {str(e)}")
            self.logger.warning("Using default CPU configuration")
    
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]:
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of feature importances
        """
        try:
            # Store feature names if DataFrame is provided
            if isinstance(X_train, pd.DataFrame):
                self.feature_names = X_train.columns.tolist()
            
            # Convert to DMatrix format
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # Evaluation list
            evals = [(dtrain, 'train')]
            
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals.append((dval, 'validation'))
            
            # Train model
            self.model = xgb.train(
                self.params,
                dtrain,
                self.num_rounds,
                evals,
                early_stopping_rounds=20,
                verbose_eval=10
            )
            
            # Get feature importance
            importance = self.model.get_score(importance_type='gain')
            self.logger.info(f"Model trained with {self.model.best_iteration} boosting rounds")
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X: Input features
            
        Returns:
            Numpy array of predictions
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            raise ValueError("Model not initialized")
        
        # Convert to DMatrix
        if not isinstance(X, xgb.DMatrix):
            dtest = xgb.DMatrix(X)
        else:
            dtest = X
            
        return self.model.predict(dtest)
    
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            self.logger.error("Cannot save uninitialized model")
            raise ValueError("Cannot save uninitialized model")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            self.model.save_model(filepath)
            
            # Save feature names if available
            if self.feature_names:
                feature_file = f"{os.path.splitext(filepath)[0]}_features.joblib"
                joblib.dump(self.feature_names, feature_file)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, filepath: str) -> None:
        """
        Load model from file
        
        Args:
            filepath: Path to the saved model
        """
        try:
            # Load model
            self.model = xgb.Booster()
            self.model.load_model(filepath)
            
            # Try to load feature names
            feature_file = f"{os.path.splitext(filepath)[0]}_features.joblib"
            if os.path.exists(feature_file):
                self.feature_names = joblib.load(feature_file)
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_feature_importance(self, plot: bool = False) -> Dict[str, float]:
        """
        Get feature importance
        
        Args:
            plot: Whether to plot feature importance
            
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            raise ValueError("Model not initialized")
            
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        
        # If feature names are available, map them to importances
        if self.feature_names:
            importance = {self.feature_names[int(k.replace('f', ''))]: v for k, v in importance.items()}
        
        # Plot if requested
        if plot:
            import matplotlib.pyplot as plt
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, values = zip(*sorted_importance)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(features)), values, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
        
        return importance