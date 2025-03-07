import numpy as np
import pandas as pd
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import joblib
from typing import Dict, Any, Optional, Tuple, Union

class MLModule:
    """
    Unified machine learning module integrating XGBoost (with GPU support) and TensorFlow models
    for trading signal generation.
    
    This module handles:
    1. Feature preprocessing for both model types
    2. Model training with GPU acceleration
    3. Prediction generation and signal confidence calculation
    4. Model persistence (saving/loading)
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('MLModule')
        
        # Initialize models as None
        self.xgb_model = None
        self.tf_model = None
        
        # GPU configuration
        self.use_gpu = getattr(config, 'use_gpu', False)
        self.gpu_device = getattr(config, 'gpu_device', 0)
        
        # XGBoost parameters
        self.xgb_rounds = getattr(config, 'xgb_rounds', 100)
        self.xgb_params = None
        
        # TensorFlow parameters
        self.sequence_length = getattr(config, 'sequence_length', 10)
        self.feature_count = getattr(config, 'feature_count', 20)
        
        # Commission and slippage for simulation
        self.commission_rate = getattr(config, 'commission_rate', 0.0004)  # 0.04%
        self.slippage_pct = getattr(config, 'slippage_pct', 0.0002)        # 0.02%
        
        # Feature names for models
        self.feature_names = None
        
        # Configure GPU
        self._configure_gpu()
        
    def _configure_gpu(self):
        """Configure GPU acceleration for TensorFlow and XGBoost"""
        if not self.use_gpu:
            self.logger.info("GPU acceleration disabled in config. Using CPU only.")
            # Default XGBoost parameters for CPU
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'rmse',
                'tree_method': 'hist',  # CPU method
                'n_jobs': -1  # Use all CPU cores
            }
            return
            
        self.logger.info("Attempting to configure GPU acceleration...")
        
        # Configure TensorFlow
        try:
            # Check for TensorFlow GPUs
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                self.logger.info(f"Found {len(gpus)} GPU(s) for TensorFlow: {gpus}")
                
                for gpu in gpus:
                    # Memory growth needs to be set before GPUs have been initialized
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        self.logger.info(f"Memory growth enabled for {gpu}")
                    except RuntimeError as e:
                        self.logger.warning(f"Memory growth configuration failed: {e}")
                
                # Set visible devices if specified
                if hasattr(self, 'gpu_device'):
                    try:
                        tf.config.set_visible_devices(gpus[self.gpu_device], 'GPU')
                        self.logger.info(f"Using GPU device {self.gpu_device} for TensorFlow")
                    except IndexError:
                        self.logger.warning(f"GPU device {self.gpu_device} not found. Using default device.")
                
                self.tf_gpu_available = True
            else:
                self.logger.warning("No GPUs detected for TensorFlow. Using CPU.")
                self.tf_gpu_available = False
                
        except Exception as e:
            self.logger.error(f"Error configuring TensorFlow GPU: {str(e)}")
            self.tf_gpu_available = False
        
        # Configure XGBoost
        try:
            # Default XGBoost parameters
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'rmse',
                'tree_method': 'hist',  # Default CPU method
                'n_jobs': -1  # Use all CPU cores
            }
            
            # Try to use GPU for XGBoost
            if self.use_gpu:
                try:
                    # Test if GPU is available for XGBoost
                    test_params = {'tree_method': 'gpu_hist'}
                    test_data = xgb.DMatrix(np.array([[1, 2, 3]]), label=np.array([1]))
                    test_model = xgb.train(test_params, test_data, num_boost_round=1)
                    
                    # Update parameters for GPU
                    self.xgb_params.update({
                        'tree_method': 'gpu_hist',       # Use GPU for histogram calculation
                        'predictor': 'gpu_predictor',    # Use GPU for prediction
                        'gpu_id': self.gpu_device        # Specify GPU device
                    })
                    
                    self.logger.info("XGBoost GPU acceleration successfully configured (gpu_hist)")
                    self.xgb_gpu_available = True
                    
                except Exception as e:
                    self.logger.warning(f"XGBoost GPU acceleration failed: {str(e)}. Falling back to CPU.")
                    self.xgb_gpu_available = False
            else:
                self.logger.info("XGBoost using CPU as configured.")
                self.xgb_gpu_available = False
                
        except Exception as e:
            self.logger.error(f"Error configuring XGBoost GPU: {str(e)}")
            self.xgb_gpu_available = False
            
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None) -> xgb.Booster:
        """
        Train an XGBoost model with GPU acceleration if available
        
        Args:
            X_train: Training features (DataFrame or numpy array)
            y_train: Training labels (price movement)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Trained XGBoost model
        """
        try:
            # Store feature names if DataFrame is provided
            if isinstance(X_train, pd.DataFrame):
                self.feature_names = X_train.columns.tolist()
                self.logger.info(f"Using {len(self.feature_names)} features: {self.feature_names}")
            
            # Convert to DMatrix format
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # Evaluation list
            evals = [(dtrain, 'train')]
            
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals.append((dval, 'validation'))
            
            # Train model
            self.logger.info(f"Training XGBoost model with {self.xgb_rounds} rounds...")
            self.logger.info(f"GPU acceleration: {'Enabled' if self.xgb_gpu_available else 'Disabled'}")
            
            self.xgb_model = xgb.train(
                self.xgb_params,
                dtrain,
                self.xgb_rounds,
                evals,
                early_stopping_rounds=20,
                verbose_eval=10
            )
            
            # Get feature importance
            importance = self.xgb_model.get_score(importance_type='gain')
            self.logger.info(f"Model trained with {self.xgb_model.best_iteration} boosting rounds")
            
            # Log top features by importance
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_importance[:5]
            self.logger.info(f"Top features: {top_features}")
            
            return self.xgb_model
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def train_lstm(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32) -> tf.keras.Model:
        """
        Train a TensorFlow LSTM model with GPU acceleration if available
        
        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training labels (price movement)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Trained TensorFlow model
        """
        try:
            # Build LSTM model
            self.logger.info("Building LSTM model...")
            self.logger.info(f"GPU acceleration: {'Enabled' if self.tf_gpu_available else 'Disabled'}")
            
            # Check input shape
            if len(X_train.shape) != 3:
                raise ValueError(f"Expected 3D input for LSTM (samples, sequence_length, features), got shape {X_train.shape}")
            
            # Update feature count and sequence length based on actual data
            _, self.sequence_length, self.feature_count = X_train.shape
            
            # Build the model
            model = Sequential([
                LSTM(units=64, return_sequences=True, 
                     input_shape=(self.sequence_length, self.feature_count)),
                Dropout(0.2),
                LSTM(units=32),
                Dropout(0.2),
                Dense(units=16, activation='relu'),
                Dense(units=1, activation='linear')  # Predict price movement
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Prepare callbacks
            callbacks = []
            
            # Early stopping to prevent overfitting
            if X_val is not None and y_val is not None:
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Model checkpoint to save best model
            checkpoint_path = os.path.join('saved_models', 'lstm_checkpoint')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss' if X_val is not None else 'loss',
                mode='min',
                save_weights_only=False
            )
            callbacks.append(model_checkpoint)
            
            # Train the model
            self.logger.info(f"Training LSTM model for {epochs} epochs with batch size {batch_size}...")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Set as current model
            self.tf_model = model
            
            # Log training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            
            self.logger.info(f"LSTM model trained for {len(history.history['loss'])} epochs")
            self.logger.info(f"Final training loss: {final_loss:.6f}")
            if final_val_loss:
                self.logger.info(f"Final validation loss: {final_val_loss:.6f}")
            
            return self.tf_model
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise
    
    def predict_xgboost(self, features) -> float:
        """
        Make prediction with XGBoost model
        
        Args:
            features: Input features (DataFrame, numpy array, or dict)
            
        Returns:
            Predicted price movement as float
        """
        if self.xgb_model is None:
            self.logger.error("XGBoost model not initialized")
            raise ValueError("XGBoost model not initialized")
        
        try:
            # Convert dictionary to features array if necessary
            if isinstance(features, dict):
                features = self._prepare_features_from_dict(features, model_type='xgboost')
            
            # Convert to DMatrix
            if not isinstance(features, xgb.DMatrix):
                dtest = xgb.DMatrix(features)
            else:
                dtest = features
                
            # Get prediction
            prediction = self.xgb_model.predict(dtest)
            
            # Extract result (may be array or single value)
            if isinstance(prediction, np.ndarray):
                result = prediction.item() if prediction.size == 1 else prediction[0]
            else:
                result = prediction
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error making XGBoost prediction: {str(e)}")
            raise
    
    def predict_lstm(self, features) -> float:
        """
        Make prediction with LSTM model
        
        Args:
            features: Input features (numpy array with shape [1, sequence_length, feature_count])
            
        Returns:
            Predicted price movement as float
        """
        if self.tf_model is None:
            self.logger.error("LSTM model not initialized")
            raise ValueError("LSTM model not initialized")
        
        try:
            # Convert dictionary to sequence features if necessary
            if isinstance(features, dict):
                features = self._prepare_features_from_dict(features, model_type='lstm')
            
            # Ensure correct shape [batch_size, sequence_length, feature_count]
            if len(features.shape) == 2:
                # Add batch dimension if missing
                features = np.expand_dims(features, axis=0)
            
            # Check shape
            if features.shape[1] != self.sequence_length or features.shape[2] != self.feature_count:
                self.logger.warning(f"Expected shape [batch, {self.sequence_length}, {self.feature_count}], "
                                  f"got {features.shape}. Attempting to reshape...")
                
                # Try to reshape if possible
                if features.shape[1] * features.shape[2] == self.sequence_length * self.feature_count:
                    features = features.reshape(-1, self.sequence_length, self.feature_count)
                    self.logger.info(f"Reshaped features to {features.shape}")
                else:
                    raise ValueError(f"Cannot reshape features to required dimensions")
            
            # Get prediction
            prediction = self.tf_model.predict(features)
            
            # Extract result
            result = prediction.item() if prediction.size == 1 else prediction[0][0]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making LSTM prediction: {str(e)}")
            raise
    
    def predict(self, features, model_type: str = 'auto') -> Dict[str, Any]:
        """
        Make prediction using specified or available model and return detailed result
        
        Args:
            features: Input features
            model_type: 'xgboost', 'lstm', or 'auto' (uses both if available)
            
        Returns:
            Dictionary with prediction results
        """
        result = {
            'prediction': 0.0,
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'models_used': []
        }
        
        try:
            # Determine which model(s) to use
            use_xgboost = (model_type.lower() in ['xgboost', 'auto'] and self.xgb_model is not None)
            use_lstm = (model_type.lower() in ['lstm', 'auto'] and self.tf_model is not None)
            
            # Make predictions
            predictions = []
            
            if use_xgboost:
                xgb_prediction = self.predict_xgboost(features)
                predictions.append(xgb_prediction)
                result['xgb_prediction'] = xgb_prediction
                result['models_used'].append('xgboost')
                
            if use_lstm:
                lstm_prediction = self.predict_lstm(features)
                predictions.append(lstm_prediction)
                result['lstm_prediction'] = lstm_prediction
                result['models_used'].append('lstm')
            
            # Combine predictions if we have multiple models
            if predictions:
                # Simple average for now, could be weighted based on model performance
                result['prediction'] = sum(predictions) / len(predictions)
                
                # Determine direction and confidence
                if result['prediction'] > 0.01:  # Small positive threshold to reduce noise
                    result['direction'] = 'BUY'
                    # Scale confidence based on prediction strength (0.01 to 0.1 maps to 60% to 100%)
                    result['confidence'] = min(100, 60 + (result['prediction'] * 400))
                elif result['prediction'] < -0.01:  # Small negative threshold to reduce noise
                    result['direction'] = 'SELL'
                    # Scale confidence based on prediction strength (-0.01 to -0.1 maps to 60% to 100%)
                    result['confidence'] = min(100, 60 + (abs(result['prediction']) * 400))
                else:
                    result['direction'] = 'NEUTRAL'
                    result['confidence'] = 0.0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            result['error'] = str(e)
            return result
    
    def _prepare_features_from_dict(self, features_dict: Dict[str, Any], model_type: str = 'xgboost') -> np.ndarray:
        """
        Convert a dictionary of features to the appropriate format for the specified model
        
        Args:
            features_dict: Dictionary of features
            model_type: 'xgboost' or 'lstm'
            
        Returns:
            Numpy array of features in the appropriate format
        """
        if model_type.lower() == 'xgboost':
            # For XGBoost, extract features in a consistent order
            feature_list = []
            
            # Standard set of features to use
            std_features = [
                'sma_7', 'sma_25',
                'bb_upper', 'bb_lower', 'bb_middle',
                'rsi', 'rsi_14',
                'volatility_14',
                'volume_sma', 'current_volume',
                'relative_volume',
                'ma_dist_7', 'ma_dist_14', 'ma_dist_25',
                'return_7', 'return_14', 'return_25',
                'body_size', 'upper_shadow', 'lower_shadow'
            ]
            
            # Extract features in order with fallbacks
            for feature in std_features:
                if feature in features_dict:
                    feature_list.append(features_dict[feature])
                elif feature == 'rsi' and 'rsi_14' in features_dict:
                    feature_list.append(features_dict['rsi_14'])
                elif feature == 'rsi_14' and 'rsi' in features_dict:
                    feature_list.append(features_dict['rsi'])
                else:
                    feature_list.append(0)  # Default value if feature not found
            
            return np.array([feature_list])
            
        elif model_type.lower() == 'lstm':
            # For LSTM, we need historical data in sequence form
            # This is a simplified implementation that assumes historical data is provided
            if 'historical_data' in features_dict:
                # Use provided historical data
                historical_data = features_dict['historical_data']
                
                # Ensure correct shape
                if len(historical_data.shape) == 2:
                    # Add batch dimension
                    historical_data = np.expand_dims(historical_data, axis=0)
                
                return historical_data
            else:
                # No historical data provided, try to create a simple sequence
                # This is just a placeholder - in a real implementation you would need
                # actual historical data to make LSTM predictions
                self.logger.warning("No historical data provided for LSTM prediction, using placeholder")
                
                # Create a placeholder sequence of the right shape
                placeholder = np.zeros((1, self.sequence_length, self.feature_count))
                
                # Fill with current feature values where possible
                feature_idx = 0
                for feature in features_dict.values():
                    if isinstance(feature, (int, float)) and feature_idx < self.feature_count:
                        # Set the most recent value (last in sequence) to the current feature value
                        placeholder[0, -1, feature_idx] = feature
                        feature_idx += 1
                
                return placeholder
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def apply_slippage_and_commission(self, prediction: float, side: str) -> float:
        """
        Apply realistic slippage and commission to prediction
        
        Args:
            prediction: Predicted price movement percentage
            side: 'BUY' or 'SELL'
            
        Returns:
            Adjusted prediction with slippage and commission
        """
        # Apply slippage based on direction
        if side == 'BUY':
            # For buys, price slips upward (worse entry price)
            adjusted_prediction = prediction - self.slippage_pct
        else:
            # For sells, price slips downward (worse entry price)
            adjusted_prediction = prediction + self.slippage_pct
        
        # Apply commission in both directions
        adjusted_prediction -= self.commission_rate
        
        return adjusted_prediction
    
    def save_xgboost(self, filepath: str) -> bool:
        """
        Save XGBoost model to file
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.xgb_model is None:
            self.logger.error("Cannot save uninitialized XGBoost model")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            self.xgb_model.save_model(filepath)
            
            # Save feature names if available
            if self.feature_names:
                feature_file = f"{os.path.splitext(filepath)[0]}_features.joblib"
                joblib.dump(self.feature_names, feature_file)
            
            self.logger.info(f"XGBoost model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {str(e)}")
            return False
    
    def save_lstm(self, filepath: str) -> bool:
        """
        Save LSTM model to file
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.tf_model is None:
            self.logger.error("Cannot save uninitialized LSTM model")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            self.tf_model.save(filepath)
            
            # Save metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'feature_count': self.feature_count
            }
            
            metadata_file = f"{os.path.splitext(filepath)[0]}_metadata.joblib"
            joblib.dump(metadata, metadata_file)
            
            self.logger.info(f"LSTM model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {str(e)}")
            return False
    
    def load_xgboost(self, filepath: str) -> bool:
        """
        Load XGBoost model from file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                self.logger.error(f"XGBoost model file not found: {filepath}")
                return False
                
            # Load model
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(filepath)
            
            # Try to load feature names
            feature_file = f"{os.path.splitext(filepath)[0]}_features.joblib"
            if os.path.exists(feature_file):
                self.feature_names = joblib.load(feature_file)
            
            self.logger.info(f"XGBoost model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {str(e)}")
            return False
    
    def load_lstm(self, filepath: str) -> bool:
        """
        Load LSTM model from file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                self.logger.error(f"LSTM model file not found: {filepath}")
                return False
                
            # Load model
            self.tf_model = load_model(filepath)
            
            # Try to load metadata
            metadata_file = f"{os.path.splitext(filepath)[0]}_metadata.joblib"
            if os.path.exists(metadata_file):
                metadata = joblib.load(metadata_file)
                self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                self.feature_count = metadata.get('feature_count', self.feature_count)
            
            self.logger.info(f"LSTM model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models
        
        Returns:
            Dictionary with model information
        """
        info = {
            'xgboost': {
                'loaded': self.xgb_model is not None,
                'gpu_enabled': self.xgb_gpu_available if hasattr(self, 'xgb_gpu_available') else False,
                'features': self.feature_names
            },
            'lstm': {
                'loaded': self.tf_model is not None,
                'gpu_enabled': self.tf_gpu_available if hasattr(self, 'tf_gpu_available') else False,
                'sequence_length': self.sequence_length,
                'feature_count': self.feature_count
            },
            'simulation': {
                'commission_rate': self.commission_rate,
                'slippage_pct': self.slippage_pct
            }
        }
        
        return info