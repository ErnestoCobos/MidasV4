import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging
import os

class LSTMModel:
    def __init__(self, config):
        self.config = config
        self.sequence_length = config.sequence_length
        self.feature_count = config.feature_count
        self.model = None
        self.logger = logging.getLogger('LSTMModel')
        
        # Enable GPU acceleration if available
        self._configure_gpu()
        
        # Create model if not loading from file
        self._build_model()
    
    def _configure_gpu(self):
        """Configure TensorFlow to use GPU if available"""
        try:
            # Check if GPUs are available
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                # Log all available GPUs
                self.logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
                
                for gpu in gpus:
                    # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        self.logger.info(f"Enabled memory growth for GPU: {gpu}")
                    except RuntimeError as e:
                        self.logger.warning(f"Error setting memory growth: {e}")
                
                # Log success
                self.logger.info("GPU acceleration enabled for LSTM model")
                
                # Set visible devices if specified in config
                if hasattr(self.config, 'gpu_device') and self.config.gpu_device is not None:
                    tf.config.set_visible_devices(gpus[self.config.gpu_device], 'GPU')
                    self.logger.info(f"Using GPU device {self.config.gpu_device}")
            else:
                self.logger.warning("No GPUs found. Running on CPU only.")
        
        except Exception as e:
            self.logger.error(f"Error configuring GPU: {str(e)}")
            self.logger.warning("Falling back to CPU execution")
    
    def _build_model(self):
        """Build the LSTM model"""
        try:
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
                optimizer=Adam(learning_rate=0.001), 
                loss='mse', 
                metrics=['mae']
            )
            
            self.model = model
            self.logger.info("LSTM model built successfully")
            
        except Exception as e:
            self.logger.error(f"Error building LSTM model: {str(e)}")
            raise
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        try:
            # Ensure model is built
            if self.model is None:
                self._build_model()
            
            callbacks = []
            
            # Early stopping to prevent overfitting
            if X_val is not None and y_val is not None:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Fit model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info(f"Model trained for {len(history.history['loss'])} epochs")
            return history
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Make predictions with the model
        
        Args:
            X: Input features (shape: [samples, sequence_length, features])
            
        Returns:
            Numpy array of predictions
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            raise ValueError("Model not initialized")
            
        return self.model.predict(X)
    
    def save(self, filepath):
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
            self.model.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to the saved model
        """
        try:
            self.model = load_model(filepath)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise