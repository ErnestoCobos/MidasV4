import logging
from typing import Optional, Tuple
import os

from models.lstm_model import LSTMModel
from models.xgboost_model import XGBoostModel

class ModelFactory:
    """Factory for creating prediction models"""
    
    @staticmethod
    def create_model(model_type: str, config):
        """
        Create a model instance based on type
        
        Args:
            model_type: Type of model ('lstm', 'xgboost', 'deepscalper', 'rl', 'ensemble')
            config: Configuration object
            
        Returns:
            Model instance
        """
        logger = logging.getLogger('ModelFactory')
        
        # Log GPU configuration
        if hasattr(config, 'use_gpu') and config.use_gpu:
            logger.info(f"GPU acceleration enabled in configuration (device: {getattr(config, 'gpu_device', 0)})")
        else:
            logger.info("GPU acceleration disabled in configuration")
        
        if model_type.lower() == 'lstm':
            logger.info("Creating LSTM model")
            return LSTMModel(config)
        
        elif model_type.lower() == 'xgboost':
            logger.info("Creating XGBoost model")
            return XGBoostModel(config)
            
        elif model_type.lower() == 'deepscalper' or model_type.lower() == 'rl':
            logger.info("Creating reinforcement learning model")
            from models.deep_scalper import RLTradingModel
            return RLTradingModel(config)
        
        elif model_type.lower() == 'ensemble':
            # For ensemble, we could create multiple models and combine them
            # This is a placeholder for future implementation
            logger.info("Creating ensemble model")
            raise NotImplementedError("Ensemble model not implemented yet")
        
        elif model_type.lower() == 'indicator':
            logger.info("Using indicator-based strategy (no ML model)")
            return None
            
        elif model_type.lower() == 'llm':
            logger.info("Using LLM-based strategy with Vultr Inference API (no local model)")
            return None
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def load_model(model_type: str, config, filepath: str):
        """
        Load a model from file
        
        Args:
            model_type: Type of model ('lstm', 'xgboost', or 'ensemble')
            config: Configuration object
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        logger = logging.getLogger('ModelFactory')
        
        # Check if model file exists
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Create a new model instance
        model = ModelFactory.create_model(model_type, config)
        
        # Load weights/parameters from file
        try:
            logger.info(f"Loading {model_type} model from {filepath}")
            model.load(filepath)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @staticmethod
    def is_gpu_available() -> Tuple[bool, bool]:
        """
        Check if GPU is available for any of the supported model types
        
        Returns:
            Tuple of (tensorflow_gpu, xgboost_gpu) availability as booleans
        """
        logger = logging.getLogger('ModelFactory')
        
        # Check TensorFlow GPU
        tf_gpu_available = False
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            tf_gpu_available = len(gpus) > 0
            if tf_gpu_available:
                logger.info(f"TensorFlow detected {len(gpus)} GPU(s): {gpus}")
            else:
                logger.info("No GPU detected for TensorFlow")
        except Exception as e:
            logger.warning(f"Error checking TensorFlow GPU: {str(e)}")
        
        # Check XGBoost GPU
        xgb_gpu_available = False
        try:
            import xgboost as xgb
            try:
                # Try to create a small test model with GPU
                test_params = {'tree_method': 'gpu_hist'}
                test_data = xgb.DMatrix(
                    [[1, 2, 3]], 
                    label=[1]
                )
                test_model = xgb.train(test_params, test_data, num_boost_round=1)
                xgb_gpu_available = True
                logger.info("XGBoost GPU acceleration available")
            except Exception as e:
                logger.info(f"XGBoost GPU not available: {str(e)}")
        except Exception as e:
            logger.warning(f"Error checking XGBoost GPU: {str(e)}")
        
        return (tf_gpu_available, xgb_gpu_available)