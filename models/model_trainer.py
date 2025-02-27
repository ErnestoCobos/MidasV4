import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import asyncio
from typing import Dict, Any, Optional, List, Tuple

from models.model_factory import ModelFactory

class ModelTrainer:
    """
    Automatic model training system for MidasScalping bot
    
    This class handles data collection, feature engineering, model training,
    and model evaluation for both LSTM and XGBoost models.
    """
    
    def __init__(self, config, db_manager=None, binance_client=None):
        """Initialize the model trainer with configuration"""
        self.config = config
        self.logger = logging.getLogger('ModelTrainer')
        self.db = db_manager
        self.binance_client = binance_client
        
        # Estado interno
        self.is_training = False
        self.last_training = {}  # Por símbolo
        
        # Create output directory for saving models
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
    
    async def train_model(self, model_type: str, symbol: str, 
                          days_of_data: int = 7, 
                          test_size: float = 0.2,
                          force_train: bool = False) -> Dict[str, Any]:
        """
        Train a model with historical data
        
        Args:
            model_type: Type of model to train ('lstm' or 'xgboost')
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            days_of_data: Number of days of historical data to use
            test_size: Proportion of data to use for testing
            force_train: Force training even if recently trained
            
        Returns:
            Training results and evaluation metrics
        """
        # Evitar entrenamientos simultáneos
        if self.is_training and not force_train:
            self.logger.warning(f"Training already in progress. Skipping request for {symbol}")
            return {"status": "skipped", "reason": "training_in_progress"}
        
        # Verificar último entrenamiento para evitar entrenamientos frecuentes
        if symbol in self.last_training and not force_train:
            last_time = self.last_training[symbol]
            hours_since_last = (datetime.now() - last_time).total_seconds() / 3600
            
            if hours_since_last < 6:  # No entrenar si se entrenó hace menos de 6 horas
                self.logger.info(f"Model for {symbol} was trained {hours_since_last:.1f} hours ago. Skipping.")
                return {"status": "skipped", "reason": "recently_trained", "hours_since_last": hours_since_last}
        
        self.is_training = True
        self.logger.info(f"Starting automatic training of {model_type} model for {symbol}")
        start_time = time.time()
        
        try:
            # Step 1: Data collection
            self.logger.info(f"Collecting {days_of_data} days of data for {symbol}")
            X_train, X_test, y_train, y_test = await self._collect_and_prepare_data(
                symbol, days_of_data, test_size
            )
            
            # Verificar que haya suficientes datos
            if len(X_train) < 500:
                self.logger.warning(f"Insufficient training data for {symbol}: {len(X_train)} samples")
                self.is_training = False
                return {"status": "error", "reason": "insufficient_data"}
            
            # Step 2: Model initialization
            self.logger.info(f"Initializing {model_type} model")
            model = ModelFactory.create_model(model_type, self.config)
            
            # Step 3: Model training
            self.logger.info(f"Training {model_type} model with {len(X_train)} samples")
            
            if model_type.lower() == 'lstm':
                history = model.train(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=getattr(self.config, 'lstm_epochs', 50),
                    batch_size=getattr(self.config, 'lstm_batch_size', 32)
                )
                training_metrics = {
                    'final_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
                    'epochs_trained': len(history.history['loss'])
                }
            
            elif model_type.lower() == 'xgboost':
                feature_importance = model.train(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test
                )
                training_metrics = {
                    'feature_importance': {str(k): float(v) for k, v in feature_importance.items()},
                    'boost_rounds': model.model.best_iteration,
                }
            
            # Step 4: Model evaluation
            self.logger.info(f"Evaluating {model_type} model")
            evaluation_metrics = self._evaluate_model(model, X_test, y_test)
            
            # Step 5: Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_type}_{symbol}_{timestamp}"
            
            if model_type.lower() == 'lstm':
                model_path = os.path.join(self.models_dir, f"{model_filename}.h5")
            else:  # xgboost
                model_path = os.path.join(self.models_dir, f"{model_filename}.model")
                
            model.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            # Step 6: Register model in database
            if self.db:
                # Determine if model should be active based on performance
                direction_accuracy = evaluation_metrics['direction_accuracy']
                activate_model = direction_accuracy > 0.55  # Activar si precisión > 55%
                
                model_data = {
                    'model_type': model_type,
                    'symbol': symbol,
                    'created_at': datetime.now(),
                    'file_path': model_path,
                    'accuracy': evaluation_metrics['direction_accuracy'] * 100,
                    'direction_accuracy': evaluation_metrics['direction_accuracy'] * 100,
                    'rmse': evaluation_metrics['rmse'],
                    'training_data_start': datetime.now() - timedelta(days=days_of_data),
                    'training_data_end': datetime.now(),
                    'is_active': activate_model,
                    'performance_score': (evaluation_metrics['direction_accuracy'] * 100) - evaluation_metrics['rmse']
                }
                
                model_id = self.db.register_model(model_data)
                self.logger.info(f"Model registered in database with ID {model_id}")
            
            # Return training results
            training_time = time.time() - start_time
            
            # Actualizar último entrenamiento
            self.last_training[symbol] = datetime.now()
            
            result = {
                'status': 'success',
                'model_type': model_type,
                'symbol': symbol,
                'training_time': training_time,
                'training_metrics': training_metrics,
                'evaluation_metrics': evaluation_metrics,
                'model_path': model_path,
                'data_points': len(X_train) + len(X_test),
                'is_active': activate_model if 'activate_model' in locals() else False
            }
            
            self.is_training = False
            return result
        
        except Exception as e:
            self.logger.error(f"Error training {model_type} model for {symbol}: {str(e)}")
            self.is_training = False
            return {"status": "error", "reason": str(e)}
    
    async def _collect_and_prepare_data(self, symbol: str, days: int, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect historical data and prepare features for model training
        
        Args:
            symbol: Trading pair symbol
            days: Number of days of historical data
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Attempt to get data from database first
            if self.db:
                self.logger.info(f"Getting data from database for {symbol}")
                df = self.db.get_training_data(symbol, start_time, end_time)
                
                if len(df) > 1000:  # Sufficient data in database
                    self.logger.info(f"Using {len(df)} records from database for {symbol}")
                    
                    # Prepare features from database data
                    if self.config.model_type.lower() == 'lstm':
                        return self._prepare_lstm_data(df, test_size)
                    else:
                        return self._prepare_xgboost_data(df, test_size)
            
            # If database has insufficient data, get from API
            if self.binance_client:
                self.logger.info(f"Getting data from Binance API for {symbol}")
                klines = await self.binance_client.get_historical_klines(
                    symbol=symbol,
                    interval=self.config.timeframe,
                    start_str=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    end_str=end_time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                if not klines:
                    raise ValueError(f"No data returned from API for {symbol}")
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                # Calculate indicators
                from data.feature_engineer import FeatureEngineer
                feature_engineer = FeatureEngineer(self.config)
                df = feature_engineer.calculate_features(df)
                
                self.logger.info(f"Calculated features for {len(df)} candles from API")
                
                # Prepare target
                df['target'] = df['close'].pct_change(1).shift(-1) * 100
                df = df.dropna()
                
                # Prepare features from API data
                if self.config.model_type.lower() == 'lstm':
                    return self._prepare_lstm_data(df, test_size)
                else:
                    return self._prepare_xgboost_data(df, test_size)
            
            raise ValueError("Neither database nor API client available for data collection")
                
        except Exception as e:
            self.logger.error(f"Error collecting and preparing data: {str(e)}")
            raise
    
    def _prepare_lstm_data(self, df: pd.DataFrame, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare LSTM data sequences
        
        Args:
            df: DataFrame with features
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['target', 'open_time', 'close_time', 'ignore', 'timestamp', 'id', 'market_data_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('indicator')]
        
        # If indicators column exists (from database), expand it
        if 'indicators' in df.columns:
            # Extract indicators from JSON column
            for idx, row in df.iterrows():
                if isinstance(row.indicators, dict):
                    for k, v in row.indicators.items():
                        if k not in df.columns:
                            df[k] = np.nan
                        df.at[idx, k] = v
            
            # Update feature columns
            feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('indicator')]
        
        # Get sequence length from config
        sequence_length = getattr(self.config, 'sequence_length', 60)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            # Get sequence
            seq = df[feature_cols].iloc[i:i+sequence_length].values
            target = df['target'].iloc[i+sequence_length-1]
            
            # Skip sequences with NaN
            if np.isnan(seq).any() or np.isnan(target):
                continue
                
            sequences.append(seq)
            targets.append(target)
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(targets)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def _prepare_xgboost_data(self, df: pd.DataFrame, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare XGBoost data
        
        Args:
            df: DataFrame with features
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['target', 'open_time', 'close_time', 'ignore', 'timestamp', 'id', 'market_data_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('indicator')]
        
        # If indicators column exists (from database), expand it
        if 'indicators' in df.columns:
            # Extract indicators from JSON column
            for idx, row in df.iterrows():
                if isinstance(row.indicators, dict):
                    for k, v in row.indicators.items():
                        if k not in df.columns:
                            df[k] = np.nan
                        df.at[idx, k] = v
            
            # Update feature columns
            feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('indicator')]
        
        # Drop rows with NaN
        df = df.dropna(subset=feature_cols + ['target'])
        
        # Get features and target
        X = df[feature_cols].values
        y = df['target'].values
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def _evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Flatten predictions if needed
        if hasattr(y_pred, 'flatten'):
            y_pred = y_pred.flatten()
        
        # Calculate metrics
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy (% of time prediction has correct sign)
        direction_correct = np.sum(np.sign(y_pred) == np.sign(y_test)) / len(y_test)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'direction_accuracy': float(direction_correct)
        }
    
    async def auto_train_all(self, symbols: List[str] = None, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Train all model types for all configured symbols
        
        Args:
            symbols: List of symbols to train models for (default: config.symbols)
            model_types: List of model types to train (default: ['xgboost', 'lstm'])
            
        Returns:
            Summary of training results
        """
        if symbols is None:
            symbols = self.config.symbols
        
        if model_types is None:
            model_types = ['xgboost', 'lstm']
        
        results = {}
        
        for symbol in symbols:
            symbol_results = {}
            
            for model_type in model_types:
                try:
                    self.logger.info(f"Training {model_type} model for {symbol}")
                    training_result = await self.train_model(model_type, symbol)
                    symbol_results[model_type] = training_result
                    
                    # Log success
                    if training_result.get('status') == 'success':
                        metrics = training_result['evaluation_metrics']
                        direction_acc = metrics['direction_accuracy'] * 100
                        self.logger.info(f"Successfully trained {model_type} model for {symbol}. " +
                                      f"Direction accuracy: {direction_acc:.2f}%")
                    else:
                        self.logger.warning(f"Training {model_type} model for {symbol} skipped or failed: {training_result.get('reason')}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_type} model for {symbol}: {str(e)}")
                    symbol_results[model_type] = {'status': 'error', 'reason': str(e)}
            
            results[symbol] = symbol_results
        
        return results
    
    def start_scheduled_training(self, interval_hours: int = 24):
        """
        Start scheduled training thread
        
        Args:
            interval_hours: Training interval in hours
        """
        self.training_thread = threading.Thread(
            target=self._scheduled_training_loop,
            args=(interval_hours,)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        self.logger.info(f"Scheduled training started with {interval_hours}h interval")
    
    def _scheduled_training_loop(self, interval_hours: int):
        """
        Training loop for scheduled training
        
        Args:
            interval_hours: Training interval in hours
        """
        while True:
            try:
                # Esperar hasta la hora programada (3 AM por defecto)
                current_hour = datetime.now().hour
                if current_hour == 3 or interval_hours < 24:
                    self.logger.info("Starting scheduled model training")
                    # Usar asyncio aquí porque train_all es async
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(self.auto_train_all())
                    loop.close()
                    
                    self.logger.info(f"Scheduled training completed with results: {results}")
                
                # Dormir hasta próxima verificación
                sleep_time = 3600  # 1 hora
                if interval_hours < 24:
                    sleep_time = interval_hours * 3600
                    
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in scheduled training: {str(e)}")
                time.sleep(3600)  # Sleep for 1 hour on error