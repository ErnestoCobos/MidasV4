import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Bidirectional, Conv1D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import random
import json

class RLTradingModel:
    """
    Risk-Aware Reinforcement Learning Model for Intraday Trading
    
    Implements a dueling Q-network with action branching architecture to capture
    fleeting intraday trading opportunities while managing risk.
    """
    
    def __init__(self, config):
        """Initialize the RL model with configuration settings"""
        self.config = config
        self.logger = logging.getLogger('RLTradingModel')
        
        # Parse RL-specific configuration
        self.state_dim = getattr(config, 'rl_state_dim', 30)  # Number of features in state
        self.sequence_length = getattr(config, 'sequence_length', 60)  # Temporal sequence length
        
        # Action space configuration
        self.action_types = getattr(config, 'action_types', 3)  # BUY, SELL, HOLD
        self.position_sizes = getattr(config, 'position_sizes', 5)  # Different position sizes
        
        # Learning parameters
        self.gamma = getattr(config, 'gamma', 0.95)  # Discount factor
        self.learning_rate = getattr(config, 'learning_rate', 0.0001)
        self.batch_size = getattr(config, 'batch_size', 64)
        self.target_update_freq = getattr(config, 'target_update_freq', 1000)
        self.tau = getattr(config, 'tau', 0.001)  # Soft update parameter
        
        # Exploration parameters
        self.epsilon = getattr(config, 'epsilon_start', 1.0)
        self.epsilon_min = getattr(config, 'epsilon_min', 0.1)
        self.epsilon_decay = getattr(config, 'epsilon_decay', 0.995)
        
        # Experience replay
        self.memory_size = getattr(config, 'memory_size', 100000)
        self.memory = deque(maxlen=self.memory_size)
        
        # Training metrics
        self.train_step = 0
        
        # Create networks
        self.primary_network = self._build_network()
        self.target_network = self._build_network()
        
        # Initialize target network weights to match primary network
        self.target_network.set_weights(self.primary_network.get_weights())
        
        self.logger.info("RL trading model initialized successfully")

    def _build_network(self) -> Model:
        """
        Build the dueling Q-network with action branching
        
        Returns:
            Keras Model instance
        """
        # Input shape: [sequence_length, state_dim]
        state_input = Input(shape=(self.sequence_length, self.state_dim))
        
        # Encoder: extract features from temporal market data
        # 1. Convolutional layer to capture short-term patterns
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(state_input)
        conv2 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(state_input)
        
        # 2. Bidirectional LSTM to capture temporal dependencies
        lstm = Bidirectional(LSTM(128, return_sequences=True))(state_input)
        
        # Combine features
        concat_features = Concatenate()([conv1, conv2, lstm])
        
        # Temporal pooling
        pooled_features = GlobalAveragePooling1D()(concat_features)
        dense1 = Dense(256, activation='relu')(pooled_features)
        dense1 = Dropout(0.3)(dense1)
        
        # Dueling architecture
        # 1. Value stream (state value)
        value_stream = Dense(128, activation='relu')(dense1)
        value = Dense(1)(value_stream)
        
        # 2. Advantage streams (one for each action type)
        advantage_streams = []
        for i in range(self.action_types):
            adv = Dense(128, activation='relu')(dense1)
            adv = Dense(self.position_sizes)(adv)
            advantage_streams.append(adv)
        
        # Combine value and advantage streams for each action branch
        outputs = []
        for i, advantage in enumerate(advantage_streams):
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
            q_values = tf.add(value, advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
            outputs.append(q_values)
        
        # Create model
        model = Model(inputs=state_input, outputs=outputs)
        
        # Define optimizer and loss
        losses = [self._huber_loss for _ in range(self.action_types)]
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=losses)
        
        return model

    def _huber_loss(self, y_true, y_pred):
        """
        Huber loss function for robust regression
        """
        delta = 1.0
        error = y_true - y_pred
        condition = tf.abs(error) < delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(condition, squared_loss, linear_loss)

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Args:
            state: Current state
            action: Action taken (tuple of action_type, position_size)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store the experience
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, explore=True):
        """
        Choose an action using epsilon-greedy policy
        
        Args:
            state: Current state
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Tuple of (action_type, position_size)
        """
        # Reshape state for network input if needed
        if state.ndim == 2:
            state = np.expand_dims(state, axis=0)  # Add batch dimension
        
        # Exploration: choose random action
        if explore and np.random.rand() < self.epsilon:
            action_type = np.random.randint(0, self.action_types)
            position_size = np.random.randint(0, self.position_sizes)
            return (action_type, position_size)
        
        # Exploitation: choose best action
        q_values = self.primary_network.predict(state)
        action_type = np.argmax([np.max(q_values[i][0]) for i in range(self.action_types)])
        position_size = np.argmax(q_values[action_type][0])
        
        return (action_type, position_size)

    def replay(self, batch_size=None):
        """
        Train the network on a batch of experiences
        
        Args:
            batch_size: Size of batch to train on (default: self.batch_size)
            
        Returns:
            Loss values
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Need enough experiences to sample a batch
        if len(self.memory) < batch_size:
            return []
        
        # Sample random batch from memory
        batch = random.sample(self.memory, batch_size)
        
        # Initialize lists to hold inputs and targets for each action branch
        states = np.zeros((batch_size, self.sequence_length, self.state_dim))
        next_states = np.zeros((batch_size, self.sequence_length, self.state_dim))
        
        # For each action branch, initialize target arrays
        targets = [np.zeros((batch_size, self.position_sizes)) for _ in range(self.action_types)]
        
        # Fill states and next_states
        for i, (state, _, _, next_state, _) in enumerate(batch):
            states[i] = state
            next_states[i] = next_state
        
        # Get Q-values for next states from target network
        next_q_values = self.target_network.predict(next_states)
        
        # For each experience in batch
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            # Extract action components
            action_type, position_size = action
            
            # For each action branch, copy the predicted Q-values
            for j in range(self.action_types):
                targets[j][i] = self.primary_network.predict(np.expand_dims(state, axis=0))[j]
            
            # Update the Q-value for the action taken
            if done:
                targets[action_type][i, position_size] = reward
            else:
                # Q-learning formula: Q(s,a) = r + γ * max_a' Q(s',a')
                max_next_q = np.max([next_q_values[j][i] for j in range(self.action_types)])
                targets[action_type][i, position_size] = reward + self.gamma * max_next_q
        
        # Train the network
        history = self.primary_network.fit(states, targets, epochs=1, verbose=0)
        
        # Update target network weights (soft update)
        self._update_target_network()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Increment train step
        self.train_step += 1
        
        return history.history['loss']

    def _update_target_network(self):
        """
        Update target network weights using soft update
        τ * primary_weights + (1 - τ) * target_weights
        """
        if self.train_step % self.target_update_freq == 0:
            primary_weights = self.primary_network.get_weights()
            target_weights = self.target_network.get_weights()
            
            for i in range(len(primary_weights)):
                target_weights[i] = self.tau * primary_weights[i] + (1 - self.tau) * target_weights[i]
                
            self.target_network.set_weights(target_weights)
            self.logger.debug("Target network updated")

    def hindsight_experience_replay(self, trajectory):
        """
        Implement Hindsight Experience Replay to augment training data
        
        Args:
            trajectory: List of (state, action, reward, next_state, done) tuples
        """
        # Extract final state and reward from trajectory
        final_state = trajectory[-1][3]
        
        # Relabel rewards based on final outcome
        for i, (state, action, _, next_state, done) in enumerate(trajectory[:-1]):
            # Calculate hindsight reward (e.g., based on final price)
            hindsight_reward = self._calculate_hindsight_reward(state, next_state, final_state)
            
            # Store relabeled experience
            self.memory.append((state, action, hindsight_reward, next_state, done))

    def _calculate_hindsight_reward(self, state, next_state, final_state):
        """
        Calculate reward with hindsight knowledge
        
        This could be based on how close the action was to the optimal action
        given knowledge of the full trajectory.
        """
        # Example implementation: reward based on consistency with final outcome
        # Extract price from state
        current_price = state[0, -1]  # Assuming price is last feature
        final_price = final_state[0, -1]
        
        # Calculate price direction from current to final
        price_direction = np.sign(final_price - current_price)
        
        # Return higher reward if action aligns with final price direction
        return price_direction * 0.5  # Scaled reward

    def predict(self, state):
        """
        Make a prediction for state
        
        Args:
            state: Current state
            
        Returns:
            Expected price movement prediction
        """
        # Reshape state for network input if needed
        if state.ndim == 2:
            state = np.expand_dims(state, axis=0)  # Add batch dimension
        
        try:
            # Get Q-values for all actions
            q_values = self.primary_network.predict(state)
            
            # Convert Q-values to expected price movement
            # Higher Q-values for BUY (action_type=0) indicate expected price increase
            # Higher Q-values for SELL (action_type=1) indicate expected price decrease
            buy_q = np.max(q_values[0][0]) if self.action_types > 0 else 0
            sell_q = np.max(q_values[1][0]) if self.action_types > 1 else 0
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            # Provide default values on error
            buy_q = 0
            sell_q = 0
        
        # Calculate expected price movement (positive = up, negative = down)
        expected_movement = buy_q - sell_q
        
        return expected_movement

    def save(self, filepath):
        """
        Save the model to file
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model architecture and weights
        model_path = filepath
        weights_path = f"{filepath}_weights.h5"
        
        # Save models
        self.primary_network.save_weights(weights_path)
        
        # Save configuration and metadata
        metadata = {
            'state_dim': self.state_dim,
            'sequence_length': self.sequence_length,
            'action_types': self.action_types,
            'position_sizes': self.position_sizes,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }
        
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Model saved to {filepath}")
        
        return model_path

    def load(self, filepath):
        """
        Load the model from file
        
        Args:
            filepath: Path to load the model from
        """
        # Load weights
        weights_path = f"{filepath}_weights.h5"
        self.primary_network.load_weights(weights_path)
        self.target_network.load_weights(weights_path)
        
        # Load configuration and metadata
        with open(f"{filepath}_config.json", 'r') as f:
            metadata = json.load(f)
        
        # Apply configuration
        self.state_dim = metadata['state_dim']
        self.sequence_length = metadata['sequence_length']
        self.action_types = metadata['action_types']
        self.position_sizes = metadata['position_sizes']
        self.gamma = metadata['gamma']
        self.learning_rate = metadata['learning_rate']
        self.epsilon = metadata['epsilon']
        self.train_step = metadata['train_step']
        
        self.logger.info(f"Model loaded from {filepath}")
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train model in supervised mode (for compatibility with model_trainer)
        
        This is a simplified supervised training wrapper around the RL training loop
        to maintain compatibility with the existing model_trainer infrastructure.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history object
        """
        self.logger.info(f"Converting supervised data to reinforcement learning format")
        
        # Create synthetic RL experiences from supervised data
        for i in range(len(X_train)):
            state = X_train[i]
            next_state = X_train[i]  # Use same state since we don't have next state
            
            # Determine action based on y_train direction
            if y_train[i] > 0.1:  # Strong positive movement expected
                action = (0, 4)  # BUY with high confidence
            elif y_train[i] > 0:  # Small positive movement expected
                action = (0, 2)  # BUY with medium confidence
            elif y_train[i] < -0.1:  # Strong negative movement expected
                action = (1, 4)  # SELL with high confidence
            elif y_train[i] < 0:  # Small negative movement expected
                action = (1, 2)  # SELL with medium confidence
            else:
                action = (2, 0)  # HOLD
            
            # Reward is proportional to price movement
            reward = y_train[i]
            
            # Add to replay memory
            self.remember(state, action, reward, next_state, False)
        
        # Train model using replay
        losses = []
        for epoch in range(epochs):
            # Multiple updates per epoch to better utilize the data
            for _ in range(max(1, len(X_train) // batch_size)):
                loss = self.replay(batch_size)
                if loss:
                    losses.append(np.mean(loss))
            
            # Log progress
            if epoch % 5 == 0:
                self.logger.info(f"RL Training Epoch {epoch}/{epochs}, Avg Loss: {np.mean(losses[-batch_size:]):.4f}")
        
        # Create mock history object for compatibility
        history = type('obj', (object,), {
            'history': {
                'loss': losses,
                'val_loss': []
            }
        })
        
        return history