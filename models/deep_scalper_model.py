import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Conv1D, GlobalAveragePooling1D,
    Dropout, Concatenate, Flatten
)
from tensorflow.keras.optimizers import Adam
import logging
import os
import json
import time
import random
import signal
import sys
from collections import deque
from typing import List, Tuple, Dict, Any, Optional

class DeepScalperModel:
    """
    DeepScalper-style RL Model:
    - Multi-input architecture (micro + macro + private state)
    - Dueling Q-network with action branching
    - Auxiliary task of volatility prediction
    - Hindsight reward with retrospective bonus
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('DeepScalperModel')

        # Multi-modal state configuration
        self.micro_dim = getattr(config, 'micro_dim', 20)     # e.g. LOB features, high-freq data
        self.macro_dim = getattr(config, 'macro_dim', 11)     # e.g. technical indicators
        self.private_dim = getattr(config, 'private_dim', 3)  # e.g. position, capital, time remaining

        # Sequence lengths for micro and macro data
        self.micro_seq_len = getattr(config, 'micro_seq_len', 30)
        self.macro_seq_len = getattr(config, 'macro_seq_len', 30)

        # Action space configuration 
        # [direction: buy/sell/hold] and [size: 5 discrete levels]
        self.action_branches = getattr(config, 'action_branches', 2)
        self.branch_sizes = getattr(config, 'branch_sizes', [3, 5])  

        # Auxiliary task configuration
        self.predict_volatility = getattr(config, 'predict_volatility', True)
        self.volatility_weight = getattr(config, 'volatility_weight', 0.5)

        # Hindsight bonus configuration
        self.h_bonus = getattr(config, 'h_bonus', 10)  # Horizon for bonus
        self.lambda_bonus = getattr(config, 'lambda_bonus', 0.3)  # Weight for bonus
        self.her_strategies = getattr(config, 'her_strategies', ['future', 'final', 'random'])
        self.her_k = getattr(config, 'her_k', 4)  # Number of goals to sample

        # RL parameters
        self.gamma = getattr(config, 'gamma', 0.95)
        self.learning_rate = getattr(config, 'learning_rate', 1e-4)
        self.batch_size = getattr(config, 'batch_size', 64)
        self.target_update_freq = getattr(config, 'target_update_freq', 1000)
        self.tau = getattr(config, 'tau', 0.001)  # Soft update parameter

        # Prioritized Experience Replay parameters
        self.memory_size = getattr(config, 'memory_size', 100000)
        self.alpha = getattr(config, 'per_alpha', 0.6)  # Priority exponent
        self.beta = getattr(config, 'per_beta', 0.4)  # Importance sampling exponent
        self.beta_increment = getattr(config, 'beta_increment', 0.001)  # Beta increment per sampling
        self.epsilon_per = getattr(config, 'epsilon_per', 0.01)  # Small constant to avoid zero priorities
        self.memory = []  # List for PER instead of deque
        self.priorities = np.ones((self.memory_size,), dtype=np.float32) * self.epsilon_per
        self.memory_index = 0  # Current position in circular buffer

        # Exploration parameters
        self.epsilon = getattr(config, 'epsilon_start', 1.0)
        self.epsilon_min = getattr(config, 'epsilon_min', 0.1)
        self.epsilon_decay = getattr(config, 'epsilon_decay', 0.995)

        # Training metrics
        self.train_step = 0
        
        # Checkpoint and save paths
        self.checkpoint_dir = getattr(config, 'checkpoint_dir', 'saved_models/checkpoints')
        self.save_interval_steps = getattr(config, 'save_interval_steps', 1000)
        self.emergency_save_path = f"{self.checkpoint_dir}/emergency_save"
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        # Create networks
        self.primary_network = self._build_network()
        self.target_network = self._build_network()
        
        # Copy weights to target network
        self.target_network.set_weights(self.primary_network.get_weights())
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.logger.info("DeepScalper model initialized successfully")
        
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        # Handle SIGINT (Ctrl+C) and SIGTERM
        try:
            signal.signal(signal.SIGINT, self._handle_shutdown_signal)
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
            self.logger.info("Registered signal handlers for graceful shutdown")
        except (ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Failed to register signal handlers: {str(e)}")
            
    def _handle_shutdown_signal(self, sig, frame):
        """Handle shutdown signal by saving model before exit"""
        self.logger.warning(f"Received shutdown signal {sig}, performing emergency save...")
        try:
            # Create timestamp for unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.emergency_save_path}_{timestamp}"
            self.save(save_path)
            self.logger.info(f"Emergency save successful to {save_path}")
        except Exception as e:
            self.logger.error(f"Emergency save failed: {str(e)}")
        
        # Exit with status code
        sys.exit(0)

    def _build_network(self) -> Model:
        """
        Build the multi-input dueling Q-network with action branching
        and optional volatility prediction branch
        """
        # Input layers
        micro_input = Input(shape=(self.micro_seq_len, self.micro_dim), name='micro_input')
        macro_input = Input(shape=(self.macro_seq_len, self.macro_dim), name='macro_input')
        private_input = Input(shape=(self.private_dim,), name='private_input')

        # ---- Micro state processing (high-frequency data) ----
        # Convolutional layers for pattern recognition
        micro_conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(micro_input)
        micro_conv2 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(micro_input)
        
        # Bidirectional LSTM for temporal dependencies
        micro_lstm = Bidirectional(LSTM(64, return_sequences=True))(micro_input)
        
        # Combine features and apply global pooling
        micro_concat = Concatenate()([micro_conv1, micro_conv2, micro_lstm])
        micro_features = GlobalAveragePooling1D()(micro_concat)
        micro_features = Dense(128, activation='relu')(micro_features)
        micro_features = Dropout(0.3)(micro_features)

        # ---- Macro state processing (technical indicators) ----
        # Similar architecture but might have different sizes
        macro_conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(macro_input)
        macro_conv2 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(macro_input)
        
        # LSTM for temporal dependencies
        macro_lstm = Bidirectional(LSTM(64, return_sequences=True))(macro_input)
        
        # Combine and pool
        macro_concat = Concatenate()([macro_conv1, macro_conv2, macro_lstm])
        macro_features = GlobalAveragePooling1D()(macro_concat)
        macro_features = Dense(128, activation='relu')(macro_features)
        macro_features = Dropout(0.3)(macro_features)

        # ---- Private state processing (position, capital) ----
        # Simple feed-forward for private state
        private_features = Dense(64, activation='relu')(private_input)
        private_features = Dense(64, activation='relu')(private_features)
        private_features = Dropout(0.2)(private_features)

        # ---- Merge all features ----
        merged_features = Concatenate()([micro_features, macro_features, private_features])
        merged_features = Dense(256, activation='relu')(merged_features)
        merged_features = Dropout(0.3)(merged_features)

        # ---- Dueling architecture ----
        # 1. Value stream (state value)
        value_stream = Dense(128, activation='relu')(merged_features)
        value = Dense(1, name='value')(value_stream)
        
        # 2. Advantage streams (one for each action branch)
        advantage_streams = []
        
        for i in range(self.action_branches):
            adv = Dense(128, activation='relu')(merged_features)
            adv = Dense(self.branch_sizes[i], name=f'advantage_{i}')(adv)
            advantage_streams.append(adv)
        
        # Combine value and advantage streams
        q_outputs = []
        for i, advantage in enumerate(advantage_streams):
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
            q_values = tf.add(
                value, 
                advantage - tf.reduce_mean(advantage, axis=1, keepdims=True),
                name=f'q_values_{i}'
            )
            q_outputs.append(q_values)
        
        # ---- Optional volatility prediction ----
        outputs = q_outputs.copy()
        
        if self.predict_volatility:
            volatility_output = Dense(1, name='volatility')(merged_features)
            outputs.append(volatility_output)
        
        # Create model
        inputs = [micro_input, macro_input, private_input]
        model = Model(inputs=inputs, outputs=outputs)
        
        # Define losses for each output
        losses = {}
        
        # Q-value losses (Huber loss)
        for i in range(self.action_branches):
            losses[f'q_values_{i}'] = self._huber_loss
        
        # Volatility loss (MSE)
        if self.predict_volatility:
            losses['volatility'] = 'mse'
        
        # Define loss weights (higher weight on Q-values)
        loss_weights = {}
        
        for i in range(self.action_branches):
            loss_weights[f'q_values_{i}'] = 1.0
        
        if self.predict_volatility:
            loss_weights['volatility'] = self.volatility_weight
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=losses,
            loss_weights=loss_weights
        )
        
        return model

    def _huber_loss(self, y_true, y_pred):
        """Huber loss for robust regression"""
        delta = 1.0
        error = y_true - y_pred
        condition = tf.abs(error) < delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(condition, squared_loss, linear_loss)

    def _get_priority(self, error):
        """Convert TD error to priority value"""
        return (np.abs(error) + self.epsilon_per) ** self.alpha
        
    def remember(self, state, action, reward, next_state, vol_label=None, done=False, error=None):
        """
        Store experience with priority in replay memory
        
        Args:
            state: Tuple of (micro_state, macro_state, private_state)
            action: Tuple of actions, one per branch
            reward: Reward received
            next_state: Next state, also a tuple
            vol_label: Volatility label (optional)
            done: Whether episode is done
            error: TD error for prioritized replay (optional)
        """
        if vol_label is None and self.predict_volatility:
            # Default volatility label
            vol_label = 0.0
            
        # Store experience
        experience = (state, action, reward, next_state, vol_label, done)
        
        # If memory is not full, add new experience
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
            # Get max priority for new experience
            max_priority = np.max(self.priorities) if self.memory else 1.0
            self.priorities[len(self.memory)-1] = max_priority
        else:
            # Replace old experience
            self.memory[self.memory_index] = experience
            if error is not None:
                # If error is provided, use it to calculate priority
                self.priorities[self.memory_index] = self._get_priority(error)
            else:
                # Otherwise use max priority for new experience
                max_priority = np.max(self.priorities)
                self.priorities[self.memory_index] = max_priority
            
            # Update index
            self.memory_index = (self.memory_index + 1) % self.memory_size

    def choose_action(self, state, explore=True):
        """
        Choose action using epsilon-greedy policy with volatility-adaptive exploration
        
        Args:
            state: Tuple of (micro_state, macro_state, private_state)
            explore: Whether to use exploration
            
        Returns:
            Tuple of actions, one per branch
        """
        # Extract states
        micro_state, macro_state, private_state = state
        
        # Add batch dimension if needed
        if micro_state.ndim == 2:
            micro_state = np.expand_dims(micro_state, axis=0)
        if macro_state.ndim == 2:
            macro_state = np.expand_dims(macro_state, axis=0)
        if private_state.ndim == 1:
            private_state = np.expand_dims(private_state, axis=0)
        
        # Get model outputs
        outputs = self.primary_network.predict([micro_state, macro_state, private_state])
        
        # Extract Q-values and volatility
        if self.predict_volatility:
            q_values = outputs[:-1]
            volatility = outputs[-1][0][0]  # Extract scalar
            
            # Adapt exploration rate based on volatility
            # Higher volatility → more exploration
            if explore:
                vol_scale = min(3.0, max(0.5, volatility * 2.0))  # Scale volatility to [0.5, 3.0]
                adaptive_epsilon = min(1.0, self.epsilon * vol_scale)
            else:
                adaptive_epsilon = 0.0
        else:
            q_values = outputs
            volatility = None
            adaptive_epsilon = self.epsilon if explore else 0.0
        
        # Exploration: choose random action with adaptive epsilon
        if explore and np.random.rand() < adaptive_epsilon:
            actions = []
            for branch_size in self.branch_sizes:
                actions.append(np.random.randint(0, branch_size))
            return tuple(actions)
        
        # Exploitation: choose best action
        # Select best action for each branch
        actions = []
        for i, branch_q in enumerate(q_values):
            actions.append(np.argmax(branch_q[0]))
        
        # If predicting volatility, scale second branch (size) inversely with volatility
        # High volatility → reduce position size
        if self.predict_volatility and volatility is not None and len(actions) > 1 and volatility > 1.5:
            # Scale down position size when volatility is high
            size_action = actions[1]
            reduced_size = max(0, size_action - int(min(size_action, volatility - 1.0)))
            actions[1] = reduced_size
        
        return tuple(actions)

    def replay(self, batch_size=None):
        """
        Train network on a batch of experiences with prioritized sampling
        
        Args:
            batch_size: Batch size (default: self.batch_size)
            
        Returns:
            Loss values
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Need enough experiences
        if len(self.memory) < batch_size:
            return []
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.memory)]
        probs = priorities / np.sum(priorities)
        
        # Sample batch of indices with priority probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)  # Anneal beta
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        
        # Initialize input arrays
        micro_batch = np.zeros((batch_size, self.micro_seq_len, self.micro_dim))
        macro_batch = np.zeros((batch_size, self.macro_seq_len, self.macro_dim))
        private_batch = np.zeros((batch_size, self.private_dim))
        
        # Initialize target arrays for each branch
        branch_targets = []
        for size in self.branch_sizes:
            branch_targets.append(np.zeros((batch_size, size)))
        
        # Volatility target if needed
        vol_targets = None
        if self.predict_volatility:
            vol_targets = np.zeros((batch_size, 1))
        
        # Next state arrays for target Q-values
        micro_next = np.zeros((batch_size, self.micro_seq_len, self.micro_dim))
        macro_next = np.zeros((batch_size, self.macro_seq_len, self.macro_dim))
        private_next = np.zeros((batch_size, self.private_dim))
        
        # Fill in the arrays
        batch = [self.memory[idx] for idx in indices]
        
        # TD errors for priority updates
        td_errors = np.zeros(batch_size)
        
        # Fill in the arrays
        for i, (state, action, reward, next_state, vol_label, done) in enumerate(batch):
            # Unpack states
            micro_state, macro_state, private_state = state
            micro_next_state, macro_next_state, private_next_state = next_state
            
            # Fill current state
            micro_batch[i] = micro_state
            macro_batch[i] = macro_state
            private_batch[i] = private_state
            
            # Fill next state
            micro_next[i] = micro_next_state
            macro_next[i] = macro_next_state
            private_next[i] = private_next_state
            
            # Fill volatility label if needed
            if self.predict_volatility:
                vol_targets[i, 0] = vol_label
        
        # Get next Q-values from target network
        next_q_values = self.target_network.predict([micro_next, macro_next, private_next])
        
        # If predicting volatility, remove volatility output
        if self.predict_volatility:
            next_q_branches = next_q_values[:-1]
        else:
            next_q_branches = next_q_values
        
        # Get current Q-values from primary network
        current_q_values = self.primary_network.predict([micro_batch, macro_batch, private_batch])
        
        # If predicting volatility, remove volatility output
        if self.predict_volatility:
            current_q_branches = current_q_values[:-1]
        else:
            current_q_branches = current_q_values
        
        # Update Q-values for each experience and compute TD errors
        for i, (state, action, reward, next_state, vol_label, done) in enumerate(batch):
            # For each action branch
            for b_idx, act_idx in enumerate(action):
                if done:
                    # If terminal state, just use reward
                    target_q = reward
                else:
                    # Otherwise, use Bellman equation
                    # Find max Q-value for next state
                    max_next_q = np.max(next_q_branches[b_idx][i])
                    
                    # Q(s,a) = r + γ * max_a' Q(s',a')
                    target_q = reward + self.gamma * max_next_q
                
                # Calculate TD error for updating priorities (using first branch)
                if b_idx == 0:
                    td_errors[i] = np.abs(target_q - current_q_branches[b_idx][i, act_idx])
                
                # Apply importance sampling weight to error
                current_q_branches[b_idx][i, act_idx] = current_q_branches[b_idx][i, act_idx] + \
                    weights[i] * (target_q - current_q_branches[b_idx][i, act_idx])
        
        # Update priorities
        for i, idx in enumerate(indices):
            self.priorities[idx] = self._get_priority(td_errors[i])
        
        # Prepare inputs and targets for training
        inputs = [micro_batch, macro_batch, private_batch]
        
        # Update branch targets with corrected Q-values
        for b_idx in range(self.action_branches):
            branch_targets[b_idx] = current_q_branches[b_idx]
        
        # Prepare targets based on output names in model
        targets = {}
        for i in range(self.action_branches):
            targets[f'q_values_{i}'] = branch_targets[i]
        
        if self.predict_volatility:
            targets['volatility'] = vol_targets
        
        # Train the network
        history = self.primary_network.fit(
            inputs, targets,
            epochs=1, batch_size=batch_size, verbose=0
        )
        
        # Update target network
        self._update_target_network()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Increment train step
        self.train_step += 1
        
        # Periodic checkpoint save during training
        if self.train_step % self.save_interval_steps == 0:
            try:
                checkpoint_path = f"{self.checkpoint_dir}/checkpoint_step_{self.train_step}"
                self.save(checkpoint_path)
                self.logger.info(f"Saved checkpoint at training step {self.train_step}")
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint: {str(e)}")
        
        return history.history['loss']

    def _update_target_network(self):
        """Update target network with soft update"""
        if self.train_step % self.target_update_freq == 0:
            primary_weights = self.primary_network.get_weights()
            target_weights = self.target_network.get_weights()
            
            for i in range(len(primary_weights)):
                target_weights[i] = self.tau * primary_weights[i] + (1 - self.tau) * target_weights[i]
                
            self.target_network.set_weights(target_weights)
            self.logger.debug("Target network updated")

    def hindsight_experience_replay(self, trajectory):
        """
        Enhanced Hindsight Experience Replay with multiple strategies
        
        Args:
            trajectory: List of experiences (state, action, reward, next_state, vol_label, done)
        """
        # Process only if we have enough experiences
        if len(trajectory) <= 1:
            return
            
        # Process each experience
        for i, (state, action, reward, next_state, vol_label, done) in enumerate(trajectory[:-1]):
            # Original experience already stored during collection
            
            # Apply selected strategies for HER
            if 'final' in self.her_strategies and len(trajectory) > 0:
                # Use final state as goal
                final_state = trajectory[-1][3]  # next_state of last experience
                self._apply_hindsight_goal(state, action, next_state, final_state, vol_label, done)
            
            if 'future' in self.her_strategies and i < len(trajectory) - self.h_bonus:
                # Look ahead h_bonus steps as in original implementation
                future_idx = i + self.h_bonus
                future_state = trajectory[future_idx][3]  # next_state at future_idx
                future_reward = trajectory[future_idx][2]
                
                # Apply bonus with future reward (original logic)
                bonus_reward = reward + self.lambda_bonus * future_reward
                self.remember(state, action, bonus_reward, next_state, vol_label, done)
                
                if i % 10 == 0:  # Log occasionally to avoid spam
                    self.logger.debug(f"Applied bonus: original={reward:.2f}, bonus={bonus_reward:.2f}")
                
                # Also use future state as goal
                self._apply_hindsight_goal(state, action, next_state, future_state, vol_label, done)
            
            if 'random' in self.her_strategies:
                # Sample k random future states to use as goals
                future_indices = [j for j in range(i + 1, len(trajectory))]
                if future_indices:
                    k_samples = min(self.her_k, len(future_indices))
                    sampled_indices = np.random.choice(future_indices, size=k_samples, replace=False)
                    
                    for future_idx in sampled_indices:
                        future_state = trajectory[future_idx][3]
                        self._apply_hindsight_goal(state, action, next_state, future_state, vol_label, done)
    
    def _apply_hindsight_goal(self, state, action, next_state, goal_state, vol_label, done):
        """
        Apply hindsight goal relabeling
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            goal_state: Goal state to use for relabeling
            vol_label: Volatility label
            done: Episode completion flag
        """
        # Extract price-related features (assuming they're in private_state)
        _, _, next_private = next_state
        _, _, goal_private = goal_state
        
        # Calculate how action moves toward goal (simplified example)
        # In real implementation, map this to appropriate features in private_state
        price_idx = 0  # Index of price in private state (adjust based on your state representation)
        
        if len(next_private) > price_idx and len(goal_private) > price_idx:
            next_price = next_private[price_idx]
            goal_price = goal_private[price_idx]
            
            # Different reward calculation based on action direction
            direction = action[0]
            
            if direction == 0:  # Buy
                hindsight_reward = 1.0 if goal_price > next_price else -1.0
            elif direction == 1:  # Sell
                hindsight_reward = 1.0 if goal_price < next_price else -1.0
            else:  # Hold
                # For hold, reward if goal price is close to current price
                price_diff = abs(goal_price - next_price)
                hindsight_reward = 1.0 if price_diff < 0.01 else -0.5  # Smaller penalty for hold
                
            # Scale reward by confidence (action[1] is size/confidence in our model)
            if len(action) > 1:
                confidence = action[1] / (self.branch_sizes[1] - 1)  # Normalize to [0,1]
                hindsight_reward *= (0.5 + 0.5 * confidence)  # Scale by confidence
                
            # Store the hindsight experience
            self.remember(state, action, hindsight_reward, next_state, vol_label, done)
    
    def extract_goal_from_state(self, state):
        """
        Extract goal information from a state for HER
        
        Args:
            state: Tuple of (micro_state, macro_state, private_state)
            
        Returns:
            Goal representation (e.g., target price level)
        """
        # Extract relevant goal information
        # For trading, this could be target price or profit level
        _, _, private_state = state
        
        # Assuming first element of private state is price
        if len(private_state) > 0:
            price = private_state[0]
            return price
        
        return None

    def calculate_goal_reward(self, state, action, next_state, goal):
        """
        Calculate reward based on how action moves toward goal
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            goal: Goal (e.g., target price)
            
        Returns:
            Reward value
        """
        # Extract current and next price
        _, _, private_state = state
        _, _, next_private = next_state
        
        if len(private_state) > 0 and len(next_private) > 0:
            current_price = private_state[0]
            next_price = next_private[0]
            
            # Calculate progress toward goal
            current_dist = abs(current_price - goal)
            next_dist = abs(next_price - goal)
            
            # Reward for moving closer to goal
            progress_reward = current_dist - next_dist
            
            # Direction-based reward
            direction = action[0]
            direction_correct = False
            
            if direction == 0:  # Buy
                direction_correct = goal > current_price
            elif direction == 1:  # Sell
                direction_correct = goal < current_price
                
            direction_reward = 0.5 if direction_correct else -0.5
            
            # Combine rewards
            total_reward = progress_reward + direction_reward
            
            return total_reward
        
        return 0.0

    def get_risk_adjusted_action(self, state, base_action):
        """
        Adjust action (especially size) based on predicted volatility
        
        Args:
            state: Tuple of (micro_state, macro_state, private_state)
            base_action: Original action tuple
            
        Returns:
            Risk-adjusted action tuple
        """
        if not self.predict_volatility:
            return base_action
            
        # Predict volatility for state
        movement_prediction = self.predict_movement(state)
        volatility = movement_prediction['volatility']
        
        # Extract action components
        direction, size = base_action
        
        # Define volatility thresholds
        LOW_VOL = 0.5
        MED_VOL = 1.0
        HIGH_VOL = 2.0
        
        # Adjust position size based on volatility
        if volatility > HIGH_VOL:
            # High volatility - reduce size by 2 levels or more
            adjusted_size = max(0, size - 2)
        elif volatility > MED_VOL:
            # Medium volatility - reduce size by 1 level
            adjusted_size = max(0, size - 1)
        elif volatility < LOW_VOL:
            # Low volatility - potentially increase size
            adjusted_size = min(self.branch_sizes[1] - 1, size + 1)
        else:
            # Normal volatility - keep size
            adjusted_size = size
        
        return (direction, adjusted_size)
        
    def predict_movement(self, state):
        """
        Predict expected price movement and volatility from state
        
        Args:
            state: Tuple of (micro_state, macro_state, private_state)
            
        Returns:
            Dictionary with movement prediction metrics:
            - expected_movement: Expected price direction/magnitude
            - volatility: Predicted market volatility
            - confidence: Model confidence in prediction
            - risk_adjusted_movement: Volatility-adjusted movement
        """
        micro_state, macro_state, private_state = state
        
        # Add batch dimension if needed
        if micro_state.ndim == 2:
            micro_state = np.expand_dims(micro_state, axis=0)
        if macro_state.ndim == 2:
            macro_state = np.expand_dims(macro_state, axis=0)
        if private_state.ndim == 1:
            private_state = np.expand_dims(private_state, axis=0)
        
        try:
            # Get Q-values and possibly volatility
            outputs = self.primary_network.predict([micro_state, macro_state, private_state])
            
            # Extract Q-values
            if self.predict_volatility:
                q_branches = outputs[:-1]
                volatility = float(outputs[-1][0][0])  # Extract scalar
            else:
                q_branches = outputs
                volatility = 1.0  # Default volatility when not predicted
            
            # Assuming branch 0 is direction (BUY/SELL/HOLD)
            direction_q = q_branches[0][0]  # First batch item
            
            if len(direction_q) >= 3:
                buy_q = float(direction_q[0])  # BUY
                sell_q = float(direction_q[1])  # SELL
                hold_q = float(direction_q[2])  # HOLD
                
                # Difference between buy and sell indicates expected movement
                expected_movement = buy_q - sell_q
                
                # Max Q-value indicates confidence
                max_q = max(buy_q, sell_q, hold_q)
                min_q = min(buy_q, sell_q, hold_q)
                range_q = max_q - min_q + 1e-6  # Avoid division by zero
                confidence = (max_q - hold_q) / range_q
                
                # Risk-adjusted movement: scale movement by inverse of volatility
                # When volatility is high, we reduce our movement expectation
                risk_adjusted_movement = expected_movement / max(1.0, volatility)
            else:
                expected_movement = 0.0
                confidence = 0.0
                risk_adjusted_movement = 0.0
            
            return {
                'expected_movement': expected_movement,
                'volatility': volatility,
                'confidence': confidence,
                'risk_adjusted_movement': risk_adjusted_movement
            }
            
        except Exception as e:
            self.logger.error(f"Error in predict_movement: {str(e)}")
            return {
                'expected_movement': 0.0,
                'volatility': 1.0,
                'confidence': 0.0,
                'risk_adjusted_movement': 0.0
            }

    def save(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save weights
        weights_path = f"{filepath}_weights.h5"
        self.primary_network.save_weights(weights_path)
        
        # Save replay buffer data to allow learning to continue after restart
        replay_buffer_path = f"{filepath}_replay_buffer.npz"
        if len(self.memory) > 0:
            # We can't directly save complex objects like tuples with states
            # So we'll save the priorities and the index, memory data will be rebuilt
            try:
                np.savez(
                    replay_buffer_path,
                    priorities=self.priorities[:len(self.memory)],
                    memory_index=np.array([self.memory_index]),
                    buffer_size=np.array([len(self.memory)])
                )
                self.logger.info(f"Replay buffer state saved to {replay_buffer_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save replay buffer: {str(e)}")
        
        # Save metadata
        metadata = {
            'micro_dim': self.micro_dim,
            'macro_dim': self.macro_dim,
            'private_dim': self.private_dim,
            'micro_seq_len': self.micro_seq_len,
            'macro_seq_len': self.macro_seq_len,
            'action_branches': self.action_branches,
            'branch_sizes': self.branch_sizes,
            'predict_volatility': self.predict_volatility,
            'volatility_weight': self.volatility_weight,
            'h_bonus': self.h_bonus,
            'lambda_bonus': self.lambda_bonus,
            'her_strategies': self.her_strategies,
            'her_k': self.her_k,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'epsilon_per': self.epsilon_per
        }
        
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Model saved to {filepath}")
        return filepath

    def load(self, filepath):
        """Load model from file"""
        # Load weights
        weights_path = f"{filepath}_weights.h5"
        self.primary_network.load_weights(weights_path)
        self.target_network.load_weights(weights_path)
        
        # Load metadata
        with open(f"{filepath}_config.json", 'r') as f:
            metadata = json.load(f)
        
        # Apply configuration
        self.micro_dim = metadata['micro_dim']
        self.macro_dim = metadata['macro_dim']
        self.private_dim = metadata['private_dim']
        self.micro_seq_len = metadata['micro_seq_len']
        self.macro_seq_len = metadata['macro_seq_len']
        self.action_branches = metadata['action_branches']
        self.branch_sizes = metadata['branch_sizes']
        self.predict_volatility = metadata['predict_volatility']
        self.volatility_weight = metadata['volatility_weight']
        self.h_bonus = metadata['h_bonus']
        self.lambda_bonus = metadata['lambda_bonus']
        self.gamma = metadata['gamma']
        self.learning_rate = metadata['learning_rate']
        self.epsilon = metadata['epsilon']
        self.train_step = metadata['train_step']
        
        # Load newer parameters if available
        if 'her_strategies' in metadata:
            self.her_strategies = metadata['her_strategies']
        if 'her_k' in metadata:
            self.her_k = metadata['her_k']
        if 'alpha' in metadata:
            self.alpha = metadata['alpha']
        if 'beta' in metadata:
            self.beta = metadata['beta']
        if 'beta_increment' in metadata:
            self.beta_increment = metadata['beta_increment']
        if 'epsilon_per' in metadata:
            self.epsilon_per = metadata['epsilon_per']
            
        # Try to load replay buffer state if it exists
        replay_buffer_path = f"{filepath}_replay_buffer.npz"
        try:
            if os.path.exists(replay_buffer_path):
                buffer_data = np.load(replay_buffer_path)
                self.priorities[:len(buffer_data['priorities'])] = buffer_data['priorities']
                self.memory_index = int(buffer_data['memory_index'][0])
                buffer_size = int(buffer_data['buffer_size'][0])
                
                # Note: The actual memory data needs to be rebuilt through training
                # We can't save and load the actual state-action tuples easily
                # But we preserved the priorities so new experiences will be properly prioritized
                
                self.logger.info(f"Loaded replay buffer state with {buffer_size} priorities from {replay_buffer_path}")
                self.logger.info(f"Actual experiences need to be recollected, but priorities are preserved")
        except Exception as e:
            self.logger.warning(f"Failed to load replay buffer: {str(e)}")
        
        # Setup autosave timer to periodically save model during training
        self._setup_autosave(filepath)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def _setup_autosave(self, filepath, save_interval_minutes=10):
        """
        Setup a timer to automatically save the model at regular intervals
        
        Args:
            filepath: Base filepath to save to
            save_interval_minutes: How often to save (in minutes)
        """
        # Import threading here to avoid issues if it's not available
        try:
            import threading
            
            # Define autosave function
            def autosave():
                while True:
                    # Sleep for the specified interval
                    time.sleep(save_interval_minutes * 60)
                    
                    # Save model with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_path = f"{filepath}_autosave_{timestamp}"
                    try:
                        self.save(save_path)
                        self.logger.info(f"Auto-saved model to {save_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to auto-save model: {str(e)}")
            
            # Start autosave thread
            try:
                autosave_thread = threading.Thread(target=autosave, daemon=True)
                autosave_thread.start()
                self.logger.info(f"Autosave timer started (every {save_interval_minutes} minutes)")
            except Exception as e:
                self.logger.warning(f"Failed to start autosave timer: {str(e)}")
                
        except ImportError:
            self.logger.warning("Threading module not available, autosave disabled")

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train model in supervised mode (for compatibility with model_trainer)
        
        Args:
            X_train: Training features (should be tuple of micro, macro, private arrays)
            y_train: Target values (price movements or action labels)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history object
        """
        self.logger.info("Converting supervised data to reinforcement learning format")
        
        # Check if X_train is already in the right format
        if not isinstance(X_train, tuple) or len(X_train) != 3:
            self.logger.warning("X_train should be a tuple of (micro, macro, private) arrays. Trying to convert...")
            
            # Try to convert (assuming X_train is a sequence of features)
            try:
                # Simple conversion: split features into micro, macro, private
                # Adjust these indices based on your actual feature layout
                micro_idx = min(self.micro_dim, X_train.shape[1])
                macro_idx = micro_idx + min(self.macro_dim, X_train.shape[1] - micro_idx)
                
                micro_features = X_train[:, :micro_idx]
                macro_features = X_train[:, micro_idx:macro_idx]
                private_features = X_train[:, macro_idx:]
                
                # Reshape for sequence inputs
                micro_seq = np.zeros((len(X_train), self.micro_seq_len, self.micro_dim))
                macro_seq = np.zeros((len(X_train), self.macro_seq_len, self.macro_dim))
                
                # Fill last timestep with current data
                for i in range(len(X_train)):
                    micro_seq[i, -1, :micro_features.shape[1]] = micro_features[i]
                    macro_seq[i, -1, :macro_features.shape[1]] = macro_features[i]
                
                X_train = (micro_seq, macro_seq, private_features)
                self.logger.info("Successfully converted X_train to required format")
                
            except Exception as e:
                self.logger.error(f"Error converting X_train: {str(e)}")
                return None
        
        # Create synthetic RL experiences from supervised data
        for i in range(len(X_train[0])):  # Use first array length
            micro_state = X_train[0][i]
            macro_state = X_train[1][i]
            private_state = X_train[2][i]
            
            # Use same state for next_state (supervised learning doesn't have next states)
            next_micro = micro_state
            next_macro = macro_state
            next_private = private_state
            
            state = (micro_state, macro_state, private_state)
            next_state = (next_micro, next_macro, next_private)
            
            # Determine action based on y_train direction
            target = y_train[i]
            
            if isinstance(target, (int, float)):
                # If target is a single value (price movement)
                if target > 0.1:  # Strong positive
                    action = (0, 4)  # BUY, high confidence
                elif target > 0:   # Weak positive
                    action = (0, 2)  # BUY, medium confidence
                elif target < -0.1:  # Strong negative
                    action = (1, 4)  # SELL, high confidence
                elif target < 0:   # Weak negative
                    action = (1, 2)  # SELL, medium confidence
                else:
                    action = (2, 0)  # HOLD
            else:
                # If target is already an action tuple
                action = target
            
            # Use target value as reward
            if isinstance(target, (int, float)):
                reward = target
            else:
                # If target is an action, use a default reward
                reward = 0.1
            
            # Calculate volatility label (if available in y_train)
            vol_label = None
            if isinstance(y_train, tuple) and len(y_train) > 1:
                vol_label = y_train[1][i]
            
            # Add to memory
            self.remember(state, action, reward, next_state, vol_label, False)
        
        # Train using replay
        losses = []
        for epoch in range(epochs):
            # Multiple updates per epoch
            for _ in range(max(1, len(X_train[0]) // batch_size)):
                loss = self.replay(batch_size)
                if loss:
                    losses.append(np.mean(loss))
            
            # Log progress
            if epoch % 5 == 0:
                self.logger.info(f"Training Epoch {epoch}/{epochs}, Avg Loss: {np.mean(losses[-batch_size:]) if losses else 'N/A':.4f}")
        
        # Create mock history object
        history = type('obj', (object,), {
            'history': {
                'loss': losses,
                'val_loss': []
            }
        })
        
        return history