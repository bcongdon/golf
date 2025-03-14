import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from typing import List, Tuple, Dict, Optional

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO algorithm.
    
    Input: Game state representation with card indices
    Output: 
        - Actor: Action probabilities
        - Critic: Value function
    """
    
    def __init__(self, input_size: int = 29, hidden_size: int = 512, action_size: int = 9, 
                 embedding_dim: int = 8, num_card_ranks: int = 13):
        super(ActorCriticNetwork, self).__init__()
        
        # Determine if CUDA is available for optimizations
        self.use_cuda = torch.cuda.is_available()
        
        # Card embedding layer (13 possible ranks -> embedding_dim)
        # Add +1 for unknown/masked cards and +1 for None (no card)
        self.card_embedding = nn.Embedding(num_card_ranks + 2, embedding_dim)  
        
        # Calculate the size after embeddings
        # Input structure expected: First part is card indices, second part is other features
        # Derive the number of card positions and non-card features from input_size
        self.num_card_positions = 14  # 6 player cards + 6 opponent cards + discard + drawn
        self.non_card_features = input_size - self.num_card_positions
        
        # Validate the input size matches our expectations
        assert input_size >= self.num_card_positions, f"Input size ({input_size}) must be at least {self.num_card_positions}"
        
        embedded_size = self.num_card_positions * embedding_dim
        total_features = embedded_size + self.non_card_features
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            # Input layer
            nn.Linear(total_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),
            
            # First hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights using Kaiming initialization
        self._initialize_weights()
        
        # Optimize memory usage for CUDA if available
        if self.use_cuda:
            # Use channels_last memory format for better performance on CUDA
            self = self.to(memory_format=torch.channels_last)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        # Convert input to channels_last format if using CUDA for better performance
        if self.use_cuda and x.dim() >= 3:
            x = x.to(memory_format=torch.channels_last)
            
        # Extract card indices and non-card features
        card_indices = x[:, :self.num_card_positions].long()  # First 14 elements are card indices
        non_card_features = x[:, self.num_card_positions:]    # Remaining elements are binary features
        
        # Apply embedding to card indices
        embedded_cards = self.card_embedding(card_indices)
        
        # Flatten the embeddings
        batch_size = embedded_cards.size(0)
        embedded_cards = embedded_cards.view(batch_size, -1)
        
        # Concatenate with non-card features
        combined_features = torch.cat([embedded_cards, non_card_features], dim=1)
        
        # Extract features
        features = self.feature_extractor(combined_features)
        
        # Get action probabilities and state value
        action_logits = self.actor(features)
        state_value = self.critic(features)
        
        return action_logits, state_value


class PPOMemory:
    """
    Memory buffer for PPO algorithm using NumPy arrays for efficiency.
    
    Args:
        batch_size: Size of batches for training
        capacity: Maximum number of transitions to store
        state_dim: Dimension of state space (must match environment state size)
    """
    
    def __init__(self, batch_size: int = 64, capacity: int = 10000, state_dim: int = 29):
        self.batch_size = batch_size
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = 1
        self.prob_dim = 1
        self.val_dim = 1
        self.reward_dim = 1
        self.done_dim = 1
        self.mask_dim = 9  # Assuming 9 possible actions
        
        # Pre-allocate memory with NumPy arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int32)
        self.probs = np.zeros((capacity, 1), dtype=np.float32)
        self.vals = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.valid_actions_masks = np.zeros((capacity, self.mask_dim), dtype=np.bool_)
        
        # Position counter
        self.pos = 0
        self.full = False
    
    def store(self, state, action, prob, val, reward, done, valid_actions_mask):
        """Store experience in memory efficiently."""
        # Convert inputs to appropriate shapes if needed
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        if np.isscalar(action):
            action = np.array([action], dtype=np.int32)
        if np.isscalar(prob):
            prob = np.array([prob], dtype=np.float32)
        if np.isscalar(val):
            val = np.array([val], dtype=np.float32)
        if np.isscalar(reward):
            reward = np.array([reward], dtype=np.float32)
        if np.isscalar(done):
            done = np.array([done], dtype=np.float32)
        
        # Store data in pre-allocated arrays
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.probs[self.pos] = prob
        self.vals[self.pos] = val
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.valid_actions_masks[self.pos] = valid_actions_mask
        
        # Update position counter
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True
    
    def clear(self):
        """Reset memory buffer."""
        self.pos = 0
        self.full = False
    
    def generate_batches(self):
        """Generate batches for training."""
        # Determine actual size of used memory
        size = self.capacity if self.full else self.pos
        
        # Generate random indices
        indices = np.arange(size, dtype=np.int64)
        np.random.shuffle(indices)
        
        # Create batches
        batch_start = np.arange(0, size, self.batch_size)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches
    
    def get_all_data(self):
        """Get all stored data as NumPy arrays."""
        size = self.capacity if self.full else self.pos
        
        return (
            self.states[:size],
            self.actions[:size].squeeze(),
            self.probs[:size].squeeze(),
            self.vals[:size].squeeze(),
            self.rewards[:size].squeeze(),
            self.dones[:size].squeeze(),
            self.valid_actions_masks[:size]
        )
    
    def __len__(self):
        """Return the current size of the memory."""
        return self.capacity if self.full else self.pos


class PPOAgent:
    """
    Proximal Policy Optimization agent for playing Golf.
    """
    
    def __init__(
        self,
        state_size: int = 29,  # Size: 14 card indices + 15 binary features (including turn progress)
        action_size: int = 9,  # Number of possible actions in the game
        hidden_size: int = 512,  # Size of hidden layers in neural network
        embedding_dim: int = 8,  # Dimension of card embeddings
        num_card_ranks: int = 13,  # Number of possible card ranks (A-K)
        actor_lr: float = 0.0003,  # Learning rate for actor
        critic_lr: float = 0.0003,  # Learning rate for critic
        gamma: float = 0.99,  # Discount factor
        gae_lambda: float = 0.95,  # GAE lambda parameter
        policy_clip: float = 0.2,  # PPO clip parameter
        batch_size: int = 64,  # Batch size for training
        n_epochs: int = 10,  # Number of epochs to train on each update
        entropy_coef: float = 0.01,  # Entropy coefficient for exploration
        value_coef: float = 0.5,  # Value loss coefficient
        max_grad_norm: float = 0.5,  # Gradient clipping
        update_interval: int = 2048  # Steps between updates
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_card_ranks = num_card_ranks
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_interval = update_interval
        self.batch_size = batch_size
        
        # Check for available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) device for acceleration")
            # MPS doesn't support mixed precision training yet
            self.use_amp = False
            # Set smaller batch sizes for MPS to prevent memory issues
            self.batch_size = min(batch_size, 32)
            # Optimize memory usage for MPS
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device for acceleration")
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            # Set tensor cores precision if available
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            # For mixed precision training when CUDA is available
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
            self.use_amp = False
        
        # Initialize actor-critic network
        self.actor_critic = ActorCriticNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            action_size=action_size,
            embedding_dim=embedding_dim,
            num_card_ranks=num_card_ranks
        ).to(self.device)
        
        # Option 1: Use a single optimizer for all parameters
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=actor_lr)
        
        # Initialize memory with proper parameters
        memory_capacity = update_interval * 2  # Set capacity to at least 2x update_interval
        self.memory = PPOMemory(batch_size=batch_size, capacity=memory_capacity, state_dim=state_size)
        
        # Training metrics
        self.learn_step_counter = 0
        self.total_steps = 0
    
    @staticmethod
    def random_action(valid_actions: List[int]) -> int:
        """
        Common implementation of random policy for both exploration and evaluation.
        Uses a 50/50 split between drawing a new card and using the discard pile in the draw phase.
        
        Args:
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        # Check if we're in the draw phase (actions 0 and 1 are valid)
        if 0 in valid_actions and 1 in valid_actions:
            # Draw phase: 50/50 split between drawing from deck and discard pile
            if random.random() < 0.5:
                return 0  # Draw from deck
            else:
                return 1  # Draw from discard
        else:
            # Not in draw phase, choose randomly from valid actions
            return random.choice(valid_actions)
    
    def _create_valid_actions_mask(self, valid_actions: List[int]) -> torch.Tensor:
        """Create a mask for valid actions."""
        mask = torch.zeros(self.action_size, dtype=torch.bool)
        mask[valid_actions] = True
        return mask
    
    def select_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action using the current policy.
        
        Args:
            state: Current state observation
            valid_actions: List of valid actions
            training: Whether the agent is in training mode
            
        Returns:
            Selected action, action probability (raw, not log), state value
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Create valid actions mask
        valid_actions_mask = self._create_valid_actions_mask(valid_actions).to(self.device)
        
        # Set network to evaluation mode for inference
        self.actor_critic.eval()
        
        with torch.no_grad():
            # Get action logits and state value
            action_logits, state_value = self.actor_critic(state_tensor)
            
            # Apply mask to logits (set invalid actions to large negative value)
            masked_logits = action_logits.clone()
            # Vectorized masking
            invalid_mask = ~valid_actions_mask.unsqueeze(0)
            masked_logits.masked_fill_(invalid_mask, -1e10)
            
            # Get action probabilities
            action_probs = F.softmax(masked_logits, dim=1)
            
            # Sample action from distribution
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
            # Get raw probability of selected action
            action_prob = action_probs[0, action].item()
            
            # Get state value
            value = state_value.item()
        
        # Set back to training mode if we're in training
        if training:
            self.actor_critic.train()
        
        return action, action_prob, value
    
    def store_transition(self, state, action, prob, val, reward, done, valid_actions):
        """Store transition in memory."""
        # Create valid actions mask
        valid_actions_mask = self._create_valid_actions_mask(valid_actions).cpu().numpy()
        
        # Store in memory
        self.memory.store(state, action, prob, val, reward, done, valid_actions_mask)
        self.total_steps += 1
    
    def _compute_advantages(self, rewards, values, dones):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        # Convert to numpy arrays if they're not already
        if isinstance(rewards, list):
            rewards = np.array(rewards)
        if isinstance(values, list):
            values = np.array(values)
        if isinstance(dones, list):
            dones = np.array(dones)
            
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = values[-1]
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            advantages[t] = last_advantage
            last_value = values[t]
            
        return advantages
    
    def _manage_mps_memory(self):
        """Monitor and manage memory usage on MPS devices."""
        if self.device.type == 'mps':
            # Clear MPS cache to free up memory
            torch.mps.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Sleep briefly to allow memory to be freed
            import time
            time.sleep(0.01)
    
    def learn(self):
        """Update policy and value networks using PPO."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Ensure network is in training mode
        self.actor_critic.train()
        
        # Manage memory for MPS devices
        self._manage_mps_memory()
            
        # Get data from memory using the new get_all_data method
        states, actions, old_probs, values, rewards, dones, valid_actions_masks = self.memory.get_all_data()
        
        # Compute advantages and returns
        advantages = self._compute_advantages(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Generate batches
        batches = self.memory.generate_batches()
        
        # Training loop
        total_loss_value = 0.0
        num_batches = 0
        for _ in range(self.n_epochs):
            for batch in batches:
                # Get batch data
                batch_states = torch.FloatTensor(states[batch]).to(self.device)
                batch_actions = torch.LongTensor(actions[batch]).to(self.device)
                batch_old_probs = torch.FloatTensor(old_probs[batch]).to(self.device)
                batch_advantages = torch.FloatTensor(advantages[batch]).to(self.device)
                batch_returns = torch.FloatTensor(returns[batch]).to(self.device)
                batch_valid_masks = torch.BoolTensor(valid_actions_masks[batch]).to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # Get new action logits and state values
                        action_logits, critic_value = self.actor_critic(batch_states)
                        critic_value = critic_value.squeeze()
                        
                        # Apply valid actions mask
                        masked_logits = action_logits.clone()
                        # Vectorized masking operation instead of loop
                        invalid_mask = ~batch_valid_masks
                        masked_logits.masked_fill_(invalid_mask, -1e10)
                        
                        # Get action probabilities
                        action_probs = F.softmax(masked_logits, dim=1)
                        dist = Categorical(action_probs)
                        
                        # Get entropy for exploration bonus
                        entropy = dist.entropy().mean()
                        
                        # Get new action log probabilities
                        new_log_probs = dist.log_prob(batch_actions)
                        
                        # Make sure old_probs are not already in log form
                        # PPO memory stores raw probabilities, so we need to take the log here
                        old_log_probs = torch.log(batch_old_probs + 1e-10)
                        
                        # Calculate probability ratio: exp(new_log_probs - old_log_probs)
                        prob_ratio = torch.exp(new_log_probs - old_log_probs)
                        
                        # PPO update
                        weighted_probs = batch_advantages * prob_ratio
                        weighted_clipped_probs = batch_advantages * torch.clamp(
                            prob_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip
                        )
                        
                        # Calculate losses
                        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                        critic_loss = F.mse_loss(critic_value, batch_returns)
                        
                        # Combined loss with entropy bonus
                        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                else:
                    # Standard training without autocast
                    # Get new action logits and state values
                    action_logits, critic_value = self.actor_critic(batch_states)
                    critic_value = critic_value.squeeze()
                    
                    # Apply valid actions mask
                    masked_logits = action_logits.clone()
                    # Vectorized masking operation instead of loop
                    invalid_mask = ~batch_valid_masks
                    masked_logits.masked_fill_(invalid_mask, -1e10)
                    
                    # Get action probabilities
                    action_probs = F.softmax(masked_logits, dim=1)
                    dist = Categorical(action_probs)
                    
                    # Get entropy for exploration bonus
                    entropy = dist.entropy().mean()
                    
                    # Get new action log probabilities
                    new_log_probs = dist.log_prob(batch_actions)
                    
                    # Make sure old_probs are not already in log form
                    # PPO memory stores raw probabilities, so we need to take the log here
                    old_log_probs = torch.log(batch_old_probs + 1e-10)
                    
                    # Calculate probability ratio: exp(new_log_probs - old_log_probs)
                    prob_ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # PPO update
                    weighted_probs = batch_advantages * prob_ratio
                    weighted_clipped_probs = batch_advantages * torch.clamp(
                        prob_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip
                    )
                    
                    # Calculate losses
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                    critic_loss = F.mse_loss(critic_value, batch_returns)
                    
                    # Combined loss with entropy bonus
                    total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Track average loss
                total_loss_value += total_loss.item()
                num_batches += 1
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    # Mixed precision training
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Manage memory for MPS devices after each batch
                if self.device.type == 'mps':
                    self._manage_mps_memory()
        
        # Clear memory after update
        self.memory.clear()
        self.learn_step_counter += 1
        
        # Return average loss if any batches were processed
        return total_loss_value / num_batches if num_batches > 0 else None
    
    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learn_step_counter': self.learn_step_counter,
            'total_steps': self.total_steps
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learn_step_counter = checkpoint['learn_step_counter']
            self.total_steps = checkpoint['total_steps']
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}") 