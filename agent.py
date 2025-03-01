import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple, Dict

class GolfMLP(nn.Module):
    """
    Enhanced multi-layer perceptron for the Golf card game.
    
    Input: Game state representation
    Output: Q-values for each possible action
    """
    
    def __init__(self, input_size: int = 56, hidden_size: int = 512, output_size: int = 9):
        super(GolfMLP, self).__init__()
        
        # Determine if CUDA is available for optimizations
        self.use_cuda = torch.cuda.is_available()
        
        # Deeper network with larger hidden size and different activation functions
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),  # LeakyReLU instead of ReLU for better gradient flow
            nn.Dropout(0.2),
            
            # First hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Second hidden layer (new)
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Third hidden layer (new)
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            
            # Fourth hidden layer (new)
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            
            # Output layer
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights using Kaiming initialization (better for LeakyReLU)
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
        return self.network(x)


class PrioritizedReplayBuffer:
    """
    Optimized Prioritized experience replay buffer with episode marking for winning trajectories.
    Uses vectorized operations for better performance.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.buffer = []  # List of experiences
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # Current position in buffer
        self.size = 0  # Current size of buffer
        
        # PER hyperparameters
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta = beta  # Importance sampling weight (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment  # Increment beta over time to reduce bias
        self.max_priority = 1.0  # Initial max priority
        
        # Episode tracking
        self.current_episode = []  # Temporarily stores the current episode
        self.win_multiplier = 2.0  # Priority multiplier for winning episodes
        
        # Pre-allocate arrays for better performance
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._dones = None
        self._initialized = False
        
        # Use a more efficient data structure for sampling
        self._segment_tree_initialized = False
        self._sum_tree = None
        self._min_tree = None
        
        # Cache for faster sampling
        self._cached_indices = None
        self._cached_weights = None
        self._last_beta = None
    
    def _initialize_arrays(self, state_shape):
        """Initialize arrays based on first experience."""
        if not self._initialized:
            # Use float16 for states to reduce memory usage (if using CUDA)
            if torch.cuda.is_available():
                self._states = np.zeros((self.capacity,) + state_shape, dtype=np.float16)
                self._next_states = np.zeros((self.capacity,) + state_shape, dtype=np.float16)
            else:
                # For CPU or MPS, use float32
                self._states = np.zeros((self.capacity,) + state_shape, dtype=np.float32)
                self._next_states = np.zeros((self.capacity,) + state_shape, dtype=np.float32)
                
            self._actions = np.zeros(self.capacity, dtype=np.int32)  # int32 is enough for actions
            self._rewards = np.zeros(self.capacity, dtype=np.float32)
            self._dones = np.zeros(self.capacity, dtype=np.bool_)
            self._initialized = True
            
            # Initialize segment trees for efficient sampling
            self._initialize_segment_trees()
    
    def _initialize_segment_trees(self):
        """Initialize segment trees for efficient sampling."""
        if not self._segment_tree_initialized:
            # Create sum and min trees for efficient sampling
            self._sum_tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)
            self._min_tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)
            self._segment_tree_initialized = True
    
    def _update_segment_trees(self, idx, priority):
        """Update segment trees with new priority."""
        if not self._segment_tree_initialized:
            return
            
        # Convert to tree index
        tree_idx = idx + self.capacity - 1
        
        # Update sum tree
        change = priority - self._sum_tree[tree_idx]
        self._sum_tree[tree_idx] = priority
        
        # Propagate changes up the tree
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._sum_tree[tree_idx] = self._sum_tree[2 * tree_idx + 1] + self._sum_tree[2 * tree_idx + 2]
        
        # Update min tree
        tree_idx = idx + self.capacity - 1
        self._min_tree[tree_idx] = priority
        
        # Propagate changes up the tree
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._min_tree[tree_idx] = min(self._min_tree[2 * tree_idx + 1], self._min_tree[2 * tree_idx + 2])
    
    def _get_priority(self, idx):
        """Get priority from segment tree."""
        if not self._segment_tree_initialized:
            return self.priorities[idx]
            
        return self._sum_tree[idx + self.capacity - 1]
    
    def _sum(self):
        """Get sum of all priorities."""
        if not self._segment_tree_initialized:
            return np.sum(self.priorities[:self.size])
            
        return self._sum_tree[0]
    
    def _min(self):
        """Get minimum priority."""
        if not self._segment_tree_initialized:
            return np.min(self.priorities[:self.size]) if self.size > 0 else 1.0
            
        return self._min_tree[0]
    
    def add(self, state, action, reward, next_state, done, win=False):
        """Add a new experience to the buffer."""
        # Store experience in the current episode
        self.current_episode.append((state, action, reward, next_state, done))
        
        # If episode is done, process the entire episode
        if done:
            # Calculate priority multiplier based on win status
            priority_multiplier = self.win_multiplier if win else 1.0
            
            # Add all transitions from the episode to the buffer with appropriate priority
            for exp in self.current_episode:
                state, action, reward, next_state, done = exp
                
                # Initialize arrays if needed
                if not self._initialized and state is not None:
                    self._initialize_arrays(np.array(state).shape)
                
                # Create an experience with max priority
                priority = self.max_priority * priority_multiplier
                
                # Add to buffer
                if self.size < self.capacity:
                    if self._initialized:
                        # Convert to float16 if using CUDA to save memory
                        if torch.cuda.is_available():
                            self._states[self.size] = state.astype(np.float16)
                            self._next_states[self.size] = next_state.astype(np.float16)
                        else:
                            # For CPU or MPS, use float32
                            self._states[self.size] = state
                            self._next_states[self.size] = next_state
                            
                        self._actions[self.size] = action
                        self._rewards[self.size] = reward
                        self._dones[self.size] = done
                    else:
                        self.buffer.append(exp)
                        
                    self.priorities[self.size] = priority
                    self._update_segment_trees(self.size, priority ** self.alpha)
                    self.size += 1
                else:
                    if self._initialized:
                        # Convert to float16 if using CUDA to save memory
                        if torch.cuda.is_available():
                            self._states[self.position] = state.astype(np.float16)
                            self._next_states[self.position] = next_state.astype(np.float16)
                        else:
                            # For CPU or MPS, use float32
                            self._states[self.position] = state
                            self._next_states[self.position] = next_state
                            
                        self._actions[self.position] = action
                        self._rewards[self.position] = reward
                        self._dones[self.position] = done
                    else:
                        self.buffer[self.position] = exp
                        
                    self.priorities[self.position] = priority
                    self._update_segment_trees(self.position, priority ** self.alpha)
                    self.position = (self.position + 1) % self.capacity
            
            # Clear the current episode
            self.current_episode = []
    
    def _sample_proportional(self, batch_size):
        """Sample indices based on proportional prioritization."""
        if not self._segment_tree_initialized:
            # Fallback to numpy sampling
            priorities = self.priorities[:self.size]
            probs = priorities ** self.alpha
            probs /= np.sum(probs)
            return np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        indices = np.zeros(batch_size, dtype=np.int32)
        
        # Get sum of priorities
        total_priority = self._sum()
        
        # Divide range into batch_size segments
        segment_size = total_priority / batch_size
        
        for i in range(batch_size):
            # Sample uniformly from segment
            mass = np.random.uniform(segment_size * i, segment_size * (i + 1))
            
            # Search for index in sum tree
            idx = 0
            while idx < self.capacity - 1:
                left = 2 * idx + 1
                if mass <= self._sum_tree[left]:
                    idx = left
                else:
                    mass -= self._sum_tree[left]
                    idx = 2 * idx + 2
            
            # Convert tree index to buffer index
            indices[i] = idx - (self.capacity - 1)
        
        return indices
    
    def sample(self, batch_size: int) -> Tuple[List, List[int], np.ndarray]:
        """Sample a batch of experiences based on priorities using vectorized operations."""
        if self.size == 0:
            return [], [], np.array([])
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Check if we can reuse cached indices and weights
        if self._cached_indices is not None and len(self._cached_indices) == batch_size and self._last_beta == self.beta:
            indices = self._cached_indices
            weights = self._cached_weights
        else:
            # Sample indices based on priorities
            indices = self._sample_proportional(min(batch_size, self.size))
            
            # Calculate importance sampling weights
            weights = self._calculate_weights(indices)
            
            # Cache for next time
            self._cached_indices = indices
            self._cached_weights = weights
            self._last_beta = self.beta
        
        # Get sampled experiences
        if self._initialized:
            # Use pre-allocated arrays for faster access
            states = self._states[indices]
            actions = self._actions[indices]
            rewards = self._rewards[indices]
            next_states = self._next_states[indices]
            dones = self._dones[indices]
            
            # Convert float16 back to float32 if needed
            if torch.cuda.is_available() and states.dtype == np.float16:
                states = states.astype(np.float32)
                next_states = next_states.astype(np.float32)
                
            experiences = list(zip(states, actions, rewards, next_states, dones))
        else:
            experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def _calculate_weights(self, indices):
        """Calculate importance sampling weights."""
        # Get priorities for the sampled indices
        if self._segment_tree_initialized:
            priorities = np.array([self._get_priority(idx) for idx in indices])
        else:
            priorities = self.priorities[indices]
        
        # Convert priorities to probabilities
        probs = priorities / self._sum()
        
        # Calculate importance sampling weights
        weights = (self.size * probs) ** (-self.beta)
        
        # Normalize weights
        weights /= weights.max()
        
        return weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors using vectorized operations."""
        # Add small epsilon to ensure non-zero priority (vectorized)
        new_priorities = np.abs(td_errors) + 1e-5
        
        # Update priorities and segment trees
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority
            self._update_segment_trees(idx, priority ** self.alpha)
        
        # Update max priority
        self.max_priority = max(self.max_priority, new_priorities.max())
        
        # Invalidate cache
        self._cached_indices = None
        self._cached_weights = None
    
    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """
    Deep Q-Network agent for playing Golf.
    """
    
    def __init__(
        self,
        state_size: int = 56,  # Size of the observation space (card representation)
        action_size: int = 9,  # Number of possible actions in the game
        hidden_size: int = 512,  # Size of hidden layers in neural network
        learning_rate: float = 0.0001,  # Learning rate for optimizer
        gamma: float = 0.99,  # Discount factor for future rewards
        epsilon_start: float = 1.0,  # Initial exploration rate
        epsilon_end: float = 0.01,  # Minimum exploration rate
        epsilon_decay: float = 0.9995,  # Rate at which exploration decreases
        epsilon_warmup_episodes: int = 1000,  # Number of episodes to keep epsilon at start value
        buffer_size: int = 100000,  # Size of replay memory
        batch_size: int = 256,  # Number of samples per training batch
        target_update: int = 250,  # Frequency of target network updates
        per_alpha: float = 0.7,  # Prioritization exponent for replay buffer
        per_beta: float = 0.5,  # Initial importance sampling correction
        per_beta_increment: float = 0.0001  # Increment for beta parameter over time
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_warmup_episodes = epsilon_warmup_episodes
        self.current_episode = 0  # Track current episode for epsilon warmup
        
        # Replay buffer
        self.buffer_size = buffer_size
        self.batch_size = max(2, batch_size)  # Ensure batch size is at least 2
        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            beta=per_beta,
            beta_increment=per_beta_increment
        )
        
        # Episode tracking for win marking
        self.current_episode_rewards = 0
        
        # Networks
        # Check for MPS (Metal Performance Shaders) support on macOS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            # Set tensor cores precision if available (Ampere GPUs and newer)
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                # Use TF32 precision (faster than FP32, almost as accurate)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
            
        self.q_network = GolfMLP(state_size, hidden_size, action_size).to(self.device)
        self.target_network = GolfMLP(state_size, hidden_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is used for prediction only
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Target network update frequency
        self.target_update = target_update
        self.update_counter = 0
        
        # For mixed precision training when CUDA is available
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def select_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """
        Select an action using an improved exploration strategy.
        
        Args:
            state: Current state observation
            valid_actions: List of valid actions
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        # Get Q-values for all actions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set network to evaluation mode for inference
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Set back to training mode if we're in training
        if training:
            self.q_network.train()
        
        # Filter for valid actions only
        valid_q = {action: q_values[action] for action in valid_actions}
        
        if training:
            # Exploration strategies
            if random.random() < self.epsilon:
                # Different exploration strategies based on game phase
                if 0 in valid_actions and 1 in valid_actions:  # Drawing phase
                    # Encourage trying both draw from deck and discard pile
                    # Bias slightly toward taking from discard pile when available
                    # since it provides more information
                    if random.random() < 0.6:
                        return 1  # Take from discard pile
                    else:
                        return 0  # Draw from deck
                else:
                    return random.choice(valid_actions)
            else:
                # Exploitation with softmax exploration
                # Apply temperature to Q-values
                temperature = max(0.5, 1.0 - self.epsilon)  # Lower temperature as epsilon decreases
                exp_q = np.exp(np.array(list(valid_q.values())) / temperature)
                probs = exp_q / np.sum(exp_q)
                
                # Sample action based on softmax probabilities
                actions = list(valid_q.keys())
                if random.random() < 0.8:  # 80% chance to use softmax
                    return np.random.choice(actions, p=probs)
                else:
                    # 20% chance to pick the best action
                    return max(valid_q, key=valid_q.get)
        else:
            # During evaluation, always pick the best action
            return max(valid_q, key=valid_q.get)
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer with win detection."""
        # Track episode rewards to detect winning episodes
        self.current_episode_rewards += reward
        
        # Add to buffer (win status will be determined at end of episode)
        is_win = False
        if done:
            # If final reward is positive, consider it a win
            is_win = self.current_episode_rewards > 0
            self.current_episode_rewards = 0  # Reset for next episode
            
        self.memory.add(state, action, reward, next_state, done, win=is_win)
    
    def learn(self):
        """Update Q-network from experiences in replay buffer with Double DQN and PER."""
        if len(self.memory) < self.batch_size:
            return
        
        # Ensure network is in training mode
        self.q_network.train()
        
        # Sample batch from prioritized replay buffer
        batch, indices, weights = self.memory.sample(self.batch_size)
        
        if not batch:  # Empty batch check
            return None
        
        # Convert to tensors - optimize for pre-allocated arrays
        if hasattr(self.memory, '_initialized') and self.memory._initialized:
            # Data is already in numpy arrays, just convert to tensors
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Handle device-specific memory optimizations
            if self.device.type == 'cuda':
                # Pin memory for faster CPU->GPU transfer (CUDA only)
                states_tensor = torch.from_numpy(np.array(states, dtype=np.float32)).pin_memory().to(self.device, non_blocking=True)
                next_states_tensor = torch.from_numpy(np.array(next_states, dtype=np.float32)).pin_memory().to(self.device, non_blocking=True)
            else:
                # For CPU or MPS, don't use pin_memory
                states_tensor = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
                next_states_tensor = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
            
            # Smaller tensors can be transferred directly
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        else:
            # Need to convert from list of tuples
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Pre-allocate numpy arrays for faster conversion
            states_np = np.array(states, dtype=np.float32)
            next_states_np = np.array(next_states, dtype=np.float32)
            
            # Handle device-specific memory optimizations
            if self.device.type == 'cuda':
                # Pin memory for faster CPU->GPU transfer (CUDA only)
                states_tensor = torch.from_numpy(states_np).pin_memory().to(self.device, non_blocking=True)
                next_states_tensor = torch.from_numpy(next_states_np).pin_memory().to(self.device, non_blocking=True)
            else:
                # For CPU or MPS, don't use pin_memory
                states_tensor = torch.from_numpy(states_np).to(self.device)
                next_states_tensor = torch.from_numpy(next_states_np).to(self.device)
            
            # Smaller tensors can be transferred directly
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Convert weights to tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Compute current Q values and target Q values using mixed precision when available
        if self.use_amp:
            with torch.cuda.amp.autocast():
                # Compute Q-values for both current and next states in a single forward pass
                # This reduces kernel launch overhead
                combined_states = torch.cat([states_tensor, next_states_tensor], dim=0)
                combined_q_values = self.q_network(combined_states)
                
                # Split the results
                batch_size = states_tensor.shape[0]
                current_q_all = combined_q_values[:batch_size]
                next_q_all = combined_q_values[batch_size:]
                
                # Gather current Q values for the actions taken
                current_q = current_q_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Compute target Q values using Double DQN
                with torch.no_grad():
                    # Get actions from current network (already computed above)
                    next_actions = next_q_all.argmax(1, keepdim=True)
                    
                    # Get Q-values from target network for those actions
                    # Use torch.no_grad() to avoid tracking gradients
                    target_q_all = self.target_network(next_states_tensor)
                    next_q = target_q_all.gather(1, next_actions).squeeze(1)
                    
                    # Compute target Q values
                    target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q
                
                # Compute loss with importance sampling weights
                # Use huber loss for better stability
                td_errors_tensor = current_q - target_q
                loss = torch.mean(weights_tensor * torch.nn.functional.huber_loss(current_q, target_q, reduction='none'))
            
            # Optimize with mixed precision
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Compute TD errors for updating priorities - do this on GPU to avoid extra transfers
            with torch.no_grad():
                td_errors = torch.abs(td_errors_tensor).detach().cpu().numpy()
        else:
            # Standard precision training
            # Compute Q-values for both current and next states in a single forward pass
            combined_states = torch.cat([states_tensor, next_states_tensor], dim=0)
            combined_q_values = self.q_network(combined_states)
            
            # Split the results
            batch_size = states_tensor.shape[0]
            current_q_all = combined_q_values[:batch_size]
            next_q_all = combined_q_values[batch_size:]
            
            # Gather current Q values for the actions taken
            current_q = current_q_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values using Double DQN
            with torch.no_grad():
                # Get actions from current network (already computed above)
                next_actions = next_q_all.argmax(1, keepdim=True)
                
                # Get Q-values from target network for those actions
                target_q_all = self.target_network(next_states_tensor)
                next_q = target_q_all.gather(1, next_actions).squeeze(1)
                
                # Compute target Q values
                target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q
            
            # Compute TD errors for updating priorities
            td_errors_tensor = current_q - target_q
            td_errors = torch.abs(td_errors_tensor).detach().cpu().numpy()
            
            # Compute loss with importance sampling weights
            # Use huber loss for better stability
            loss = torch.mean(weights_tensor * torch.nn.functional.huber_loss(current_q, target_q, reduction='none'))
            
            # Optimize the model
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            
            self.optimizer.step()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            # Use faster in-place copy for target network update
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(param.data)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        """Save model weights and PER state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'current_episode': self.current_episode,  # Save current episode for warmup tracking
            # Don't save the buffer itself, but save the PER parameters
            'per_alpha': self.memory.alpha,
            'per_beta': self.memory.beta,
            'per_max_priority': self.memory.max_priority,
            # Save AMP scaler if using mixed precision
            'amp_scaler': self.scaler.state_dict() if self.use_amp else None
        }, path)
    
    def load(self, path: str):
        """Load model weights and PER state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        
        # Load current episode if available
        if 'current_episode' in checkpoint:
            self.current_episode = checkpoint['current_episode']
        
        # If the saved model has PER parameters, load them
        if 'per_alpha' in checkpoint and 'per_beta' in checkpoint and 'per_max_priority' in checkpoint:
            self.memory.alpha = checkpoint['per_alpha']
            self.memory.beta = checkpoint['per_beta']
            self.memory.max_priority = checkpoint['per_max_priority']
            
        # Load AMP scaler if available and using mixed precision
        if self.use_amp and 'amp_scaler' in checkpoint and checkpoint['amp_scaler'] is not None:
            self.scaler.load_state_dict(checkpoint['amp_scaler'])
            
    def update_epsilon(self, decay_rate=None):
        """
        Update epsilon using a piecewise strategy:
        - Keep epsilon at epsilon_start for epsilon_warmup_episodes
        - Then decay according to the decay formula
        
        Args:
            decay_rate: Optional custom decay rate (if None, use self.epsilon_decay)
        """
        # Increment episode counter
        self.current_episode += 1
        
        # During warmup period, keep epsilon at start value
        if self.current_episode <= self.epsilon_warmup_episodes:
            # Keep epsilon at start value
            return
        
        # After warmup, apply decay
        if decay_rate is None:
            # Use exponential decay with epsilon_decay parameter
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
        else:
            # Use alternative decay formula if provided
            # Formula: epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-decay_rate * (episode - warmup))
            post_warmup_episode = self.current_episode - self.epsilon_warmup_episodes
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(-decay_rate * post_warmup_episode)
            ) 