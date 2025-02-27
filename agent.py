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
    
    def __init__(self, input_size: int = 60, hidden_size: int = 512, output_size: int = 9):
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
    Prioritized experience replay buffer with episode marking for winning trajectories.
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
                # Create an experience with max priority
                priority = self.max_priority * priority_multiplier
                
                # Add to buffer
                if self.size < self.capacity:
                    self.buffer.append(exp)
                    self.priorities[self.size] = priority
                    self.size += 1
                else:
                    self.buffer[self.position] = exp
                    self.priorities[self.position] = priority
                    self.position = (self.position + 1) % self.capacity
            
            # Clear the current episode
            self.current_episode = []
    
    def sample(self, batch_size: int) -> Tuple[List, List[int], np.ndarray]:
        """Sample a batch of experiences based on priorities."""
        if self.size == 0:
            return [], [], np.array([])
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, min(batch_size, self.size), p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Get sampled experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-5  # Add small epsilon to ensure non-zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """
    Deep Q-Network agent for playing Golf.
    """
    
    def __init__(
        self,
        state_size: int = 60,  # Simplified state space (removed suits)
        action_size: int = 9,
        hidden_size: int = 512,  # Increased from 256 to 512
        learning_rate: float = 0.0001,  # Reduced from 0.0005 to 0.0001 for more stable learning
        gamma: float = 0.99,  # Discount factor
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,  # Reduced from 0.05 to 0.01 for more exploitation in late stages
        epsilon_decay: float = 0.9995,  # Adjusted for slower decay
        buffer_size: int = 100000,  # Increased from 50000 to 100000
        batch_size: int = 256,  # Increased from 128 to 256
        target_update: int = 1000,  # Increased from 500 to 1000
        per_alpha: float = 0.7,  # Prioritization exponent
        per_beta: float = 0.5,  # Importance sampling start value
        per_beta_increment: float = 0.0001  # Beta increment per learning step
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
                elif 2 <= max(valid_actions) <= 7:  # Card replacement phase
                    # Encourage revealing unknown cards
                    # Extract the current player's revealed cards from the state
                    # This is a simplified approach - in practice, you'd need to extract this info from the state
                    # For now, we'll just use a random approach that favors actions 2-7 (replacing cards)
                    replace_actions = [a for a in valid_actions if 2 <= a <= 7]
                    if replace_actions:
                        return random.choice(replace_actions)
                    else:
                        return random.choice(valid_actions)
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
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)  # Importance sampling weights
        
        # Compute current Q values and target Q values using mixed precision when available
        if self.use_amp:
            with torch.cuda.amp.autocast():
                # Compute current Q values
                current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute target Q values using Double DQN
                with torch.no_grad():
                    # Get actions from current network
                    next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                    
                    # Get Q-values from target network for those actions
                    next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                    
                    # Compute target Q values
                    target_q = rewards + (1 - dones) * self.gamma * next_q
                
                # Compute loss with importance sampling weights
                loss = torch.mean(weights * (current_q - target_q) ** 2)
            
            # Optimize with mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            # Compute current Q values
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values using Double DQN
            with torch.no_grad():
                # Get actions from current network
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                
                # Get Q-values from target network for those actions
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                
                # Compute target Q values
                target_q = rewards + (1 - dones) * self.gamma * next_q
            
            # Compute TD errors for updating priorities
            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
            
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, td_errors)
            
            # Compute loss with importance sampling weights
            loss = torch.mean(weights * (current_q - target_q) ** 2)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            
            self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
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
        
        # If the saved model has PER parameters, load them
        if 'per_alpha' in checkpoint and 'per_beta' in checkpoint and 'per_max_priority' in checkpoint:
            self.memory.alpha = checkpoint['per_alpha']
            self.memory.beta = checkpoint['per_beta']
            self.memory.max_priority = checkpoint['per_max_priority']
            
        # Load AMP scaler if available and using mixed precision
        if self.use_amp and 'amp_scaler' in checkpoint and checkpoint['amp_scaler'] is not None:
            self.scaler.load_state_dict(checkpoint['amp_scaler']) 