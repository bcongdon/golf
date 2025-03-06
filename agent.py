import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple, Dict
from replay_buffer import PrioritizedReplayBuffer
from golf_game_v2 import Action

class GolfMLP(nn.Module):
    """
    Enhanced multi-layer perceptron for the Golf card game with card embeddings.
    
    Input: Game state representation with card indices
    Output: Q-values for each possible action
    """
    
    def __init__(self, input_size: int = 28, hidden_size: int = 512, output_size: int = 9, 
                 embedding_dim: int = 8, num_card_ranks: int = 13):
        super(GolfMLP, self).__init__()
        
        # Determine if CUDA is available for optimizations
        self.use_cuda = torch.cuda.is_available()
        
        # Card embedding layer (13 possible ranks -> embedding_dim)
        # Add +1 for unknown/masked cards and +1 for None (no card)
        self.card_embedding = nn.Embedding(num_card_ranks + 2, embedding_dim)  
        
        # Calculate the size after embeddings
        # 14 card positions (6 player cards + 6 opponent cards + discard + drawn)
        self.num_card_positions = 14
        embedded_size = self.num_card_positions * embedding_dim
        
        # Add the non-card features (12 binary flags for revealed cards + 2 game state flags)
        self.non_card_features = 14
        total_features = embedded_size + self.non_card_features
        
        # Network with two hidden layers and no dropout
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(total_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),  # LeakyReLU instead of ReLU for better gradient flow
            
            # First hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),

            # Second hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),
            
            # Output layer
            nn.Linear(hidden_size, output_size)
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
        
        # Pass through the network
        return self.network(combined_features)


class DQNAgent:
    """
    Deep Q-Network agent for playing Golf.
    """
    
    def __init__(
        self,
        state_size: int = 28,  # Size: 14 card indices + 14 binary features
        action_size: int = 9,  # Number of possible actions in the game
        hidden_size: int = 512,  # Size of hidden layers in neural network
        embedding_dim: int = 8,  # Dimension of card embeddings
        num_card_ranks: int = 13,  # Number of possible card ranks (A-K)
        learning_rate: float = 0.0001,  # Learning rate for optimizer
        gamma: float = 0.99,  # Discount factor for future rewards
        epsilon_start: float = 1.0,  # Initial exploration rate
        epsilon_end: float = 0.01,  # Minimum exploration rate
        epsilon_decay_episodes: int = 10000,  # Number of episodes to decay epsilon from start to end
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
        self.embedding_dim = embedding_dim
        self.num_card_ranks = num_card_ranks
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
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
            
        self.q_network = GolfMLP(
            input_size=state_size, 
            hidden_size=hidden_size, 
            output_size=action_size,
            embedding_dim=embedding_dim,
            num_card_ranks=num_card_ranks
        ).to(self.device)
        
        self.target_network = GolfMLP(
            input_size=state_size, 
            hidden_size=hidden_size, 
            output_size=action_size,
            embedding_dim=embedding_dim,
            num_card_ranks=num_card_ranks
        ).to(self.device)
        
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
        # Check if we're in the draw phase (both DRAW_FROM_DECK and DRAW_FROM_DISCARD are valid)
        if Action.DRAW_FROM_DECK in valid_actions and Action.DRAW_FROM_DISCARD in valid_actions:
            # Draw phase: 50/50 split between drawing from deck and discard pile
            if random.random() < 0.5:
                return Action.DRAW_FROM_DECK
            else:
                return Action.DRAW_FROM_DISCARD
        else:
            # Now in Swap phase.
            # If can discard, do so with 50% probability
            if Action.DISCARD in valid_actions:
                if random.random() < 0.5:
                    return Action.DISCARD

            # Otherwise, randomly choose an action
            return random.choice(valid_actions)
    
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
                # Use common random policy for exploration
                return self.random_action(valid_actions)
            else:
                # Exploitation - simply choose the action with highest Q-value
                return max(valid_q, key=valid_q.get)
        else:
            # During evaluation, always pick the best action
            return max(valid_q, key=valid_q.get)
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        # Track episode rewards
        self.current_episode_rewards += reward
        
        # Validate experience components
        if state is None or action is None or reward is None or next_state is None or done is None:
            print(f"Warning: Attempted to add experience with None values: ({state}, {action}, {reward}, {next_state}, {done})")
            return
        
        # Create experience tuple
        experience = (state, action, reward, next_state, done)
        
        # Add to buffer
        self.memory.add(experience)
        
        # Reset episode rewards if done
        if done:
            self.current_episode_rewards = 0
    
    def learn(self):
        """Update Q-network from experiences in replay buffer with Double DQN and PER."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Ensure network is in training mode
        self.q_network.train()
        
        try:
            # Sample batch from prioritized replay buffer
            batch, indices, weights = self.memory.sample(self.batch_size)
            
            # Check if batch is empty
            if not batch or len(batch) == 0:
                print("Warning: Empty batch returned from replay buffer")
                return None
            
            # Convert to tensors
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to numpy arrays
            states_np = np.array(states, dtype=np.float32)
            next_states_np = np.array(next_states, dtype=np.float32)
            
            # Convert to tensors
            states_tensor = torch.from_numpy(states_np).to(self.device)
            next_states_tensor = torch.from_numpy(next_states_np).to(self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        except (TypeError, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            print(batch)
            # stack trace
            import traceback
            traceback.print_exc()
            return None
        
        # Compute current Q values and target Q values using mixed precision when available
        if self.use_amp:
            with torch.cuda.amp.autocast():
                # Compute Q-values for current states
                current_q_all = self.q_network(states_tensor)
                current_q = current_q_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Compute target Q values using Double DQN
                with torch.no_grad():
                    # Get actions from current network
                    next_q_all = self.q_network(next_states_tensor)
                    next_actions = next_q_all.argmax(1, keepdim=True)
                    
                    # Get Q-values from target network for those actions
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
            # Compute Q-values for current states
            current_q_all = self.q_network(states_tensor)
            current_q = current_q_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values using Double DQN
            with torch.no_grad():
                # Get actions from current network
                next_q_all = self.q_network(next_states_tensor)
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
    
        return loss.item()
    
    def save(self, path: str):
        """Save model weights and all agent parameters."""
        torch.save({
            # Network architecture parameters
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.hidden_size,
            'embedding_dim': self.embedding_dim,
            'num_card_ranks': self.num_card_ranks,
            
            # Network weights and optimizer state
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            
            # Learning parameters
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update': self.target_update,
            
            # Exploration parameters
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_episodes': self.epsilon_decay_episodes,
            'epsilon_warmup_episodes': self.epsilon_warmup_episodes,
            'current_episode': self.current_episode,
            
            # Buffer parameters
            'buffer_size': self.buffer_size,
            'per_alpha': self.memory.alpha,
            'per_beta': self.memory.beta,
            'per_beta_increment': self.memory.beta_increment,
            'per_max_priority': self.memory.max_priority,
            
            # Training state
            'update_counter': self.update_counter,
            'current_episode_rewards': self.current_episode_rewards,
            
            # Device and optimization settings
            'use_amp': self.use_amp,
            'amp_scaler': self.scaler.state_dict() if self.use_amp else None
        }, path)
    
    def load(self, path: str):
        """Load model weights and all agent parameters."""
        try:
            # First try loading with weights_only=False (legacy mode)
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            # If that fails, try with explicit safe globals for numpy
            import numpy as np
            from torch.serialization import safe_globals, add_safe_globals
            
            # Add numpy scalar types to safe globals
            numpy_types = [
                np.int64, np.int32, np.float64, np.float32,
                np.bool_, np.ndarray, np.dtype,
                np.core.multiarray.scalar
            ]
            
            with safe_globals(*numpy_types):
                checkpoint = torch.load(path, map_location=self.device)
        
        # Load architecture parameters with defaults from current instance
        architecture_changed = False
        if any(param not in checkpoint for param in ['state_size', 'action_size', 'hidden_size', 'embedding_dim', 'num_card_ranks']):
            print("Warning: Loading legacy model format. Using current architecture parameters.")
        else:
            architecture_changed = (
                checkpoint.get('state_size', self.state_size) != self.state_size or
                checkpoint.get('action_size', self.action_size) != self.action_size or
                checkpoint.get('hidden_size', self.hidden_size) != self.hidden_size or
                checkpoint.get('embedding_dim', self.embedding_dim) != self.embedding_dim or
                checkpoint.get('num_card_ranks', self.num_card_ranks) != self.num_card_ranks
            )
        
        if architecture_changed:
            # Update architecture parameters
            self.state_size = checkpoint.get('state_size', self.state_size)
            self.action_size = checkpoint.get('action_size', self.action_size)
            self.hidden_size = checkpoint.get('hidden_size', self.hidden_size)
            self.embedding_dim = checkpoint.get('embedding_dim', self.embedding_dim)
            self.num_card_ranks = checkpoint.get('num_card_ranks', self.num_card_ranks)
            
            # Recreate networks with loaded parameters
            self.q_network = GolfMLP(
                input_size=self.state_size,
                hidden_size=self.hidden_size,
                output_size=self.action_size,
                embedding_dim=self.embedding_dim,
                num_card_ranks=self.num_card_ranks
            ).to(self.device)
            
            self.target_network = GolfMLP(
                input_size=self.state_size,
                hidden_size=self.hidden_size,
                output_size=self.action_size,
                embedding_dim=self.embedding_dim,
                num_card_ranks=self.num_card_ranks
            ).to(self.device)
        
        # Load network weights
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        
        # Load learning parameters with defaults
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.batch_size = checkpoint.get('batch_size', self.batch_size)
        self.target_update = checkpoint.get('target_update', self.target_update)
        
        # Recreate optimizer with loaded learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load exploration parameters with defaults
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon_start)
        self.epsilon_end = checkpoint.get('epsilon_end', self.epsilon_end)
        self.epsilon_decay_episodes = checkpoint.get('epsilon_decay_episodes', self.epsilon_decay_episodes)
        self.epsilon_warmup_episodes = checkpoint.get('epsilon_warmup_episodes', self.epsilon_warmup_episodes)
        self.current_episode = checkpoint.get('current_episode', self.current_episode)
        
        # Load buffer parameters with defaults
        self.buffer_size = checkpoint.get('buffer_size', self.buffer_size)
        # Recreate buffer with loaded parameters
        self.memory = PrioritizedReplayBuffer(
            capacity=self.buffer_size,
            alpha=checkpoint.get('per_alpha', self.memory.alpha),
            beta=checkpoint.get('per_beta', self.memory.beta),
            beta_increment=checkpoint.get('per_beta_increment', self.memory.beta_increment)
        )
        if 'per_max_priority' in checkpoint:
            self.memory.max_priority = checkpoint['per_max_priority']
        
        # Load training state with defaults
        self.update_counter = checkpoint.get('update_counter', self.update_counter)
        self.current_episode_rewards = checkpoint.get('current_episode_rewards', self.current_episode_rewards)
        
        # Load optimization settings with defaults
        self.use_amp = checkpoint.get('use_amp', self.use_amp)
        if self.use_amp and 'amp_scaler' in checkpoint and checkpoint['amp_scaler'] is not None:
            self.scaler = torch.cuda.amp.GradScaler()
            self.scaler.load_state_dict(checkpoint['amp_scaler'])
    
    def update_epsilon(self):
        """
        Update epsilon using a piecewise strategy:
        - Keep epsilon at epsilon_start for epsilon_warmup_episodes
        - Then use exponential decay from epsilon_start to epsilon_end over epsilon_decay_episodes
        """
        # Increment episode counter
        self.current_episode += 1
        
        # During warmup period, keep epsilon at start value
        if self.current_episode <= self.epsilon_warmup_episodes:
            self.epsilon = self.epsilon_start
            return
        
        # After warmup, apply exponential decay
        # Calculate how many episodes into the decay period we are
        decay_episodes = self.current_episode - self.epsilon_warmup_episodes
        
        # Calculate decay rate to reach epsilon_end after epsilon_decay_episodes
        # Using the formula: epsilon_end = epsilon_start * (decay_rate ^ epsilon_decay_episodes)
        # So decay_rate = (epsilon_end / epsilon_start) ^ (1 / epsilon_decay_episodes)
        decay_rate = (self.epsilon_end / self.epsilon_start) ** (1.0 / self.epsilon_decay_episodes)
        
        # Apply exponential decay
        self.epsilon = self.epsilon_start * (decay_rate ** min(decay_episodes, self.epsilon_decay_episodes))
        
        # Ensure we don't go below epsilon_end
        self.epsilon = max(self.epsilon_end, self.epsilon) 