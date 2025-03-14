import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Any, Optional
import os
import time
from tqdm import tqdm, trange

# Neural network for Deep CFR
class DeepCFRNetwork(nn.Module):
    """
    Neural network for Deep CFR algorithm.
    
    This network predicts advantages for each action in a given state.
    """
    
    def __init__(self, input_size: int = 28, hidden_size: int = 512, output_size: int = 9, 
                 embedding_dim: int = 8, num_card_ranks: int = 13):
        super(DeepCFRNetwork, self).__init__()
        
        # Determine if CUDA is available for optimizations
        self.use_cuda = torch.cuda.is_available()
        
        # Card embedding layer (13 possible ranks -> embedding_dim)
        self.card_embedding = nn.Embedding(num_card_ranks + 1, embedding_dim)  # +1 for unknown/masked cards
        
        # Calculate the size after embeddings
        # 14 card positions (6 player cards + 6 opponent cards + discard + drawn)
        self.num_card_positions = 14
        embedded_size = self.num_card_positions * embedding_dim
        
        # Add the non-card features (12 binary flags for revealed cards + 2 game state flags)
        self.non_card_features = 14
        total_features = embedded_size + self.non_card_features
        
        # Network with three hidden layers
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(total_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),
            
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
        
        # Pass through the network
        return self.network(combined_features)


class ReservoirBuffer:
    """
    Reservoir buffer for storing advantage samples.
    
    This buffer maintains a fixed-size random sample of experiences
    when more experiences are added than capacity.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.count = 0
    
    def add(self, experience: Tuple):
        """Add an experience to the buffer using reservoir sampling."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # Reservoir sampling
            idx = random.randint(0, self.count)
            if idx < self.capacity:
                self.buffer[idx] = experience
        self.count += 1
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences from the buffer."""
        if len(self.buffer) == 0:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class DeepCFRAgent:
    """
    Deep Counterfactual Regret Minimization (Deep CFR) agent for playing Golf.
    
    This agent uses Deep CFR to learn a strategy for the imperfect information game.
    """
    
    def __init__(
        self,
        state_size: int = 29,  # 14 card indices + 15 binary features (including turn progress)
        action_size: int = 9,  # Number of possible actions in the game
        hidden_size: int = 512,  # Size of hidden layers in neural network
        embedding_dim: int = 8,  # Dimension of card embeddings
        num_card_ranks: int = 13,  # Number of possible card ranks
        learning_rate: float = 0.0001,  # Learning rate for optimizer
        buffer_size: int = 100000,  # Size of advantage buffer
        batch_size: int = 256,  # Number of samples per training batch
        cfr_iterations: int = 100,  # Number of CFR iterations
        traversals_per_iter: int = 100,  # Number of traversals per CFR iteration
        advantage_train_steps: int = 200,  # Number of training steps for advantage network (reduced from 1000)
        strategy_train_steps: int = 200,  # Number of training steps for strategy network (reduced from 1000)
        epsilon: float = 0.1,  # Exploration parameter
        early_stopping_patience: int = 5,  # Early stopping patience for training
        early_stopping_threshold: float = 0.0001  # Threshold for early stopping
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_card_ranks = num_card_ranks
        self.learning_rate = learning_rate
        self.cfr_iterations = cfr_iterations
        self.traversals_per_iter = traversals_per_iter
        self.advantage_train_steps = advantage_train_steps
        self.strategy_train_steps = strategy_train_steps
        self.epsilon = epsilon
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        # Buffer parameters
        self.buffer_size = buffer_size
        self.batch_size = max(2, batch_size)  # Ensure batch size is at least 2
        
        # Initialize advantage buffers for each player
        self.advantage_buffers = [ReservoirBuffer(buffer_size) for _ in range(2)]
        
        # Initialize strategy buffer
        self.strategy_buffer = ReservoirBuffer(buffer_size)
        
        # Initialize networks
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
        
        # Initialize advantage networks for each player
        self.advantage_networks = [
            DeepCFRNetwork(
                input_size=state_size,
                hidden_size=hidden_size,
                output_size=action_size,
                embedding_dim=embedding_dim,
                num_card_ranks=num_card_ranks
            ).to(self.device) for _ in range(2)
        ]
        
        # Initialize strategy network
        self.strategy_network = DeepCFRNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=action_size,
            embedding_dim=embedding_dim,
            num_card_ranks=num_card_ranks
        ).to(self.device)
        
        # Initialize optimizers
        self.advantage_optimizers = [
            optim.Adam(network.parameters(), lr=learning_rate)
            for network in self.advantage_networks
        ]
        
        self.strategy_optimizer = optim.Adam(
            self.strategy_network.parameters(), lr=learning_rate
        )
        
        # For mixed precision training when CUDA is available
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.advantage_scalers = [torch.cuda.amp.GradScaler() for _ in range(2)]
            self.strategy_scaler = torch.cuda.amp.GradScaler()
        
        # Current iteration
        self.current_iteration = 0
        
        # For tracking training progress
        self.advantage_losses = [[] for _ in range(2)]
        self.strategy_losses = []
    
    @staticmethod
    def random_action(valid_actions: List[int]) -> int:
        """
        Select a random action from valid actions.
        
        Args:
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        # Check if we're in the draw phase (both actions 0 and 1 are valid)
        if 0 in valid_actions and 1 in valid_actions:
            # Draw phase: 50/50 split between drawing from deck (0) and discard pile (1)
            if random.random() < 0.5:
                return 0  # Draw from deck
            else:
                return 1  # Take from discard pile
        else:
            # Not in draw phase, choose randomly from valid actions
            return random.choice(valid_actions)
    
    def select_action(self, state: np.ndarray, valid_actions: List[int], player: int = 0, training: bool = True) -> int:
        """
        Select an action using the current strategy.
        
        Args:
            state: Current state observation
            valid_actions: List of valid actions
            player: Current player (0 or 1)
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        # Exploration during training
        if training and random.random() < self.epsilon:
            return self.random_action(valid_actions)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get advantages from strategy network
        self.strategy_network.eval()
        with torch.no_grad():
            advantages = self.strategy_network(state_tensor).cpu().numpy()[0]
        
        # Filter for valid actions only
        valid_advantages = {action: advantages[action] for action in valid_actions}
        
        # Select action with highest advantage
        return max(valid_advantages, key=valid_advantages.get)
    
    def _cfr_traverse(self, env, player: int, iteration: int, traverser: int):
        """
        Traverse the game tree using CFR.
        
        Args:
            env: Game environment
            player: Current player
            iteration: Current CFR iteration
            traverser: Player for whom we're calculating advantages
            
        Returns:
            Expected value for the traverser
        """
        # Check if game is over
        if env.game_over:
            # Get final payoff based on scores
            player0_score = env._calculate_score(0)
            player1_score = env._calculate_score(1)
            
            # Calculate rewards as negative of scores (lower score is better)
            # plus a small bonus (0.5) for winning
            rewards = [-player0_score, -player1_score]
            
            # Add winning bonus
            if player0_score < player1_score:
                # Player 0 wins
                rewards[0] += 10
            elif player0_score > player1_score:
                # Player 1 wins
                rewards[1] += 10
                
            return rewards[traverser]
        # Get current state and valid actions
        state = env._get_observation()
        valid_actions = env._get_valid_actions()
        
        # If it's the traverser's turn
        if player == traverser:
            # Get advantages from the advantage network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                advantages = self.advantage_networks[player](state_tensor).cpu().numpy()[0]
            
            # Calculate regret-matching strategy
            strategy = self._regret_matching(advantages, valid_actions)
            
            # Sample action from strategy
            action = np.random.choice(valid_actions, p=[strategy[a] for a in valid_actions])
            
            # Take action
            next_state, _, done, _ = env.step(action)  # Ignore immediate reward
            
            # Recursive call
            expected_value = self._cfr_traverse(env, 1 - player, iteration, traverser)
            
            # Calculate advantages for each action
            action_values = {}
            for a in valid_actions:
                # Create a copy of the environment
                env_copy = self._clone_env(env)
                
                # Take action in the copy
                next_state_copy, _, done_copy, _ = env_copy.step(a)  # Ignore immediate reward
                
                # Recursive call to get value
                action_values[a] = self._cfr_traverse(env_copy, 1 - player, iteration, traverser)
            
            # Calculate regrets
            for a in valid_actions:
                # Regret = action value - expected value
                regret = action_values[a] - expected_value
                
                # Store advantage sample
                self.advantage_buffers[player].add((state, a, regret, iteration))
            
            return expected_value
        
        # If it's the opponent's turn
        else:
            # Get strategy from the strategy network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                advantages = self.strategy_network(state_tensor).cpu().numpy()[0]
            
            # Calculate regret-matching strategy
            strategy = self._regret_matching(advantages, valid_actions)
            
            # Store strategy sample
            for a in valid_actions:
                self.strategy_buffer.add((state, a, strategy[a], iteration))
            
            # Sample action from strategy
            action = np.random.choice(valid_actions, p=[strategy[a] for a in valid_actions])
            
            # Take action
            next_state, _, done, _ = env.step(action)  # Ignore immediate reward
            
            # Recursive call
            return self._cfr_traverse(env, 1 - player, iteration, traverser)
    
    def _regret_matching(self, advantages: np.ndarray, valid_actions: List[int]) -> Dict[int, float]:
        """
        Calculate a strategy using regret matching.
        
        Args:
            advantages: Advantage values for each action
            valid_actions: List of valid actions
            
        Returns:
            Strategy dictionary mapping actions to probabilities
        """
        # Filter advantages for valid actions
        valid_advantages = {a: max(0, advantages[a]) for a in valid_actions}
        
        # Calculate sum of positive advantages
        advantage_sum = sum(valid_advantages.values())
        
        # If all advantages are negative or zero, use uniform strategy
        if advantage_sum <= 0:
            return {a: 1.0 / len(valid_actions) for a in valid_actions}
        
        # Calculate strategy using regret matching
        return {a: valid_advantages[a] / advantage_sum for a in valid_actions}
    
    def _clone_env(self, env):
        """
        Create a lightweight copy of the environment.
        
        This is an optimized version that avoids deep copying the entire environment.
        Instead, it only copies the essential state variables needed for CFR.
        
        Args:
            env: Environment to clone
            
        Returns:
            Cloned environment
        """
        # Create a new environment
        cloned_env = type(env)(num_players=env.num_players, normalize_rewards=env.normalize_rewards)
        
        # Copy essential state variables
        cloned_env.deck = env.deck.copy()
        cloned_env.player_hands = [hand.copy() for hand in env.player_hands]
        cloned_env.current_player = env.current_player
        cloned_env.discard_pile = env.discard_pile.copy()
        cloned_env.game_over = env.game_over
        cloned_env.revealed_cards = [revealed.copy() for revealed in env.revealed_cards]
        cloned_env.drawn_card = env.drawn_card
        cloned_env.drawn_from_discard = env.drawn_from_discard
        cloned_env.final_round = env.final_round
        cloned_env.last_player = env.last_player
        cloned_env.turn_count = env.turn_count
        cloned_env.max_turns = env.max_turns
        
        return cloned_env
    
    def train_advantage_network(self, player: int):
        """
        Train the advantage network for a player.
        
        Args:
            player: Player index (0 or 1)
        """
        if len(self.advantage_buffers[player]) < self.batch_size:
            return None
        
        # Set network to training mode
        self.advantage_networks[player].train()
        
        # Track total loss
        total_loss = 0
        best_loss = float('inf')
        patience_counter = 0
        
        # Train for multiple steps with tqdm progress bar
        train_steps = trange(self.advantage_train_steps, desc=f"Training Adv Net P{player}", leave=False)
        for step in train_steps:
            # Sample batch from buffer
            batch = self.advantage_buffers[player].sample(self.batch_size)
            
            # Check if batch is empty
            if not batch or len(batch) == 0:
                continue
            
            # Unpack batch
            states, actions, regrets, iterations = zip(*batch)
            
            # Convert to numpy arrays
            states_np = np.array(states, dtype=np.float32)
            
            # Convert to tensors
            states_tensor = torch.from_numpy(states_np).to(self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            regrets_tensor = torch.tensor(regrets, dtype=torch.float32, device=self.device)
            iterations_tensor = torch.tensor(iterations, dtype=torch.float32, device=self.device)
            
            # Calculate weights based on iteration (more recent iterations have higher weight)
            weights = torch.pow(iterations_tensor / self.current_iteration, 2)
            
            # Train with mixed precision if available
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    advantages = self.advantage_networks[player](states_tensor)
                    predicted_regrets = advantages.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                    
                    # Calculate loss (weighted MSE)
                    loss = torch.mean(weights * (predicted_regrets - regrets_tensor) ** 2)
                
                # Backward pass with mixed precision
                self.advantage_optimizers[player].zero_grad(set_to_none=True)
                self.advantage_scalers[player].scale(loss).backward()
                self.advantage_scalers[player].unscale_(self.advantage_optimizers[player])
                torch.nn.utils.clip_grad_norm_(self.advantage_networks[player].parameters(), 1.0)
                self.advantage_scalers[player].step(self.advantage_optimizers[player])
                self.advantage_scalers[player].update()
            else:
                # Forward pass
                advantages = self.advantage_networks[player](states_tensor)
                predicted_regrets = advantages.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Calculate loss (weighted MSE)
                loss = torch.mean(weights * (predicted_regrets - regrets_tensor) ** 2)
                
                # Backward pass
                self.advantage_optimizers[player].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.advantage_networks[player].parameters(), 1.0)
                self.advantage_optimizers[player].step()
            
            # Track loss
            current_loss = loss.item()
            total_loss += current_loss
            
            # Update progress bar
            train_steps.set_postfix({'loss': f"{current_loss:.6f}"})
            
            # Early stopping check
            if current_loss < best_loss - self.early_stopping_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                train_steps.set_description(f"Early stopping at step {step+1}/{self.advantage_train_steps}")
                break
        
        # Calculate average loss
        steps_completed = min(step + 1, self.advantage_train_steps)
        avg_loss = total_loss / steps_completed if steps_completed > 0 else 0
        self.advantage_losses[player].append(avg_loss)
        
        return avg_loss
    
    def train_strategy_network(self):
        """Train the strategy network."""
        if len(self.strategy_buffer) < self.batch_size:
            return None
        
        # Set network to training mode
        self.strategy_network.train()
        
        # Track total loss
        total_loss = 0
        best_loss = float('inf')
        patience_counter = 0
        
        # Train for multiple steps with tqdm progress bar
        train_steps = trange(self.strategy_train_steps, desc="Training Strategy Net", leave=False)
        for step in train_steps:
            # Sample batch from buffer
            batch = self.strategy_buffer.sample(self.batch_size)
            
            # Check if batch is empty
            if not batch or len(batch) == 0:
                continue
            
            # Unpack batch
            states, actions, probs, iterations = zip(*batch)
            
            # Convert to numpy arrays
            states_np = np.array(states, dtype=np.float32)
            
            # Convert to tensors
            states_tensor = torch.from_numpy(states_np).to(self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            probs_tensor = torch.tensor(probs, dtype=torch.float32, device=self.device)
            iterations_tensor = torch.tensor(iterations, dtype=torch.float32, device=self.device)
            
            # Calculate weights based on iteration (more recent iterations have higher weight)
            weights = torch.pow(iterations_tensor / self.current_iteration, 2)
            
            # Train with mixed precision if available
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    logits = self.strategy_network(states_tensor)
                    
                    # Apply softmax to get probabilities
                    probs_all = torch.nn.functional.softmax(logits, dim=1)
                    predicted_probs = probs_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                    
                    # Calculate loss (weighted cross-entropy)
                    loss = -torch.mean(weights * probs_tensor * torch.log(predicted_probs + 1e-8))
                
                # Backward pass with mixed precision
                self.strategy_optimizer.zero_grad(set_to_none=True)
                self.strategy_scaler.scale(loss).backward()
                self.strategy_scaler.unscale_(self.strategy_optimizer)
                torch.nn.utils.clip_grad_norm_(self.strategy_network.parameters(), 1.0)
                self.strategy_scaler.step(self.strategy_optimizer)
                self.strategy_scaler.update()
            else:
                # Forward pass
                logits = self.strategy_network(states_tensor)
                
                # Apply softmax to get probabilities
                probs_all = torch.nn.functional.softmax(logits, dim=1)
                predicted_probs = probs_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Calculate loss (weighted cross-entropy)
                loss = -torch.mean(weights * probs_tensor * torch.log(predicted_probs + 1e-8))
                
                # Backward pass
                self.strategy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.strategy_network.parameters(), 1.0)
                self.strategy_optimizer.step()
            
            # Track loss
            current_loss = loss.item()
            total_loss += current_loss
            
            # Update progress bar
            train_steps.set_postfix({'loss': f"{current_loss:.6f}"})
            
            # Early stopping check
            if current_loss < best_loss - self.early_stopping_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                train_steps.set_description(f"Early stopping at step {step+1}/{self.strategy_train_steps}")
                break
        
        # Calculate average loss
        steps_completed = min(step + 1, self.strategy_train_steps)
        avg_loss = total_loss / steps_completed if steps_completed > 0 else 0
        self.strategy_losses.append(avg_loss)
        
        return avg_loss
    
    def _batch_cfr_traverse(self, envs, player_indices, iteration, traverser):
        """
        Process multiple CFR traversals in parallel to improve efficiency.
        
        Args:
            envs: List of game environments
            player_indices: List of current players for each environment
            iteration: Current CFR iteration
            traverser: Player for whom we're calculating advantages
            
        Returns:
            List of expected values for the traverser
        """
        results = []
        
        for env, player in zip(envs, player_indices):
            # Process each environment
            value = self._cfr_traverse(env, player, iteration, traverser)
            results.append(value)
            
        return results
    
    def train(self, env):
        """
        Train the agent using Deep CFR.
        
        Args:
            env: Game environment
        """
        # Increment iteration counter
        self.current_iteration += 1
        
        # Perform CFR traversals with tqdm progress bar
        traversals = trange(self.traversals_per_iter, desc="CFR Traversals", leave=False)
        
        # Batch size for parallel traversals
        batch_size = min(10, self.traversals_per_iter)  # Process up to 10 environments at once
        
        for i in range(0, self.traversals_per_iter, batch_size):
            # Create batch of environments
            batch_count = min(batch_size, self.traversals_per_iter - i)
            
            # Player 0 traversals
            envs_p0 = [env.__class__(num_players=env.num_players) for _ in range(batch_count)]
            self._batch_cfr_traverse(envs_p0, [0] * batch_count, self.current_iteration, 0)
            
            # Player 1 traversals
            envs_p1 = [env.__class__(num_players=env.num_players) for _ in range(batch_count)]
            self._batch_cfr_traverse(envs_p1, [0] * batch_count, self.current_iteration, 1)
            
            # Update progress bar
            traversals.update(batch_count)
            traversals.set_postfix({
                'P0 Buffer': len(self.advantage_buffers[0]),
                'P1 Buffer': len(self.advantage_buffers[1]),
                'Strat Buffer': len(self.strategy_buffer)
            })
        
        # Train advantage networks
        advantage_loss_0 = self.train_advantage_network(0)
        advantage_loss_1 = self.train_advantage_network(1)
        
        # Train strategy network
        strategy_loss = self.train_strategy_network()
        
        return {
            'advantage_loss_0': advantage_loss_0,
            'advantage_loss_1': advantage_loss_1,
            'strategy_loss': strategy_loss
        }
    
    def save(self, path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'advantage_network_0': self.advantage_networks[0].state_dict(),
            'advantage_network_1': self.advantage_networks[1].state_dict(),
            'strategy_network': self.strategy_network.state_dict(),
            'advantage_optimizer_0': self.advantage_optimizers[0].state_dict(),
            'advantage_optimizer_1': self.advantage_optimizers[1].state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'current_iteration': self.current_iteration,
            'embedding_dim': self.embedding_dim,
            'num_card_ranks': self.num_card_ranks,
            'advantage_losses': self.advantage_losses,
            'strategy_losses': self.strategy_losses,
            'amp_advantage_scalers': [scaler.state_dict() for scaler in self.advantage_scalers] if self.use_amp else None,
            'amp_strategy_scaler': self.strategy_scaler.state_dict() if self.use_amp else None
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check if the saved model uses embeddings
        if 'embedding_dim' in checkpoint and 'num_card_ranks' in checkpoint:
            # If embedding parameters differ, recreate the networks
            if checkpoint['embedding_dim'] != self.embedding_dim or checkpoint['num_card_ranks'] != self.num_card_ranks:
                self.embedding_dim = checkpoint['embedding_dim']
                self.num_card_ranks = checkpoint['num_card_ranks']
                
                # Recreate networks with the loaded embedding parameters
                for i in range(2):
                    self.advantage_networks[i] = DeepCFRNetwork(
                        input_size=self.state_size,
                        hidden_size=self.hidden_size,
                        output_size=self.action_size,
                        embedding_dim=self.embedding_dim,
                        num_card_ranks=self.num_card_ranks
                    ).to(self.device)
                
                self.strategy_network = DeepCFRNetwork(
                    input_size=self.state_size,
                    hidden_size=self.hidden_size,
                    output_size=self.action_size,
                    embedding_dim=self.embedding_dim,
                    num_card_ranks=self.num_card_ranks
                ).to(self.device)
                
                # Recreate optimizers
                self.advantage_optimizers = [
                    optim.Adam(network.parameters(), lr=self.learning_rate)
                    for network in self.advantage_networks
                ]
                
                self.strategy_optimizer = optim.Adam(
                    self.strategy_network.parameters(), lr=self.learning_rate
                )
        
        # Load network weights
        self.advantage_networks[0].load_state_dict(checkpoint['advantage_network_0'])
        self.advantage_networks[1].load_state_dict(checkpoint['advantage_network_1'])
        self.strategy_network.load_state_dict(checkpoint['strategy_network'])
        
        # Load optimizer states
        self.advantage_optimizers[0].load_state_dict(checkpoint['advantage_optimizer_0'])
        self.advantage_optimizers[1].load_state_dict(checkpoint['advantage_optimizer_1'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
        
        # Load current iteration
        self.current_iteration = checkpoint['current_iteration']
        
        # Load loss history if available
        if 'advantage_losses' in checkpoint:
            self.advantage_losses = checkpoint['advantage_losses']
        if 'strategy_losses' in checkpoint:
            self.strategy_losses = checkpoint['strategy_losses']
            
        # Load AMP scalers if available and using mixed precision
        if self.use_amp and 'amp_advantage_scalers' in checkpoint and checkpoint['amp_advantage_scalers'] is not None:
            for i, scaler_state in enumerate(checkpoint['amp_advantage_scalers']):
                self.advantage_scalers[i].load_state_dict(scaler_state)
        
        if self.use_amp and 'amp_strategy_scaler' in checkpoint and checkpoint['amp_strategy_scaler'] is not None:
            self.strategy_scaler.load_state_dict(checkpoint['amp_strategy_scaler']) 