import numpy as np
from typing import List, Tuple, Any, Set
import random


class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for efficient sampling based on priority.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize a SumTree with the given capacity.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        # Tree capacity is 2*capacity - 1 (capacity leaves + capacity-1 internal nodes)
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.valid_indices = set()  # Track valid indices
    
    def total_priority(self) -> float:
        """
        Get the total priority in the tree.
        
        Returns:
            Sum of all priorities
        """
        return self.tree[0]
    
    def update(self, idx: int, priority: float):
        """
        Update the priority at the given index.
        
        Args:
            idx: Index in the tree
            priority: New priority value
        """
        # Ensure priority is at least a small positive value
        priority = max(priority, 1e-5)
        
        # Calculate change in priority
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        
        # If this is a leaf node, add the corresponding buffer index to valid_indices
        if idx >= self.capacity - 1:
            buffer_idx = idx - (self.capacity - 1)
            self.valid_indices.add(buffer_idx)
        
        # Propagate change up the tree
        self._propagate(idx, change)
    
    def _propagate(self, idx: int, change: float):
        """
        Propagate the priority change up the tree.
        
        Args:
            idx: Index of the node that was updated
            change: Change in priority
        """
        # Loop until we reach the root
        while idx > 0:
            # Get parent index
            parent = (idx - 1) // 2
            
            # Update parent's priority
            self.tree[parent] += change
            
            # Move to parent
            idx = parent
    
    def get_leaf(self, value: float) -> Tuple[int, int, float]:
        """
        Find the leaf node that corresponds to the given value.
        
        Args:
            value: Value to search for (in range [0, total_priority])
            
        Returns:
            Tuple of (leaf_idx, buffer_idx, priority)
        """
        return self._retrieve(0, value)
    
    def _retrieve(self, idx: int, value: float) -> Tuple[int, int, float]:
        """
        Recursively search for the leaf node corresponding to the given value.
        
        Args:
            idx: Current node index
            value: Value to search for
            
        Returns:
            Tuple of (leaf_idx, buffer_idx, priority)
        """
        # Get left and right children
        left = 2 * idx + 1
        right = left + 1
        
        # If we're at a leaf node, return it
        if left >= len(self.tree):
            # Convert tree index to buffer index
            buffer_idx = idx - (self.capacity - 1)
            return idx, buffer_idx, max(self.tree[idx], 1e-5)  # Ensure non-zero priority
        
        # Otherwise, go left or right based on value
        # If value <= left's priority, go left
        # Otherwise, go right and subtract left's priority from value
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])


class PrioritizedReplayBuffer:
    """
    A replay buffer that samples experiences based on their priorities.
    
    Higher priority experiences are sampled more frequently.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001, epsilon: float = 0.01):
        """
        Initialize a prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Exponent for prioritization (0 = uniform, 1 = full prioritization)
            beta: Exponent for importance sampling (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each time we sample
            epsilon: Small positive value to ensure non-zero priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Initialize buffer and priorities
        self.buffer = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        # Track valid indices (indices with actual experiences)
        self.valid_indices = set()
        
        # Track current size and position
        self.size = 0
        self.position = 0
        
        # Maximum priority seen so far (for new experiences)
        self.max_priority = 1.0
    
    def add(self, experience: Any):
        """
        Add a new experience to the buffer.
        
        Args:
            experience: Experience to add (can be any object)
        """
        # Get the next available index
        idx = self.position
        
        # Store experience in buffer
        self.buffer[idx] = experience
        
        # Update priority with max priority for new experience
        # Using max priority ensures new experiences are sampled at least once
        self.priorities[idx] = self.max_priority ** self.alpha
        
        # Add to valid indices
        self.valid_indices.add(idx)
        
        # Update position
        self.position = (self.position + 1) % self.capacity
        
        # Update size
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], np.ndarray]:
        """
        Sample experiences from the buffer based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, indices, weights)
        """
        # Initialize arrays for indices and weights
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        experiences = []
        
        # If buffer is empty, return empty arrays
        if self.size == 0:
            return [None] * batch_size, indices, weights
        
        # Get valid indices and their priorities
        valid_indices = list(self.valid_indices)
        print(f"DEBUG: Valid indices: {valid_indices}")
        
        # Get priorities for valid indices
        valid_priorities = self.priorities[valid_indices]
        print(f"DEBUG: Priorities: {valid_priorities}")
        
        # Calculate total priority
        total_priority = np.sum(valid_priorities)
        print(f"DEBUG: Total priority: {total_priority}")
        
        # If total priority is too small, use uniform sampling
        if total_priority < 1e-6:
            sampled_indices = np.random.choice(valid_indices, size=batch_size)
            for i, idx in enumerate(sampled_indices):
                indices[i] = idx
                experiences.append(self.buffer[idx])
                weights[i] = 1.0
            return experiences, indices, weights
        
        # Sample based on priorities
        for i in range(batch_size):
            # Sample random value between 0 and total priority
            value = np.random.uniform(0, total_priority)
            
            # Find index corresponding to this value
            cumulative_priority = 0
            for j, idx in enumerate(valid_indices):
                cumulative_priority += valid_priorities[j]
                if cumulative_priority > value:
                    sampled_idx = idx
                    sampled_priority = valid_priorities[j]
                    break
            else:
                # Fallback if we didn't find an index (shouldn't happen)
                sampled_idx = valid_indices[0]
                sampled_priority = valid_priorities[0]
            
            # Store sampled index and experience
            indices[i] = sampled_idx
            experiences.append(self.buffer[sampled_idx])
            
            # Calculate weight for importance sampling
            # P(i) = p_i^alpha / sum_k p_k^alpha
            # w_i = (1/N * 1/P(i))^beta = (N * sum_k p_k^alpha / p_i^alpha)^beta
            prob = sampled_priority / total_priority
            weight = (len(valid_indices) * prob) ** (-self.beta)
            weights[i] = weight
        
        # Normalize weights to have max weight = 1
        if len(weights) > 0:
            weights = weights / np.max(weights)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Return sampled experiences, indices, and weights
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for given indices.
        
        Args:
            indices: List of indices to update
            priorities: List of priorities to set
        """
        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive
            priority = max(priority, self.epsilon)
            
            # Update max priority if needed
            self.max_priority = max(self.max_priority, priority)
            
            # Check if index is valid
            if 0 <= idx < self.capacity:
                # Add to valid indices
                self.valid_indices.add(idx)
                
                # Update priority in the priorities array
                self.priorities[idx] = priority ** self.alpha
                
                
    def __len__(self) -> int:
        """
        Get current size of buffer.
        
        Returns:
            Number of experiences in buffer
        """
        return self.size 