"""
Replay Buffer implementation from OpenAI Baselines.
Source: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

This implementation has been adapted to work with our existing code.
"""

import numpy as np
import random
from typing import List, Tuple, Any

from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.size = 0  # For compatibility with PrioritizedReplayBuffer

    def __len__(self):
        return len(self._storage)

    def add(self, experience: Any):
        """Add a new experience to the buffer.
        
        Args:
            experience: Experience to add (can be any object)
        """
        # Skip None experiences
        if experience is None:
            print("Warning: Attempted to add None experience to replay buffer")
            return
            
        # Check if any component of the experience is None
        if hasattr(experience, '__iter__') and not isinstance(experience, str):
            if any(x is None for x in experience):
                print(f"Warning: Experience contains None values: {experience}")
                return

        if self._next_idx >= len(self._storage):
            self._storage.append(experience)
        else:
            self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self.size = min(self.size + 1, self._maxsize)  # Update size for compatibility

    def _encode_sample(self, idxes):
        experiences = []
        for i in idxes:
            experiences.append(self._storage[i])
        return experiences

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        experiences: List
            batch of experiences
        indices: List
            indices of sampled experiences
        weights: np.array
            array of ones (no importance sampling)
        """
        if len(self._storage) == 0:
            raise ValueError("Cannot sample from an empty buffer")
            
        actual_batch_size = min(batch_size, len(self._storage))
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(actual_batch_size)]
        weights = np.ones_like(idxes, dtype=np.float32)
        return self._encode_sample(idxes), idxes, weights
        
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        No-op method for compatibility with PrioritizedReplayBuffer.
        Regular replay buffer doesn't use priorities.
        
        Parameters
        ----------
        indices: [int]
            List of indices of sampled transitions
        priorities: [float]
            List of updated priorities
        """
        pass  # No-op for regular replay buffer


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        beta_increment: float
            Amount to increase beta each time we sample
        epsilon: float
            Small positive value to ensure non-zero priority

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        assert alpha >= 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.size = 0  # For compatibility with existing code

        # Find capacity that is a power of 2
        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self.max_priority = 1.0  # For compatibility with existing code

    def add(self, experience: Any):
        """Add a new experience to the buffer.
        
        Args:
            experience: Experience to add (can be any object)
        """
        # Skip None experiences
        if experience is None:
            print("Warning: Attempted to add None experience to replay buffer")
            return
            
        # Check if any component of the experience is None
        if hasattr(experience, '__iter__') and not isinstance(experience, str):
            if any(x is None for x in experience):
                print(f"Warning: Experience contains None values: {experience}")
                return
                
        idx = self._next_idx
        super().add(experience)
        self._it_sum[idx] = self.max_priority ** self.alpha
        self._it_min[idx] = self.max_priority ** self.alpha
        self.size = min(self.size + 1, self._maxsize)  # For compatibility with existing code

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], np.ndarray]:
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        experiences: List
            batch of experiences
        indices: List
            indices of sampled experiences
        weights: np.array
            importance weights
        """
        if len(self._storage) == 0:
            raise ValueError("Cannot sample from an empty buffer")
            
        actual_batch_size = min(batch_size, len(self._storage))
        idxes = self._sample_proportional(actual_batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = self._encode_sample(idxes)
        
        # Final check to ensure no None values
        for i in range(len(experiences)):
            if experiences[i] is None:
                # If we still have a None, replace it with a random valid experience
                random_idx = random.randint(0, len(self._storage) - 1)
                experiences[i] = self._storage[random_idx]
                idxes[i] = random_idx
                
        return experiences, idxes, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities of sampled transitions.

        sets priority of transition at index indices[i] in buffer
        to priorities[i].

        Parameters
        ----------
        indices: [int]
            List of indices of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled indices denoted by
            variable `indices`.
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive
            priority = max(priority, self.epsilon)
            
            # Update max priority if needed
            self.max_priority = max(self.max_priority, priority)
            
            # Check if index is valid
            if 0 <= idx < len(self._storage):
                self._it_sum[idx] = priority ** self.alpha
                self._it_min[idx] = priority ** self.alpha 

def create_replay_buffer(size: int, use_per: bool = False, alpha: float = 0.6, beta: float = 0.4, 
                        beta_increment: float = 0.001, epsilon: float = 0.01):
    """
    Factory function to create either a regular or prioritized replay buffer.
    
    Parameters
    ----------
    size: int
        Size of the replay buffer
    use_per: bool
        Whether to use prioritized experience replay
    alpha: float
        Prioritization exponent (0 = uniform, 1 = full prioritization)
    beta: float
        Initial importance sampling correction (0 = no correction, 1 = full correction)
    beta_increment: float
        Amount to increase beta each time we sample
    epsilon: float
        Small positive value to ensure non-zero priority
        
    Returns
    -------
    ReplayBuffer or PrioritizedReplayBuffer
        The created replay buffer
    """
    if use_per:
        return PrioritizedReplayBuffer(
            capacity=size,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment,
            epsilon=epsilon
        )
    else:
        return ReplayBuffer(size=size) 