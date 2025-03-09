import pytest
import numpy as np
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, create_replay_buffer
from segment_tree import SumSegmentTree, MinSegmentTree


def test_replay_buffer_initialization():
    capacity = 100
    buffer = ReplayBuffer(size=capacity)
    assert buffer._maxsize == capacity
    assert len(buffer) == 0
    assert buffer._next_idx == 0


def test_replay_buffer_add():
    buffer = ReplayBuffer(size=3)
    
    # Add experiences
    buffer.add("exp1")
    assert len(buffer) == 1
    assert buffer._storage[0] == "exp1"
    
    buffer.add("exp2")
    assert len(buffer) == 2
    assert buffer._storage[1] == "exp2"
    
    # Test wrapping around
    buffer.add("exp3")
    buffer.add("exp4")
    assert len(buffer) == 3
    assert buffer._storage[0] == "exp4"


def test_replay_buffer_sample():
    buffer = ReplayBuffer(size=4)
    
    # Add some experiences
    experiences = ["exp1", "exp2", "exp3", "exp4"]
    for exp in experiences:
        buffer.add(exp)
    
    # Sample experiences
    batch_size = 2
    sampled_exp, indices, weights = buffer.sample(batch_size)
    
    assert len(sampled_exp) == batch_size
    assert len(indices) == batch_size
    assert len(weights) == batch_size
    assert all(exp in experiences for exp in sampled_exp)
    assert all(0 <= idx < len(buffer) for idx in indices)
    assert all(w == 1.0 for w in weights)  # Regular replay buffer uses uniform weights


def test_prioritized_buffer_initialization():
    capacity = 100
    buffer = PrioritizedReplayBuffer(capacity)
    assert buffer._maxsize == capacity
    assert len(buffer) == 0
    assert buffer._next_idx == 0
    assert buffer.max_priority == 1.0


def test_prioritized_buffer_add():
    buffer = PrioritizedReplayBuffer(capacity=3)
    
    # Add experiences
    buffer.add("exp1")
    assert len(buffer) == 1
    assert buffer._storage[0] == "exp1"
    
    buffer.add("exp2")
    assert len(buffer) == 2
    assert buffer._storage[1] == "exp2"
    
    # Test wrapping around
    buffer.add("exp3")
    buffer.add("exp4")
    assert len(buffer) == 3
    assert buffer._storage[0] == "exp4"


def test_prioritized_buffer_sample():
    buffer = PrioritizedReplayBuffer(capacity=4)
    
    # Add some experiences
    experiences = ["exp1", "exp2", "exp3", "exp4"]
    for exp in experiences:
        buffer.add(exp)
    
    # Sample experiences
    batch_size = 2
    sampled_exp, indices, weights = buffer.sample(batch_size)
    
    assert len(sampled_exp) == batch_size
    assert len(indices) == batch_size
    assert len(weights) == batch_size
    assert all(exp in experiences for exp in sampled_exp)
    assert all(0 <= idx < len(buffer) for idx in indices)
    assert all(0 <= w <= 1 for w in weights)


def test_prioritized_buffer_update_priorities():
    buffer = PrioritizedReplayBuffer(capacity=4)
    
    # Add experiences
    for i in range(4):
        buffer.add(f"exp{i}")
    
    # Sample and update priorities
    sampled_exp, indices, _ = buffer.sample(2)
    new_priorities = [0.5, 1.0]
    buffer.update_priorities(indices, new_priorities)
    
    # Sample again to verify priority changes
    _, new_indices, new_weights = buffer.sample(4)
    assert len(new_indices) == 4
    assert len(new_weights) == 4


def test_empty_buffer_sampling():
    buffer = PrioritizedReplayBuffer(capacity=4)
    
    # Sample from empty buffer should raise ValueError
    try:
        buffer.sample(2)
        assert False, "Expected ValueError when sampling from empty buffer"
    except ValueError:
        pass  # Expected behavior


def test_prioritized_buffer_beta_annealing():
    buffer = PrioritizedReplayBuffer(capacity=4, beta=0.4, beta_increment=0.1)
    
    # Add experiences
    for i in range(4):
        buffer.add(f"exp{i}")
    
    initial_beta = buffer.beta
    
    # Sample multiple times to test beta annealing
    for _ in range(3):
        buffer.sample(2)
    
    assert buffer.beta > initial_beta
    assert buffer.beta <= 1.0


def test_prioritized_buffer_priority_bounds():
    buffer = PrioritizedReplayBuffer(capacity=4, epsilon=0.01)
    
    # Add experiences
    for i in range(4):
        buffer.add(f"exp{i}")
    
    # Update with very small and large priorities
    indices = [0, 1]
    priorities = [0.0, 100.0]  # Should be bounded by epsilon and max_priority
    buffer.update_priorities(indices, priorities)
    
    # The actual priorities should be bounded
    assert buffer.max_priority >= 100.0


def test_prioritized_buffer_partial_sampling():
    """Test sampling when buffer is not full and during filling."""
    buffer = PrioritizedReplayBuffer(capacity=100)
    
    # Try sampling when empty - should raise ValueError
    try:
        buffer.sample(32)
        assert False, "Expected ValueError when sampling from empty buffer"
    except ValueError:
        pass  # Expected behavior
    
    # Add just a few experiences
    for i in range(5):
        buffer.add(f"exp{i}")
    
    # Sample with batch_size larger than buffer size
    # Should only return the available experiences (5)
    experiences, indices, weights = buffer.sample(32)
    assert len(experiences) == 5  # Only 5 experiences available
    assert len(indices) == 5
    assert len(weights) == 5
    assert all(exp is not None for exp in experiences)  # All should be valid
    assert all(isinstance(exp, str) for exp in experiences)  # All should be strings
    
    # Sample with batch_size smaller than buffer size
    experiences, indices, weights = buffer.sample(3)
    assert len(experiences) == 3
    assert len(indices) == 3
    assert len(weights) == 3
    assert all(exp is not None for exp in experiences)  # All should be valid
    assert all(isinstance(exp, str) for exp in experiences)  # All should be strings
    assert all(0 <= idx < len(buffer) for idx in indices)
    assert all(0 <= w <= 1 for w in weights)


def test_no_none_values_in_sample():
    """Test that sample never returns None values in the experiences list."""
    buffer = PrioritizedReplayBuffer(capacity=100)
    
    # Add some experiences
    for i in range(10):
        buffer.add(f"exp{i}")
    
    # Sample with different batch sizes
    for batch_size in [5, 10, 15]:
        experiences, indices, weights = buffer.sample(batch_size)
        
        # Check that we get min(batch_size, buffer.size) experiences
        expected_size = min(batch_size, len(buffer))
        assert len(experiences) == expected_size
        assert len(indices) == expected_size
        assert len(weights) == expected_size
        
        # Check that all experiences are valid (not None)
        assert all(exp is not None for exp in experiences)
        assert all(isinstance(exp, str) for exp in experiences)


def test_none_values_in_add():
    """Test that the buffer handles None values correctly in the add method."""
    buffer = PrioritizedReplayBuffer(capacity=10)
    
    # Add a None experience
    buffer.add(None)
    
    # Add some valid experiences
    for i in range(5):
        buffer.add(f"exp{i}")
    
    # Sample from the buffer
    experiences, indices, weights = buffer.sample(3)
    
    # Check that all experiences are valid (not None)
    assert len(experiences) == 3
    assert all(exp is not None for exp in experiences)
    assert all(isinstance(exp, str) for exp in experiences)


def test_create_replay_buffer():
    """Test that create_replay_buffer creates the correct type of buffer."""
    # Test creating a regular replay buffer
    buffer = create_replay_buffer(size=100, use_per=False)
    assert isinstance(buffer, ReplayBuffer)
    assert not isinstance(buffer, PrioritizedReplayBuffer)
    assert buffer._maxsize == 100
    
    # Test creating a prioritized replay buffer
    buffer = create_replay_buffer(size=100, use_per=True, alpha=0.8, beta=0.6, beta_increment=0.002)
    assert isinstance(buffer, PrioritizedReplayBuffer)
    assert buffer._maxsize == 100
    assert buffer.alpha == 0.8
    assert buffer.beta == 0.6
    assert buffer.beta_increment == 0.002
    
    # Test default parameters
    buffer = create_replay_buffer(size=100, use_per=True)
    assert buffer.alpha == 0.6
    assert buffer.beta == 0.4
    assert buffer.beta_increment == 0.001
    assert buffer.epsilon == 0.01
