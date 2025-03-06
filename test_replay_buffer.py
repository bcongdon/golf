import pytest
import numpy as np
from replay_buffer import SumTree, PrioritizedReplayBuffer


def test_sum_tree_initialization():
    capacity = 4
    tree = SumTree(capacity)
    assert len(tree.tree) == 2 * capacity - 1
    assert tree.capacity == capacity
    assert len(tree.valid_indices) == 0
    assert np.all(tree.tree == 0)


def test_sum_tree_update():
    capacity = 4
    tree = SumTree(capacity)
    
    # Update a leaf node
    tree.update(3, 1.0)  # idx 3 is first leaf
    assert tree.tree[3] == 1.0
    assert tree.tree[1] == 1.0  # parent
    assert tree.tree[0] == 1.0  # root
    assert 0 in tree.valid_indices  # buffer index 0
    
    # Update another leaf node
    tree.update(4, 2.0)  # idx 4 is second leaf
    assert tree.tree[4] == 2.0
    assert tree.tree[1] == 3.0  # left parent (sum of children)
    assert tree.tree[2] == 0.0  # right parent
    assert tree.tree[0] == 3.0  # root
    assert 1 in tree.valid_indices  # buffer index 1


def test_sum_tree_get_leaf():
    capacity = 4
    tree = SumTree(capacity)
    
    # Set up tree with known values
    tree.update(3, 1.0)  # buffer idx 0
    tree.update(4, 2.0)  # buffer idx 1
    tree.update(5, 3.0)  # buffer idx 2
    
    # Test retrieving leaves
    idx, buffer_idx, priority = tree.get_leaf(0.5)  # Should get first leaf (1.0)
    assert buffer_idx == 0
    assert priority == 1.0
    
    idx, buffer_idx, priority = tree.get_leaf(4.5)  # Should get third leaf (3.0)
    assert buffer_idx == 2
    assert priority == 3.0


def test_prioritized_buffer_initialization():
    capacity = 100
    buffer = PrioritizedReplayBuffer(capacity)
    assert buffer.capacity == capacity
    assert len(buffer) == 0
    assert buffer.position == 0
    assert buffer.max_priority == 1.0


def test_prioritized_buffer_add():
    buffer = PrioritizedReplayBuffer(capacity=3)
    
    # Add experiences
    buffer.add("exp1")
    assert len(buffer) == 1
    assert buffer.buffer[0] == "exp1"
    
    buffer.add("exp2")
    assert len(buffer) == 2
    assert buffer.buffer[1] == "exp2"
    
    # Test wrapping around
    buffer.add("exp3")
    buffer.add("exp4")
    assert len(buffer) == 3
    assert buffer.buffer[0] == "exp4"


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
    assert all(0 <= idx < buffer.capacity for idx in indices)
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
    min_priority = buffer.epsilon ** buffer.alpha
    actual_priority = buffer.tree.tree[buffer.capacity - 1 + indices[0]]
    # Use np.isclose for floating point comparison
    assert np.isclose(actual_priority, min_priority, rtol=1e-7) or actual_priority > min_priority, \
        f"Priority {actual_priority} should be >= {min_priority}"
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
    assert all(0 <= idx < buffer.capacity for idx in indices)
    assert all(0 <= w <= 1 for w in weights)
