import unittest
import numpy as np
from replay_buffer import PrioritizedReplayBuffer

class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Tests for the PrioritizedReplayBuffer class."""
    
    def test_initialization(self):
        """Test that the buffer initializes correctly."""
        capacity = 100
        buffer = PrioritizedReplayBuffer(capacity)
        
        self.assertEqual(buffer.capacity, capacity)
        self.assertEqual(buffer.size, 0)
        self.assertEqual(len(buffer.buffer), capacity)
        self.assertEqual(buffer.max_priority, 1.0)
    
    def test_add(self):
        """Test adding experiences to the buffer."""
        capacity = 10
        buffer = PrioritizedReplayBuffer(capacity)
        
        # Add some experiences
        for i in range(5):
            buffer.add(f"experience_{i}")
        
        # Check size
        self.assertEqual(buffer.size, 5)
        
        # Check buffer contents
        for i in range(5):
            self.assertEqual(buffer.buffer[i], f"experience_{i}")
        
        # Check valid indices
        self.assertEqual(buffer.valid_indices, set(range(5)))
        
        # Check priorities
        for i in range(5):
            self.assertGreater(buffer.priorities[i], 0)
    
    def test_sample_empty(self):
        """Test sampling from an empty buffer."""
        buffer = PrioritizedReplayBuffer(10)
        
        # This should not raise an exception
        experiences, indices, weights = buffer.sample(3)
        
        # But it should return empty lists
        self.assertEqual(len(experiences), 3)  # Will be filled with random values
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(weights), 3)
    
    def test_sample(self):
        """Test sampling from the buffer."""
        capacity = 10
        buffer = PrioritizedReplayBuffer(capacity)
        
        # Add some experiences
        for i in range(capacity):
            buffer.add(f"experience_{i}")
        
        # Sample from buffer
        batch_size = 5
        experiences, indices, weights = buffer.sample(batch_size)
        
        # Check that we got the expected batch size
        self.assertEqual(len(experiences), batch_size)
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(len(weights), batch_size)
        
        # Check that the experiences match what's in the buffer
        for i, exp in enumerate(experiences):
            self.assertEqual(exp, buffer.buffer[indices[i]])
    
    def test_update_priorities(self):
        """Test updating priorities."""
        capacity = 10
        buffer = PrioritizedReplayBuffer(capacity)
        
        # Add some experiences
        for i in range(capacity):
            buffer.add(f"experience_{i}")
        
        # Sample from buffer
        batch_size = 5
        experiences, indices, _ = buffer.sample(batch_size)
        
        # Update priorities for sampled indices
        priorities = np.array([10.0, 5.0, 1.0, 0.5, 0.1])
        buffer.update_priorities(indices, priorities)
        
        # Check that priorities were updated
        for idx, priority in zip(indices, priorities):
            self.assertEqual(buffer.priorities[idx], priority ** buffer.alpha)
        
        # Sample many times to verify distribution matches priorities
        samples = 1000
        counts = np.zeros(capacity)
        
        for _ in range(samples):
            _, new_indices, _ = buffer.sample(1)
            counts[new_indices[0]] += 1
        
        # Check that indices with higher priorities are sampled more often
        # This is a statistical test, so it might fail occasionally
        # We'll just check that the indices we updated have higher counts
        # than the ones we didn't update
        updated_counts = np.sum(counts[indices])
        not_updated_counts = np.sum(counts) - updated_counts
        
        # Updated indices should be sampled more than not updated ones
        # since we gave them much higher priorities
        self.assertGreater(updated_counts / len(indices), 
                          not_updated_counts / (capacity - len(indices)))
    
    def test_zero_priority_handling(self):
        """Test handling of zero or very small total priorities."""
        capacity = 10
        
        buffer = PrioritizedReplayBuffer(capacity)
        
        # Add some experiences
        for i in range(5):
            buffer.add(f"experience_{i}")
        
        # Manually set all priorities to zero
        # This simulates a case where all priorities might become very small
        for i in range(buffer.size):
            buffer.priorities[i] = 0
        
        # Verify total priority is very small
        valid_priorities = np.array([buffer.priorities[i] for i in buffer.valid_indices])
        self.assertLessEqual(np.sum(valid_priorities), 1e-4)
        
        # Sample from buffer - this should use uniform sampling as a fallback
        batch_size = 3
        experiences, indices, weights = buffer.sample(batch_size)
        
        # Check that we still get the expected batch size
        self.assertEqual(len(experiences), batch_size)
    
    def test_capacity_overflow(self):
        """Test that the buffer correctly handles overflow."""
        capacity = 5
        buffer = PrioritizedReplayBuffer(capacity)
        
        # Add more experiences than capacity
        for i in range(capacity * 2):
            buffer.add(f"experience_{i}")
        
        # Check size
        self.assertEqual(buffer.size, capacity)
        
        # Check that only the most recent experiences are kept
        for i in range(capacity):
            expected_idx = (i + capacity) % capacity
            expected_exp = f"experience_{i + capacity}"
            self.assertEqual(buffer.buffer[expected_idx], expected_exp)
    
    def test_multiple_overflows(self):
        """Test that priorities are correctly updated after multiple buffer overflows."""
        capacity = 3  # Very small capacity to force multiple overflows
        buffer = PrioritizedReplayBuffer(capacity)
        
        # Fill the buffer with initial experiences
        for i in range(capacity):
            buffer.add(f"experience_{i}")
        
        # Sample all experiences
        experiences, indices, _ = buffer.sample(capacity)
        print(f"Initial experiences: {experiences}")
        print(f"Initial indices: {indices}")
        
        # Set very different priorities to make the test more robust
        # Higher priority means more frequent sampling
        # Index 0 gets high priority, index 1 gets low priority, index 2 gets medium priority
        priorities = np.array([10.0, 0.1, 1.0])
        buffer.update_priorities(indices, priorities)
        
        # Print the priorities
        print(f"Priorities after setting: {[buffer.priorities[i] for i in range(capacity)]}")
        print(f"Valid indices: {buffer.valid_indices}")
        
        # Sample many times to verify distribution matches priorities
        samples = 1000
        counts = np.zeros(capacity)
        
        for _ in range(samples):
            _, sampled_indices, _ = buffer.sample(1)
            counts[sampled_indices[0]] += 1
        
        print(f"Counts after setting priorities: {counts}")
        
        # The index with highest priority (0) should be sampled most often
        # In our implementation, higher priority values are sampled more frequently
        self.assertGreater(counts[0], counts[2])
        self.assertGreater(counts[2], counts[1])
        
        # Now overflow the buffer completely
        for i in range(capacity, capacity * 2):
            buffer.add(f"experience_{i}")
        
        # Print buffer contents after overflow
        print(f"Buffer after overflow: {buffer.buffer}")
        
        # Sample all experiences again
        experiences, indices, _ = buffer.sample(capacity)
        print(f"Experiences after overflow: {experiences}")
        print(f"Indices after overflow: {indices}")
        
        # Set the same pattern of priorities
        # Higher priority means more frequent sampling
        priorities = np.array([10.0, 0.1, 1.0])
        buffer.update_priorities(indices, priorities)
        
        # Print the priorities again
        print(f"Priorities after overflow and setting: {[buffer.priorities[i] for i in range(capacity)]}")
        print(f"Valid indices after overflow: {buffer.valid_indices}")
        
        # Sample many times again
        counts = np.zeros(capacity)
        
        for _ in range(samples):
            _, sampled_indices, _ = buffer.sample(1)
            counts[sampled_indices[0]] += 1
        
        print(f"Counts after overflow and setting priorities: {counts}")
        
        # The index with highest priority should still be sampled most often
        # In our implementation, higher priority values are sampled more frequently
        self.assertGreater(counts[0], counts[2])

if __name__ == "__main__":
    unittest.main() 