#!/usr/bin/env python3
"""
Profile the training process to identify bottlenecks.
This script runs a short training session with profiling enabled.
"""

import os
import sys
import argparse
import torch
import time
from datetime import datetime
import cProfile
import pstats
import io
from golf_game import GolfGame
from agent import DQNAgent
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Profile Golf Card Game AI training')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to profile')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the neural network')
    parser.add_argument('--output-dir', type=str, default='logs/profile', help='Directory to save profile results')
    parser.add_argument('--profile-cuda', action='store_true', help='Profile CUDA operations')
    return parser.parse_args()

def profile_environment():
    """Profile the environment step function."""
    print("Profiling environment step function...")
    env = GolfGame(num_players=2)
    state = env.reset()
    
    # Time environment steps
    start_time = time.time()
    num_steps = 1000
    for _ in range(num_steps):
        valid_actions = env._get_valid_actions()
        action = np.random.choice(valid_actions)
        next_state, reward, done, info = env.step(action)
        if done:
            state = env.reset()
        else:
            state = next_state
    end_time = time.time()
    
    print(f"Environment: {num_steps} steps took {end_time - start_time:.4f} seconds")
    print(f"Average time per step: {(end_time - start_time) / num_steps * 1000:.4f} ms")
    return (end_time - start_time) / num_steps

def profile_agent_select_action(agent):
    """Profile the agent's select_action function."""
    print("Profiling agent select_action function...")
    env = GolfGame(num_players=2)
    state = env.reset()
    valid_actions = env._get_valid_actions()
    
    # Time select_action
    start_time = time.time()
    num_actions = 1000
    for _ in range(num_actions):
        action = agent.select_action(state, valid_actions)
    end_time = time.time()
    
    print(f"Agent select_action: {num_actions} actions took {end_time - start_time:.4f} seconds")
    print(f"Average time per action: {(end_time - start_time) / num_actions * 1000:.4f} ms")
    return (end_time - start_time) / num_actions

def profile_agent_learn(agent):
    """Profile the agent's learn function."""
    print("Profiling agent learn function...")
    env = GolfGame(num_players=2)
    
    # Fill replay buffer with some experiences
    state = env.reset()
    for _ in range(1000):
        valid_actions = env._get_valid_actions()
        action = np.random.choice(valid_actions)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        if done:
            state = env.reset()
        else:
            state = next_state
    
    # Time learn function
    start_time = time.time()
    num_learns = 100
    for _ in range(num_learns):
        agent.learn()
    end_time = time.time()
    
    print(f"Agent learn: {num_learns} learn steps took {end_time - start_time:.4f} seconds")
    print(f"Average time per learn step: {(end_time - start_time) / num_learns * 1000:.4f} ms")
    return (end_time - start_time) / num_learns

def profile_cuda_operations(agent):
    """Profile CUDA operations using PyTorch profiler."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA profiling")
        return
    
    print("Profiling CUDA operations...")
    
    # Create dummy inputs
    batch_size = 256
    state_size = agent.state_size
    dummy_states = torch.randn(batch_size, state_size, device=agent.device)
    dummy_actions = torch.randint(0, agent.action_size, (batch_size,), device=agent.device)
    dummy_rewards = torch.randn(batch_size, device=agent.device)
    dummy_next_states = torch.randn(batch_size, state_size, device=agent.device)
    dummy_dones = torch.zeros(batch_size, device=agent.device)
    
    # Profile forward pass
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Forward pass
        q_values = agent.q_network(dummy_states)
        
        # Backward pass
        loss = torch.mean((q_values.gather(1, dummy_actions.unsqueeze(1)).squeeze(1) - dummy_rewards) ** 2)
        loss.backward()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return prof

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize agent
    state_size = 60
    action_size = 9
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size
    )
    
    print(f"Using device: {agent.device}")
    
    # Profile environment
    env_time = profile_environment()
    
    # Profile agent select_action
    select_action_time = profile_agent_select_action(agent)
    
    # Profile agent learn
    learn_time = profile_agent_learn(agent)
    
    # Profile CUDA operations if requested
    if args.profile_cuda and torch.cuda.is_available():
        cuda_prof = profile_cuda_operations(agent)
        cuda_prof_path = os.path.join(args.output_dir, "cuda_profile.txt")
        with open(cuda_prof_path, "w") as f:
            f.write(cuda_prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"CUDA profile saved to {cuda_prof_path}")
    
    # Profile full training loop
    print("\nProfiling full training loop...")
    pr = cProfile.Profile()
    pr.enable()
    
    # Run a short training session
    env = GolfGame(num_players=2)
    
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Agent's turn
            valid_actions = env._get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            # Store experience and learn
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Opponent's turn if not done
            if not done:
                valid_actions = env._get_valid_actions()
                opponent_action = np.random.choice(valid_actions)
                next_state, _, done, _ = env.step(opponent_action)
                if not done:
                    state = next_state
        
        print(f"Episode {episode+1}/{args.episodes}, Reward: {episode_reward:.2f}")
    
    pr.disable()
    
    # Save profiling results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_path = os.path.join(args.output_dir, f"profile_{timestamp}.txt")
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)  # Print top 50 functions by cumulative time
    
    with open(profile_path, 'w') as f:
        f.write(s.getvalue())
    
    # Save summary
    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("Performance Profile Summary\n")
        f.write("=========================\n\n")
        f.write(f"Device: {agent.device}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Hidden size: {args.hidden_size}\n\n")
        
        f.write("Component Times:\n")
        f.write(f"- Environment step: {env_time * 1000:.4f} ms\n")
        f.write(f"- Agent select_action: {select_action_time * 1000:.4f} ms\n")
        f.write(f"- Agent learn: {learn_time * 1000:.4f} ms\n\n")
        
        f.write("Bottleneck Analysis:\n")
        total_time = env_time + select_action_time + learn_time
        f.write(f"- Environment: {env_time / total_time * 100:.1f}%\n")
        f.write(f"- Agent select_action: {select_action_time / total_time * 100:.1f}%\n")
        f.write(f"- Agent learn: {learn_time / total_time * 100:.1f}%\n\n")
        
        f.write(f"Detailed profile saved to: {profile_path}\n")
    
    print(f"\nProfiling complete. Results saved to {profile_path}")
    print(f"Summary saved to {summary_path}")
    
    # Print bottleneck analysis
    print("\nBottleneck Analysis:")
    total_time = env_time + select_action_time + learn_time
    print(f"- Environment: {env_time / total_time * 100:.1f}%")
    print(f"- Agent select_action: {select_action_time / total_time * 100:.1f}%")
    print(f"- Agent learn: {learn_time / total_time * 100:.1f}%")
    
    # Recommendations
    print("\nRecommendations:")
    if env_time / total_time > 0.5:
        print("- The environment is the main bottleneck. Consider optimizing the GolfGame implementation.")
        print("  Try using vectorized operations in NumPy or implementing a batched environment.")
    elif select_action_time / total_time > 0.3:
        print("- The agent's action selection is slow. Consider optimizing the select_action method.")
        print("  Try using batched inference or reducing CPU-GPU transfers.")
    elif learn_time / total_time > 0.5:
        print("- The learning step is the bottleneck. Consider optimizing the learn method.")
        print("  Try using larger batch sizes or reducing the frequency of learning steps.")
    
    if torch.cuda.is_available():
        print("- For CUDA optimization, try using mixed precision training and reducing CPU-GPU transfers.")
        print("  Run with --profile-cuda to get detailed CUDA operation timings.")

if __name__ == "__main__":
    main() 