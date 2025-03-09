import argparse
import os
import numpy as np
import json
import time
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Ray and RLlib imports
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

# PettingZoo environment
from golf_pettingzoo_env import env as golf_env
from golf_game_v2 import GameConfig

# Evaluation and monitoring utilities
from ray.rllib.policy.policy import Policy
from ray.tune.logger import pretty_print


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a Golf agent using Ray RLlib')
    
    # Training parameters
    parser.add_argument('--algorithm', type=str, default='PPO', 
                        choices=['PPO', 'DQN', 'IMPALA'],
                        help='RL algorithm to use')
    parser.add_argument('--num-gpus', type=float, default=0, 
                        help='Number of GPUs to use (can be fractional)')
    parser.add_argument('--train-batch-size', type=int, default=4000, 
                        help='Size of the training batch')
    parser.add_argument('--total-timesteps', type=int, default=5000000, 
                        help='Total timesteps to train for')
    parser.add_argument('--lr', type=float, default=5e-5, 
                        help='Learning rate')
                        
    # Environment parameters
    parser.add_argument('--num-players', type=int, default=2, 
                        help='Number of players in the golf game')
    parser.add_argument('--grid-rows', type=int, default=2, 
                        help='Number of rows in the card grid')
    parser.add_argument('--grid-cols', type=int, default=3, 
                        help='Number of columns in the card grid')
    
    # Saving and logging
    parser.add_argument('--checkpoint-freq', type=int, default=10, 
                        help='Checkpoint frequency in iterations')
    parser.add_argument('--save-dir', type=str, default='./ray_results', 
                        help='Directory to save results')
    parser.add_argument('--exp-name', type=str, default=None, 
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-path', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    
    # Evaluation
    parser.add_argument('--evaluate-episodes', type=int, default=20, 
                        help='Number of episodes to evaluate on')
    parser.add_argument('--evaluate-interval', type=int, default=10, 
                        help='Evaluation interval in iterations')
    
    return parser.parse_args()


def env_creator(config):
    """Create the Golf PettingZoo environment with given config."""
    game_config = GameConfig(
        num_players=config.get("num_players", 2),
        grid_rows=config.get("grid_rows", 2),
        grid_cols=config.get("grid_cols", 3)
    )
    return golf_env(config=game_config, render_mode=config.get("render_mode", None))


def setup_rllib_config(args, env_name):
    """
    Set up the RLlib algorithm configuration based on command line arguments.
    """
    # Common configuration items
    if args.algorithm == "PPO":
        config = PPOConfig()
        # Configure the environment
        config = config.environment(env=env_name)
        config = config.environment(env_config={
            "num_players": args.num_players,
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
        })
        
        # Configure resources
        config = config.resources(num_gpus=args.num_gpus)
        
        # Configure framework
        config = config.framework(framework="torch")
        
        # Configure training parameters
        config = config.training(
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            train_batch_size=args.train_batch_size,
            lr=args.lr,
        )
        
        # Configure multi-agent settings
        config = config.multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        
    elif args.algorithm == "DQN":
        config = DQNConfig()
        # Configure the environment
        config = config.environment(env=env_name)
        config = config.environment(env_config={
            "num_players": args.num_players,
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
        })
        
        # Configure resources
        config = config.resources(num_gpus=args.num_gpus)
        
        # Configure framework
        config = config.framework(framework="torch")
        
        # Configure training parameters
        config = config.training(
            learning_starts=1000,
            buffer_size=50000,
            train_batch_size=args.train_batch_size,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=True,
            dueling=True,
            double_q=True,
            lr=args.lr,
        )
        
        # Configure multi-agent settings
        config = config.multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        
    elif args.algorithm == "IMPALA":
        config = ImpalaConfig()
        # Configure the environment
        config = config.environment(env=env_name)
        config = config.environment(env_config={
            "num_players": args.num_players,
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
        })
        
        # Configure resources
        config = config.resources(num_gpus=args.num_gpus)
        
        # Configure framework
        config = config.framework(framework="torch")
        
        # Configure training parameters
        config = config.training(
            gamma=0.99,
            lambda_=0.95,
            train_batch_size=args.train_batch_size,
            lr=args.lr,
        )
        
        # Configure multi-agent settings
        config = config.multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
        
    # Set log level
    config = config.debugging(log_level="WARN")
    
    return config


def create_experiment_name(args):
    """Create a unique experiment name based on arguments and timestamp."""
    if args.exp_name:
        return args.exp_name
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"golf_{args.algorithm}_{args.num_players}p_{timestamp}"


def plot_training_results(result_dir, metrics=["episode_reward_mean", "episode_len_mean"]):
    """Plot training results from a Ray Tune experiment."""
    # Find the progress file
    progress_file = list(Path(result_dir).glob("**/progress.csv"))
    if not progress_file:
        print(f"No progress file found in {result_dir}")
        return
    
    # Use the first progress file
    progress_file = progress_file[0]
    
    # Read the data
    data = np.genfromtxt(progress_file, delimiter=',', names=True, dtype=None, encoding='utf-8')
    
    # Create a figure for each metric
    for metric in metrics:
        if metric in data.dtype.names:
            plt.figure(figsize=(10, 6))
            plt.plot(data['training_iteration'], data[metric])
            plt.xlabel('Training Iterations')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric} over Training')
            plt.grid(True)
            
            # Save the figure
            plot_path = os.path.join(os.path.dirname(progress_file), f"{metric}_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved {metric} plot to {plot_path}")


def evaluate_policy(algo, env_creator, config, num_episodes=20):
    """Evaluate a trained policy in the environment."""
    env = env_creator(config)
    env.reset(seed=42)
    
    rewards = {agent: [] for agent in env.possible_agents}
    episode_lengths = []
    
    for _ in range(num_episodes):
        env.reset()
        done = {a: False for a in env.agents}
        episode_reward = {a: 0 for a in env.agents}
        step_count = 0
        
        # Get the first observation for each agent
        observations = {agent: env.observe(agent) for agent in env.agents}
        
        while not all(done.values()):
            current_agent = env.agent_selection
            
            # Skip if the agent is done
            if done[current_agent]:
                env.step(None)  # Skip the agent's turn
                continue
                
            # Get action from policy
            action = algo.compute_single_action(
                observations[current_agent], 
                policy_id="shared_policy"
            )
            
            # Take the action in the environment
            env.step(action)
            
            # Update observations and rewards
            next_observations = {agent: env.observe(agent) for agent in env.agents}
            for agent in env.agents:
                if agent == current_agent:
                    episode_reward[agent] += env.rewards[agent]
                    
            # Update done flags
            done = {agent: env.terminations[agent] or env.truncations[agent] for agent in env.agents}
            
            # Update observations for the next step
            observations = next_observations
            step_count += 1
            
        episode_lengths.append(step_count)
        for agent in env.possible_agents:
            rewards[agent].append(episode_reward.get(agent, 0))
    
    # Calculate statistics
    stats = {
        "mean_reward": {agent: np.mean(rewards[agent]) for agent in env.possible_agents},
        "mean_episode_length": np.mean(episode_lengths),
    }
    
    return stats


def main():
    args = parse_args()
    
    # Initialize Ray
    ray.init()
    
    # Register the environment
    env_name = "golf_v0"
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))
    
    # Create experiment name and directory
    exp_name = create_experiment_name(args)
    save_dir = os.path.join(args.save_dir, exp_name)
    
    # Set up RLlib configuration
    config = setup_rllib_config(args, env_name)
    
    # Resume from checkpoint if specified
    if args.resume and args.checkpoint_path:
        config = config.checkpoint_config(resume=True)
        algorithm = config.build(checkpoint_path=args.checkpoint_path)
        print(f"Resuming training from checkpoint: {args.checkpoint_path}")
    else:
        algorithm = config.build()
    
    # Training loop with manual evaluation
    max_iterations = args.total_timesteps // args.train_batch_size
    
    for i in range(1, max_iterations + 1):
        # Train one iteration
        result = algorithm.train()
        
        # Print progress
        print(f"\nIteration {i}/{max_iterations}")
        print(pretty_print(result))
        
        # Save checkpoint
        if i % args.checkpoint_freq == 0:
            checkpoint_dir = algorithm.save()
            print(f"Checkpoint saved to {checkpoint_dir}")
            
        # Periodic evaluation
        if i % args.evaluate_interval == 0:
            print("\nEvaluating policy...")
            eval_stats = evaluate_policy(
                algorithm, 
                env_creator, 
                config.env_config,
                num_episodes=args.evaluate_episodes
            )
            print("Evaluation results:")
            print(json.dumps(eval_stats, indent=4))
    
    # Final checkpoint
    final_checkpoint = algorithm.save()
    print(f"Final checkpoint saved to {final_checkpoint}")
    
    # Generate plots from the results
    plot_training_results(save_dir)
    
    # Clean up
    algorithm.stop()
    ray.shutdown()


if __name__ == "__main__":
    main() 