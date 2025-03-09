import ray
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from golf_pettingzoo_env import env as golf_env
from golf_game_v2 import GameConfig

import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained RLLib agent on the Golf environment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file to load')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--evaluation-seed', type=int, default=42,
                        help='Random seed for evaluation')
    return parser.parse_args()


def env_creator(config):
    """Create the Golf PettingZoo environment with given config."""
    game_config = GameConfig(
        num_players=config.get("num_players", 2),
        grid_rows=config.get("grid_rows", 2),
        grid_cols=config.get("grid_cols", 3)
    )
    render_mode = "human" if config.get("render", False) else None
    return golf_env(config=game_config, render_mode=render_mode)


def test_trained_agent(checkpoint_path, num_episodes=10, render=False, seed=42):
    """Test a trained agent from a checkpoint."""
    # Initialize Ray
    ray.init()
    
    # Register the environment
    env_name = "golf_v0"
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))
    
    # Create environment config
    env_config = {
        "num_players": 2,
        "grid_rows": 2,
        "grid_cols": 3,
        "render": render
    }
    
    # Load the trained agent
    config = PPOConfig()
    config = config.environment(env=env_name)
    config = config.environment(env_config=env_config)
    config = config.framework(framework="torch")
    
    # Set up multi-agent config
    config = config.multi_agent(
        policies={"shared_policy": (None, None, None, {})},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    )
    
    # Create the algorithm from checkpoint
    algo = config.build(checkpoint_path=checkpoint_path)
    
    # Create the environment
    env = PettingZooEnv(env_creator(env_config))
    
    # Stats tracking
    episode_rewards = []
    episode_lengths = []
    win_rate = 0
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode+1}/{num_episodes} ===")
        env.reset(seed=seed + episode)
        
        done = {a: False for a in env.env.agents}
        episode_reward = {a: 0 for a in env.env.agents}
        step_count = 0
        
        # Get initial observations
        observations = {agent: env.env.observe(agent) for agent in env.env.agents}
        
        while not all(done.values()):
            current_agent = env.env.agent_selection
            
            # Skip if the agent is done
            if done[current_agent]:
                env.env.step(None)  # Skip the agent's turn
                continue
                
            # Get action from policy
            action = algo.compute_single_action(
                observations[current_agent], 
                policy_id="shared_policy"
            )
            
            # Print action information
            valid_actions = env.env.infos[current_agent].get("valid_actions", [])
            action_str = f"Action: {action}"
            if valid_actions:
                action_str += f" (Valid actions: {valid_actions})"
            print(f"Agent {current_agent}: {action_str}")
            
            # Take the action in the environment
            env.env.step(action)
            
            # Render if requested
            if render:
                env.env.render()
                
            # Update observations and rewards
            next_observations = {agent: env.env.observe(agent) for agent in env.env.agents}
            for agent in env.env.agents:
                if agent == current_agent:
                    episode_reward[agent] += env.env.rewards[agent]
                    
            # Update done flags
            done = {agent: env.env.terminations[agent] or env.env.truncations[agent] 
                   for agent in env.env.agents}
            
            # Update observations for the next step
            observations = next_observations
            step_count += 1
        
        # Collect episode statistics
        episode_lengths.append(step_count)
        rewards = {agent: episode_reward.get(agent, 0) for agent in env.env.possible_agents}
        episode_rewards.append(rewards)
        
        # Determine if player_0 (our trained agent) won
        scores = [info.get("score", float('inf')) for agent, info in env.env.infos.items()]
        if len(scores) >= 2 and scores[0] < scores[1]:
            win_rate += 1
        
        # Print episode results
        print(f"Episode {episode+1} finished after {step_count} steps")
        print(f"Rewards: {rewards}")
        print(f"Scores: {scores}")
    
    # Print overall statistics
    print("\n=== Overall Statistics ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
    
    # Calculate and print mean rewards per agent
    mean_rewards = {}
    for agent in env.env.possible_agents:
        agent_rewards = [ep_reward.get(agent, 0) for ep_reward in episode_rewards]
        mean_rewards[agent] = np.mean(agent_rewards)
    print(f"Mean rewards: {mean_rewards}")
    
    # Print win rate
    win_rate_percentage = (win_rate / num_episodes) * 100
    print(f"Win rate: {win_rate}/{num_episodes} ({win_rate_percentage:.2f}%)")
    
    # Clean up
    env.close()
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    test_trained_agent(
        args.checkpoint, 
        num_episodes=args.num_episodes, 
        render=args.render,
        seed=args.evaluation_seed
    ) 