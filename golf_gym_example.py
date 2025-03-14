import numpy as np
import time
from golf_gym_env import GolfGymEnv
from golf_game_v2 import GameConfig

def main():
    """
    Example of using the GolfGymEnv with a simple random agent.
    """
    # Create a custom configuration if needed
    config = GameConfig(
        num_players=2,
        grid_rows=2,
        grid_cols=3,
        initial_revealed=2,
        max_turns=100,
        normalize_rewards=True
    )
    
    # Create the environment with render_mode
    env = GolfGymEnv(config, render_mode='human')
    
    # Run a few episodes
    num_episodes = 3
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n===== Episode {episode+1} =====")
        
        # Reset returns tuple of (observation, info) in Gymnasium
        obs, info = env.reset(seed=episode)  # Set different seed for each episode
        
        episode_reward = 0
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            # Render the environment (optional)
            if step_count % 5 == 0:  # Render every 5 steps to avoid too much output
                print(f"\n--- Step {step_count} ---")
                env.render()
            
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Random agent: select random valid action
            action = np.random.choice(valid_actions)
            
            # Take a step in the environment
            # In Gymnasium, step returns (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print step information
            print(f"Action: {action}, Reward: {reward:.4f}")
            
            # Update episode reward
            episode_reward += reward
            step_count += 1
            
            # Small delay to make output readable
            time.sleep(0.1)
        
        # Final render
        print(f"\n--- Final state ---")
        env.render()
        
        # Print episode summary
        print(f"\nEpisode {episode+1} finished after {step_count} steps")
        print(f"Total reward: {episode_reward:.4f}")
        
        if "scores" in info:
            print(f"Final scores: {info['scores']}")
            
        # Report if episode was terminated or truncated
        if terminated:
            print("Episode terminated (game completed normally)")
        elif truncated:
            print("Episode truncated (max turns reached)")
        
        total_rewards.append(episode_reward)
    
    # Print summary
    print("\n===== Summary =====")
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.4f}")

if __name__ == "__main__":
    main() 