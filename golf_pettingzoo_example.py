import numpy as np
import time
from golf_pettingzoo_env import env as golf_env
from golf_game_v2 import GameConfig

def main():
    """
    Example of using the GolfPettingZooEnv with random agents.
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
    env = golf_env(config=config, render_mode='human')
    
    # Run a few episodes
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n===== Episode {episode+1} =====")
        
        # Reset the environment
        env.reset(seed=episode)  # Set different seed for each episode
        
        # For tracking episode stats
        step_count = 0
        ep_rewards = {agent: 0 for agent in env.agents}
        
        # Main game loop
        for agent in env.agent_iter():
            # Only render occasionally to avoid too much output
            if step_count % 5 == 0:
                print(f"\n--- Step {step_count} ---")
                print(f"Current agent: {agent}")
                env.render()
            
            # Get observation and info for current agent
            obs, reward, termination, truncation, info = env.last()
            
            # Update cumulative rewards
            ep_rewards[agent] += reward
            
            # Check if the episode is over
            done = termination or truncation
            if done:
                break
                
            # Get valid actions for the current agent
            valid_actions = info.get("valid_actions", [])
            if not valid_actions:
                # Fallback if valid_actions not in info
                valid_actions = env.get_valid_actions(agent)
            
            # Random agent: select random valid action
            if valid_actions:
                action = np.random.choice(valid_actions)
            else:
                # If somehow no valid actions, pass a default action that will be checked by the env
                action = 0
            
            # Take a step in the environment
            env.step(action)
            
            # Print action information
            print(f"Agent {agent} took action {action}, received reward: {reward:.4f}")
            
            # Increment step counter
            step_count += 1
            
            # Small delay to make output readable
            time.sleep(0.1)
        
        # Final render
        print(f"\n--- Final state ---")
        env.render()
        
        # Get final rewards and scores from last info
        final_infos = {agent: env.infos[agent] for agent in env.agents}
        scores = {
            agent: info.get("score", "Unknown") 
            for agent, info in final_infos.items()
        }
        
        # Print episode summary
        print(f"\nEpisode {episode+1} finished after {step_count} steps")
        print(f"Final scores: {scores}")
        print(f"Episode rewards: {ep_rewards}")
        
        # Determine winner
        min_score = float('inf')
        winner = None
        
        for agent, score in scores.items():
            if score != "Unknown" and score < min_score:
                min_score = score
                winner = agent
                
        if winner:
            print(f"Winner: {winner} with score {min_score}")
        else:
            print("No winner determined")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main() 