import os
import argparse
import numpy as np
import time
import torch
from stable_baselines3 import PPO
from train_ppo_lstm import make_env, LSTMActorCriticPolicy, GolfLSTMFeatureExtractor
from golf_pettingzoo_env import env as golf_env
from golf_game_v2 import GameConfig

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Play Golf against a trained AI')
    
    # Environment parameters
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    parser.add_argument('--grid_rows', type=int, default=2, help='Number of rows in card grid')
    parser.add_argument('--grid_cols', type=int, default=3, help='Number of columns in card grid')
    parser.add_argument('--initial_revealed', type=int, default=2, help='Initial revealed cards per player')
    parser.add_argument('--max_turns', type=int, default=200, help='Maximum number of turns')
    parser.add_argument('--normalize_rewards', action='store_true', help='Normalize rewards')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    
    # Play parameters
    parser.add_argument('--num_games', type=int, default=5, help='Number of games to play')
    parser.add_argument('--human_player', action='store_true', help='Enable human player mode')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between moves (seconds)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def human_player_action(env, valid_actions):
    """Get action from human player."""
    print("\nValid actions:")
    action_names = {
        0: "Draw from deck",
        1: "Draw from discard",
        2: "Replace card at position 0",
        3: "Replace card at position 1",
        4: "Replace card at position 2",
        5: "Replace card at position 3",
        6: "Replace card at position 4",
        7: "Replace card at position 5",
        8: "Discard drawn card"
    }
    
    for action in valid_actions:
        print(f"  {action}: {action_names.get(action, f'Unknown action {action}')}")
    
    while True:
        try:
            action = int(input("\nEnter your action: "))
            if action in valid_actions:
                return action
            else:
                print(f"Invalid action! Please choose from: {valid_actions}")
        except ValueError:
            print("Please enter a number.")

def ai_vs_random(args):
    """Play trained AI agent against random opponent."""
    # Create the environment
    config = GameConfig(
        num_players=args.num_players,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        initial_revealed=args.initial_revealed,
        max_turns=args.max_turns,
        normalize_rewards=args.normalize_rewards
    )
    
    try:
        # Use the AEC environment directly with render_mode for human play
        env = golf_env(config=config, render_mode='human')
        print("Environment created successfully!")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    # Load the model
    try:
        model = PPO.load(args.model_path, policy=LSTMActorCriticPolicy)
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize statistics
    wins = 0
    losses = 0
    ties = 0
    total_rewards = 0
    
    for game in range(args.num_games):
        print(f"\n===== Game {game+1}/{args.num_games} =====")
        
        # Reset environment
        env.reset(seed=args.seed + game)
        
        # Reset LSTM states
        if hasattr(model.policy, 'reset_lstm_states'):
            model.policy.reset_lstm_states()
        
        # Game loop
        step = 0
        player0_reward = 0
        
        for agent in env.agent_iter():
            # Get current observation, reward, etc.
            obs, reward, termination, truncation, info = env.last()
            
            # Track reward for player 0
            if agent == "player_0":
                player0_reward += reward
            
            # Check if game is over
            done = termination or truncation
            if done:
                break
            
            # Print current state
            print(f"\n--- Step {step} ---")
            print(f"Current player: {agent}")
            env.render()
            
            # Get valid actions
            valid_actions = info.get("valid_actions", [])
            if not valid_actions:
                valid_actions = env.get_valid_actions(agent)
            
            # Get action based on player type
            if agent == "player_0":  # AI player
                if args.human_player:
                    action = human_player_action(env, valid_actions)
                    agent_type = "Human"
                else:
                    # Get action from trained model
                    # Reset LSTM states if needed
                    action, _ = model.predict(np.array(obs), deterministic=True)
                    
                    # Ensure valid action
                    if action not in valid_actions:
                        # Create mask for valid actions
                        mask = np.zeros(env.action_spaces[agent].n)
                        mask[valid_actions] = 1
                        
                        # Get action probabilities
                        action_dist = model.policy.get_distribution(np.array(obs).reshape(1, -1))
                        
                        # Apply mask
                        masked_probs = action_dist.distribution.probs.numpy() * mask
                        masked_probs = masked_probs / np.sum(masked_probs)
                        
                        # Sample new action
                        action = np.random.choice(np.arange(len(masked_probs)), p=masked_probs)
                    
                    agent_type = "AI"
            else:  # Random opponent
                action = np.random.choice(valid_actions)
                agent_type = "Random"
            
            # Take step
            env.step(action)
            
            # Print action
            print(f"{agent_type} player {agent} took action {action}")
            
            # Delay between steps for readability
            time.sleep(args.delay)
            step += 1
        
        # Get final scores
        final_infos = {agent: env.infos[agent] for agent in env.agents}
        scores = {
            agent: info.get("score", float('inf')) 
            for agent, info in final_infos.items()
        }
        
        # Determine winner
        player0_score = scores.get("player_0", float('inf'))
        player1_score = scores.get("player_1", float('inf'))
        
        print(f"\n--- Game Results ---")
        print(f"Player 0 (AI) score: {player0_score}")
        print(f"Player 1 (Random) score: {player1_score}")
        print(f"Player 0 (AI) reward: {player0_reward:.2f}")
        
        # Update statistics
        if player0_score < player1_score:
            wins += 1
            result = "Won"
        elif player0_score > player1_score:
            losses += 1
            result = "Lost"
        else:
            ties += 1
            result = "Tied"
        
        print(f"Result: AI {result}\n")
        total_rewards += player0_reward
    
    # Print summary
    print(f"\n===== Summary =====")
    print(f"Games played: {args.num_games}")
    print(f"Wins: {wins} ({(wins/args.num_games)*100:.1f}%)")
    print(f"Losses: {losses} ({(losses/args.num_games)*100:.1f}%)")
    print(f"Ties: {ties} ({(ties/args.num_games)*100:.1f}%)")
    print(f"Average reward: {total_rewards/args.num_games:.2f}")
    
    # Close environment
    env.close()

def main():
    """Main entry point."""
    args = parse_args()
    ai_vs_random(args)

if __name__ == "__main__":
    main() 