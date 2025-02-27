import argparse
import os
import numpy as np
import torch
from golf_game import GolfGame
from agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Play Golf against a trained agent')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to the trained model')
    parser.add_argument('--mode', type=str, choices=['play', 'watch'], default='play', 
                        help='Play against the agent or watch the agent play')
    parser.add_argument('--games', type=int, default=1, help='Number of games to play/watch')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size of the neural network')
    return parser.parse_args()

def get_human_action(env):
    """Get action from human player."""
    valid_actions = env._get_valid_actions()
    
    # Display valid actions
    print("\nValid actions:")
    if 0 in valid_actions:
        print("0: Draw from deck")
    if 1 in valid_actions:
        print("1: Take from discard pile")
    
    for action in valid_actions:
        if 2 <= action <= 7:
            print(f"{action}: Replace card at position {action-2}")
    if 8 in valid_actions:
        print("8: Discard drawn card")
    
    # Get input
    while True:
        try:
            action = int(input("\nEnter your action: "))
            if action in valid_actions:
                return action
            else:
                print("Invalid action. Try again.")
        except ValueError:
            print("Please enter a number.")

def play_game(agent, mode='play'):
    """Play a game of Golf against the agent or watch the agent play."""
    env = GolfGame()
    state = env.reset()
    done = False
    
    # Set player roles
    if mode == 'play':
        human_player = 0
        agent_player = 1
    else:  # watch mode
        human_player = None
        agent_player = 0
    
    # Game loop
    while not done:
        env.render()
        
        # Determine whose turn it is
        current_player = env.current_player
        
        if current_player == human_player:
            # Human's turn
            action = get_human_action(env)
        else:
            # Agent's turn
            valid_actions = env._get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            print(f"\nAgent selects action: {action}")
            
            # Add a pause if watching
            if mode == 'watch':
                input("Press Enter to continue...")
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        # Print any errors or info
        if 'error' in info:
            print(f"Error: {info['error']}")
        
        # Announce final round
        if 'final_round' in info and info['final_round']:
            player_name = "You" if info['trigger_player'] == human_player else "Agent"
            print(f"\n{player_name} revealed all cards! Final round begins - each player gets one last turn.")
    
    # Game over
    env.render()
    scores = [env._calculate_score(p) for p in range(env.num_players)]
    print("Game over!")
    print(f"Final scores: {scores}")
    
    winner = np.argmin(scores)
    if winner == human_player:
        print("You win!")
    elif human_player is not None:
        print("Agent wins!")
    else:
        print(f"Player {winner} wins!")
    
    return winner

def main(args):
    # Initialize agent
    state_size = 60  # Simplified observation space (removed suits)
    action_size = 9   # Updated: removed action 8 (knock)
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size
    )
    
    # Load model
    if os.path.exists(args.model):
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print(f"Model file {args.model} not found. Using untrained agent.")
    
    # Play games
    agent_wins = 0
    human_wins = 0
    
    for game in range(args.games):
        print(f"\n===== Game {game+1}/{args.games} =====\n")
        winner = play_game(agent, args.mode)
        
        if args.mode == 'play':
            if winner == 0:
                human_wins += 1
            else:
                agent_wins += 1
    
    # Print summary
    if args.mode == 'play':
        print(f"\nFinal score: You {human_wins} - {agent_wins} Agent")

if __name__ == "__main__":
    args = parse_args()
    main(args) 