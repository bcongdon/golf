import argparse
import os
import numpy as np
import torch
import random
from collections import defaultdict
from typing import List, Dict, Any, Optional

from golf_game import GolfGame
from agent import DQNAgent
from llm_agent import LLMAgent, ClaudeAgent, GPTAgent, DeepSeekAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Play Golf with multiple agents')
    parser.add_argument('--agents', type=str, nargs='+', default=['dqn', 'human'],
                        help='List of agents to play (dqn, claude, gpt, deepseek, human)')
    parser.add_argument('--models', type=str, nargs='+', default=['models/best_model.pth'],
                        help='Paths to the trained models for DQN agents')
    parser.add_argument('--llm-models', type=str, nargs='+', 
                        default=[],
                        help='LLM model names for LLM agents')
    parser.add_argument('--games', type=int, default=1, help='Number of games to play')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size of the neural network')
    parser.add_argument('--verbose', action='store_true', help='Print detailed game information')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    return parser.parse_args()

def get_human_action(env):
    """Get action from human player."""
    valid_actions = env._get_valid_actions()
    player_idx = env.current_player
    
    # Explain game mechanics
    print("\nNote: You can only see your revealed cards. Cards are revealed when you replace them.")
    print("      Matching cards in the same column cancel out (worth 0 points).")
    print("      Aces are worth 1, number cards are face value, face cards (J,Q,K) are worth 10.")
    print("      Twos (2) are worth -2 points.")
    print("      The goal is to have the lowest score.\n")
    
    # Display valid actions
    print("\nValid actions:")
    if 0 in valid_actions:
        print("0: Draw from deck")
    if 1 in valid_actions and env.discard_pile:
        print("1: Take from discard pile (Note: If you take a card from the discard pile, you MUST place it in your hand)")
    
    # Group replace card actions
    replace_actions = [action for action in valid_actions if 2 <= action <= 7]
    if replace_actions:
        print("\nReplace card at position:")
        for action in replace_actions:
            position = action - 2
            row = position // 3
            col = position % 3
            
            # Indicate if the card is revealed or not
            if position in env.revealed_cards[player_idx]:
                card = env.player_hands[player_idx][position]
                suit, rank = card
                suit_symbol = ["♣", "♦", "♥", "♠"][suit]
                rank_symbol = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"][rank]
                card_str = f"{rank_symbol}{suit_symbol}"
                value = env._card_value(card)
                print(f"{action}: Position {position} (row {row}, column {col}) - Revealed card: {card_str}({value})")
            else:
                print(f"{action}: Position {position} (row {row}, column {col}) - Hidden card")
    
    if 8 in valid_actions:
        print("\n8: Discard drawn card")
    
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

def create_agent(agent_type, model_path=None, llm_model=None, hidden_size=128):
    """Create an agent based on the specified type."""
    if agent_type == 'dqn':
        state_size = 60  # Simplified observation space (removed suits)
        action_size = 9
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size
        )
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded DQN model from {model_path}")
        else:
            print(f"Model file {model_path} not found. Using untrained DQN agent.")
        return agent
    elif agent_type == 'claude':
        model_name = llm_model if llm_model else "claude-3-sonnet-20240229"
        return ClaudeAgent(model_name=model_name)
    elif agent_type == 'gpt':
        model_name = llm_model if llm_model else "gpt-4"
        return GPTAgent(model_name=model_name)
    elif agent_type == 'deepseek':
        model_name = llm_model if llm_model else "deepseek-reasoner"
        return DeepSeekAgent(model_name=model_name)
    elif agent_type == 'human':
        return None  # Human player is handled separately
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def display_game_state(env, current_player, agent_type):
    """Display the game state in a consistent format for all players."""
    if agent_type == "human":
        print(f"\n=== Player {current_player}'s turn (YOU) ===")
    else:
        print(f"\n=== Player {current_player}'s turn ({agent_type}) ===")
    
    # Show player hands
    for p in range(env.num_players):
        if p == current_player and agent_type == "human":
            print(f"Player {p}'s hand (YOU):")
        elif p == current_player:
            print(f"Player {p}'s hand (CURRENT PLAYER):")
        else:
            print(f"Player {p}'s hand:")
        
        # Display cards in a 2x3 grid format
        # First show position labels
        print("  Positions:")
        print("  0 1 2")
        print("  3 4 5")
        
        for row in range(2):
            row_str = "  "
            for col in range(3):
                pos = row * 3 + col
                card = env.player_hands[p][pos]
                # Only show revealed cards
                if pos in env.revealed_cards[p]:
                    suit, rank = card
                    suit_symbol = ["♣", "♦", "♥", "♠"][suit]
                    rank_symbol = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"][rank]
                    card_str = f"{rank_symbol}{suit_symbol}"
                    value = env._card_value(card)
                    row_str += f"{card_str}({value}) "
                else:
                    row_str += "?? "
            print(row_str)
        
        # Add score if all cards are revealed
        if len(env.revealed_cards[p]) == 6:
            score = env._calculate_score(p)
            print(f"  Score: {score}")
        print()
    
    # Show discard pile
    if env.discard_pile:
        top_discard = env.discard_pile[-1]
        suit, rank = top_discard
        suit_symbol = ["♣", "♦", "♥", "♠"][suit]
        rank_symbol = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"][rank]
        print(f"Top of discard pile: {rank_symbol}{suit_symbol}")
    else:
        print("Discard pile is empty")
    
    # Show drawn card (if any)
    if env.drawn_card is not None:
        suit, rank = env.drawn_card
        suit_symbol = ["♣", "♦", "♥", "♠"][suit]
        rank_symbol = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"][rank]
        card_str = f"{rank_symbol}{suit_symbol}"
        value = env._card_value(env.drawn_card)
        
        if env.drawn_from_discard:
            if agent_type == "human":
                print(f"Drawn card: {card_str}({value}) - You MUST replace one of your cards with this card")
            else:
                print(f"Drawn card: {card_str}({value}) - Must replace one of the cards")
        else:
            if agent_type == "human":
                print(f"Drawn card: {card_str}({value}) - You can replace one of your cards with this or discard it")
            else:
                print(f"Drawn card: {card_str}({value})")
    
    # Show final round info
    if env.final_round:
        print("FINAL ROUND: Game will end after each player gets one more turn")
    
    print()  # Add an extra line for readability

def play_game(agents, agent_types, verbose=True):
    """Play a game with multiple agents."""
    env = GolfGame(num_players=len(agents))
    state = env.reset()
    done = False
    
    # Game loop
    while not done:
        # Determine whose turn it is
        current_player = env.current_player
        agent_type = agent_types[current_player]
        agent = agents[current_player]
        
        # Display game state before action
        if verbose:
            display_game_state(env, current_player, agent_type)
        
        # Get valid actions
        valid_actions = env._get_valid_actions()
        
        if agent_type == 'human':
            # Human's turn
            action = get_human_action(env)
        else:
            # Agent's turn
            if isinstance(agent, LLMAgent):
                action = agent.select_action(state, valid_actions, env, training=False)
            else:
                action = agent.select_action(state, valid_actions, training=False)
            
            if verbose:
                print(f"\nPlayer {current_player} ({agent_type}) selects action: {action}")
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        # Print any errors or info
        if 'error' in info and verbose:
            print(f"Error: {info['error']}")
        
        # Announce final round
        if 'final_round' in info and info['final_round'] and verbose:
            trigger_player = info['trigger_player']
            trigger_agent_type = agent_types[trigger_player]
            print(f"\nPlayer {trigger_player} ({trigger_agent_type}) revealed all cards! Final round begins.")
        
        # Display game state after action if it's not a human player
        # (human players will see the updated state on their next turn)
        if verbose and agent_type != 'human' and not done and env.drawn_card is None:
            print("\n=== After action ===")
            display_game_state(env, env.current_player, agent_types[env.current_player])
        
        # Add a separator between turns if not the last turn
        if not done and verbose:
            print("\n" + "-" * 50 + "\n")
    
    # Game over
    if verbose:
        print("\n=== GAME OVER ===")
        display_game_state(env, env.current_player, agent_types[env.current_player])
    
    scores = [env._calculate_score(p) for p in range(env.num_players)]
    if verbose:
        print("Game over!")
        print(f"Final scores: {scores}")
    
    winner = np.argmin(scores)
    if verbose:
        print(f"Player {winner} ({agent_types[winner]}) wins!")
    
    return winner, scores

def main(args):
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Validate inputs
    if len(args.agents) < 2:
        print("Error: At least 2 agents are required")
        return
    
    # Create agents
    agents = []
    agent_types = []
    
    for i, agent_type in enumerate(args.agents):
        if agent_type == 'dqn':
            model_path = args.models[0] if i < len(args.models) else None
            agent = create_agent(agent_type, model_path=model_path, hidden_size=args.hidden_size)
        elif agent_type in ['claude', 'gpt', 'deepseek']:
            llm_model = args.llm_models[0] if i < len(args.llm_models) else None
            agent = create_agent(agent_type, llm_model=llm_model)
        elif agent_type == 'human':
            agent = None  # Human player is handled separately
        else:
            print(f"Unknown agent type: {agent_type}. Skipping.")
            continue
        
        agents.append(agent)
        agent_types.append(agent_type)
    
    # Play games
    wins = defaultdict(int)
    all_scores = defaultdict(list)
    
    for game in range(args.games):
        print(f"\n===== Game {game+1}/{args.games} =====\n")
        winner, scores = play_game(agents, agent_types, verbose=args.verbose)
        
        # Record results
        wins[agent_types[winner]] += 1
        for i, agent_type in enumerate(agent_types):
            all_scores[agent_type].append(scores[i])
    
    # Print summary
    print("\n===== Results =====")
    print("Wins:")
    for agent_type, win_count in wins.items():
        win_percentage = (win_count / args.games) * 100
        print(f"{agent_type}: {win_count} ({win_percentage:.1f}%)")
    
    print("\nAverage Scores:")
    for agent_type, scores in all_scores.items():
        avg_score = sum(scores) / len(scores)
        print(f"{agent_type}: {avg_score:.2f}")

if __name__ == "__main__":
    args = parse_args()
    main(args) 