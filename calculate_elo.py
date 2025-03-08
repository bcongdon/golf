#!/usr/bin/env python3
"""
Calculate ELO ratings for multiple DQN models.

This script loads multiple models from a directory and runs them against each other
to calculate relative ELO scores, which helps compare model performance.
"""

import os
import argparse
import numpy as np
import torch
import random
import logging
import json
import itertools
import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from agent import DQNAgent
from golf_game_v2 import GolfGame, GameConfig, Action
from reflex_agent import ReflexAgent


def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('elo_calculator')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate ELO ratings for multiple DQN models')
    
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Directory containing model files')
    parser.add_argument('--output', type=str, default='elo_ratings.json',
                        help='Output file for ELO ratings (default: elo_ratings.json)')
    parser.add_argument('--games-per-matchup', type=int, default=100,
                        help='Number of DECISIVE games to play for each model matchup (ties excluded) (default: 100)')
    parser.add_argument('--games-vs-baseline', type=int, default=100,
                        help='Number of DECISIVE games to play against baseline agent (ties excluded) (default: 100)')
    parser.add_argument('--initial-elo', type=float, default=1200.0,
                        help='Initial ELO rating for all models (default: 1200.0)')
    parser.add_argument('--k-factor', type=float, default=32.0,
                        help='K-factor for ELO calculation (default: 32.0)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of the results')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip evaluation against baseline agent')
    
    return parser.parse_args()


def load_models(models_dir: str, logger: logging.Logger) -> Dict[str, DQNAgent]:
    """
    Load all model files from the specified directory.
    
    Args:
        models_dir: Directory containing model files
        logger: Logger instance
        
    Returns:
        Dictionary mapping model names to loaded DQNAgent instances
    """
    models = {}
    
    # Get all .pt files in the directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        logger.error(f"No model files found in {models_dir}")
        return {}
    
    logger.info(f"Found {len(model_files)} model files")
    
    # Load each model
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        
        try:
            # Initialize a new agent
            agent = DQNAgent()
            agent.load(model_path)
            models[model_name] = agent
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
    
    return models


def play_game(agent1: DQNAgent, agent2: DQNAgent) -> Tuple[int, int, int]:
    """
    Play a single game between two agents.
    
    Args:
        agent1: First agent (player 0)
        agent2: Second agent (player 1)
        
    Returns:
        Tuple of (winner, agent1_score, agent2_score)
        winner is 0 for agent1, 1 for agent2, -1 for draw
    """
    # Initialize environment
    config = GameConfig(num_players=2)
    env = GolfGame(config)
    
    # Reset environment
    state = env.reset()
    done = False
    
    while not done:
        current_player = env.current_player
        valid_actions = env._get_valid_actions()
        
        # Select action based on current player
        if current_player == 0:
            action = agent1.select_action(state, valid_actions, training=False)
        else:
            action = agent2.select_action(state, valid_actions, training=False)
        
        # Take action
        state, _, done, info = env.step(action)
    
    # Get final scores
    scores = info.get("scores", [0, 0])
    agent1_score = scores[0]
    agent2_score = scores[1]
    
    # Determine winner
    if agent1_score < agent2_score:
        winner = 0  # Agent 1 wins
    elif agent2_score < agent1_score:
        winner = 1  # Agent 2 wins
    else:
        winner = -1  # Draw
    
    return winner, agent1_score, agent2_score


def calculate_expected_score(rating1: float, rating2: float) -> float:
    """
    Calculate expected score for player 1 against player 2.
    
    Args:
        rating1: ELO rating of player 1
        rating2: ELO rating of player 2
        
    Returns:
        Expected score for player 1 (between 0 and 1)
    """
    return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))


def update_elo(rating1: float, rating2: float, score: float, k_factor: float) -> Tuple[float, float]:
    """
    Update ELO ratings based on game outcome.
    
    Args:
        rating1: Current ELO rating of player 1
        rating2: Current ELO rating of player 2
        score: Actual score for player 1 (1.0 for win, 0.5 for draw, 0.0 for loss)
        k_factor: K-factor for ELO calculation
        
    Returns:
        Tuple of (new_rating1, new_rating2)
    """
    expected1 = calculate_expected_score(rating1, rating2)
    expected2 = calculate_expected_score(rating2, rating1)
    
    new_rating1 = rating1 + k_factor * (score - expected1)
    new_rating2 = rating2 + k_factor * ((1.0 - score) - expected2)
    
    return new_rating1, new_rating2


def run_tournament(models: Dict[str, DQNAgent], games_per_matchup: int, 
                  initial_elo: float, k_factor: float, logger: logging.Logger) -> Dict[str, float]:
    """
    Run a tournament between all models and calculate ELO ratings.
    
    Args:
        models: Dictionary mapping model names to loaded DQNAgent instances
        games_per_matchup: Number of DECISIVE games to play for each model matchup (ties excluded)
        initial_elo: Initial ELO rating for all models
        k_factor: K-factor for ELO calculation
        logger: Logger instance
        
    Returns:
        Dictionary mapping model names to final ELO ratings
    """
    if len(models) < 2:
        logger.error("Need at least 2 models to run a tournament")
        return {name: initial_elo for name in models}
    
    # Initialize ELO ratings
    elo_ratings = {name: initial_elo for name in models}
    
    # Track win/loss/draw statistics
    stats = {name: {'wins': 0, 'losses': 0, 'draws': 0, 'avg_score': 0.0, 'total_games': 0} for name in models}
    
    # Generate all possible matchups
    matchups = list(itertools.combinations(models.keys(), 2))
    
    logger.info(f"Running tournament with {len(models)} models, {len(matchups)} matchups")
    logger.info(f"Playing until {games_per_matchup} decisive games per matchup (ties excluded)")
    
    # Run tournament
    total_games_estimate = len(matchups) * games_per_matchup * 1.5  # Estimate accounting for ties
    with tqdm.tqdm(total=total_games_estimate, desc="Playing games") as pbar:
        for model1_name, model2_name in matchups:
            model1 = models[model1_name]
            model2 = models[model2_name]
            
            # Play until we have games_per_matchup decisive games
            decisive_games = 0
            total_games = 0
            
            logger.debug(f"Starting matchup: {model1_name} vs {model2_name}")
            
            while decisive_games < games_per_matchup:
                # Play game with model1 as player 0 and model2 as player 1
                winner, model1_score, model2_score = play_game(model1, model2)
                total_games += 1
                
                # Update statistics
                if winner == 0:  # Model 1 wins
                    stats[model1_name]['wins'] += 1
                    stats[model2_name]['losses'] += 1
                    stats[model1_name]['total_games'] += 1
                    stats[model2_name]['total_games'] += 1
                    stats[model1_name]['avg_score'] += model1_score
                    stats[model2_name]['avg_score'] += model2_score
                    decisive_games += 1
                    
                    # Update ELO ratings (only for decisive games)
                    elo_ratings[model1_name], elo_ratings[model2_name] = update_elo(
                        elo_ratings[model1_name], elo_ratings[model2_name], 1.0, k_factor
                    )
                    
                elif winner == 1:  # Model 2 wins
                    stats[model1_name]['losses'] += 1
                    stats[model2_name]['wins'] += 1
                    stats[model1_name]['total_games'] += 1
                    stats[model2_name]['total_games'] += 1
                    stats[model1_name]['avg_score'] += model1_score
                    stats[model2_name]['avg_score'] += model2_score
                    decisive_games += 1
                    
                    # Update ELO ratings (only for decisive games)
                    elo_ratings[model1_name], elo_ratings[model2_name] = update_elo(
                        elo_ratings[model1_name], elo_ratings[model2_name], 0.0, k_factor
                    )
                    
                else:  # Draw
                    stats[model1_name]['draws'] += 1
                    stats[model2_name]['draws'] += 1
                    # Don't update total_games or avg_score for draws
                    # Don't update ELO ratings for draws
                
                pbar.update(1)
                
                # Update progress bar description with tie rate
                if total_games % 10 == 0:
                    tie_rate = (total_games - decisive_games) / total_games * 100
                    pbar.set_description(f"Playing games (Tie rate: {tie_rate:.1f}%)")
            
            logger.debug(f"Completed matchup: {model1_name} vs {model2_name}, "
                       f"Decisive games: {decisive_games}, Total games: {total_games}, "
                       f"Tie rate: {(total_games - decisive_games) / total_games * 100:.1f}%")
    
    # Calculate average scores (excluding draws)
    for name in stats:
        decisive_games = stats[name]['total_games']
        if decisive_games > 0:
            stats[name]['avg_score'] /= decisive_games
    
    # Log results
    logger.info("Tournament results (excluding ties):")
    for name, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True):
        decisive_games = stats[name]['wins'] + stats[name]['losses']
        if decisive_games > 0:
            win_rate = stats[name]['wins'] / decisive_games * 100
            logger.info(f"{name}: ELO={rating:.1f}, Win Rate={win_rate:.1f}%, Avg Score={stats[name]['avg_score']:.2f}, Decisive Games={decisive_games}")
        else:
            logger.info(f"{name}: ELO={rating:.1f}, No decisive games")
    
    # Return ELO ratings and stats
    return elo_ratings, stats


def evaluate_against_baseline(models: Dict[str, DQNAgent], games_per_model: int, 
                             logger: logging.Logger) -> Dict[str, Dict]:
    """
    Evaluate each model against a baseline reflex agent.
    
    Args:
        models: Dictionary mapping model names to loaded DQNAgent instances
        games_per_model: Number of DECISIVE games to play for each model (ties excluded)
        logger: Logger instance
        
    Returns:
        Dictionary mapping model names to performance statistics
    """
    # Initialize reflex agent as baseline
    reflex_agent = ReflexAgent(player_id=1)
    
    # Track statistics for each model
    baseline_stats = {}
    
    logger.info(f"Evaluating {len(models)} models against baseline reflex agent")
    logger.info(f"Playing until {games_per_model} decisive games per model (ties excluded)")
    
    # Evaluate each model
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} against baseline reflex agent")
        
        # Initialize statistics
        stats = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'avg_score': 0.0,
            'avg_opponent_score': 0.0,
            'total_games': 0
        }
        
        # Play until we have games_per_model decisive games
        decisive_games = 0
        total_games = 0
        
        # Play games
        with tqdm.tqdm(total=games_per_model, desc=f"Playing {model_name} vs Reflex") as pbar:
            while decisive_games < games_per_model:
                # Initialize environment
                config = GameConfig(num_players=2)
                env = GolfGame(config)
                
                # Reset environment
                state = env.reset()
                done = False
                
                while not done:
                    current_player = env.current_player
                    valid_actions = env._get_valid_actions()
                    
                    # Select action based on current player
                    if current_player == 0:
                        # Model plays as player 0
                        action = model.select_action(state, valid_actions, training=False)
                    else:
                        # Reflex agent plays as player 1
                        action = reflex_agent.select_action(state, valid_actions)
                    
                    # Take action
                    state, _, done, info = env.step(action)
                
                # Get final scores
                scores = info.get("scores", [0, 0])
                model_score = scores[0]
                reflex_score = scores[1]
                
                total_games += 1
                
                # Update statistics
                if model_score < reflex_score:  # Model wins
                    stats['wins'] += 1
                    stats['total_games'] += 1
                    stats['avg_score'] += model_score
                    stats['avg_opponent_score'] += reflex_score
                    decisive_games += 1
                    pbar.update(1)
                elif reflex_score < model_score:  # Reflex agent wins
                    stats['losses'] += 1
                    stats['total_games'] += 1
                    stats['avg_score'] += model_score
                    stats['avg_opponent_score'] += reflex_score
                    decisive_games += 1
                    pbar.update(1)
                else:  # Draw
                    stats['draws'] += 1
                    # Don't update total_games or avg_score for draws
                
                # Update progress bar description with tie rate
                if total_games % 10 == 0:
                    tie_rate = (total_games - decisive_games) / total_games * 100
                    pbar.set_description(f"Playing {model_name} vs Reflex (Tie rate: {tie_rate:.1f}%)")
        
        # Calculate averages (excluding draws)
        if stats['total_games'] > 0:
            stats['avg_score'] /= stats['total_games']
            stats['avg_opponent_score'] /= stats['total_games']
            stats['win_rate'] = stats['wins'] / stats['total_games'] * 100
        else:
            stats['win_rate'] = 0.0
        
        # Log results
        decisive_games = stats['total_games']
        tie_rate = stats['draws'] / total_games * 100 if total_games > 0 else 0
        
        logger.info(f"{model_name} vs Reflex: Win Rate={stats['win_rate']:.1f}%, "
                   f"Avg Score={stats['avg_score']:.2f}, "
                   f"Avg Opponent Score={stats['avg_opponent_score']:.2f}, "
                   f"Decisive Games={decisive_games}, "
                   f"Total Games={total_games}, "
                   f"Tie Rate={tie_rate:.1f}%")
        
        # Store statistics
        baseline_stats[model_name] = stats
    
    return baseline_stats


def save_results(elo_ratings: Dict[str, float], stats: Dict[str, Dict], 
                baseline_stats: Dict[str, Dict], output_file: str, logger: logging.Logger):
    """
    Save ELO ratings and statistics to a JSON file.
    
    Args:
        elo_ratings: Dictionary mapping model names to ELO ratings
        stats: Dictionary mapping model names to statistics
        baseline_stats: Dictionary mapping model names to baseline performance statistics
        output_file: Output file path
        logger: Logger instance
    """
    # Calculate win rates excluding ties
    win_rates = {}
    for name, model_stats in stats.items():
        decisive_games = model_stats['wins'] + model_stats['losses']
        if decisive_games > 0:
            win_rates[name] = model_stats['wins'] / decisive_games * 100
        else:
            win_rates[name] = 0.0
    
    # Combine ELO ratings and statistics
    results = {
        'elo_ratings': {name: float(rating) for name, rating in elo_ratings.items()},
        'statistics': stats,
        'baseline_statistics': baseline_stats,
        'win_rates_excluding_ties': win_rates,
        'metadata': {
            'ties_excluded': True,
            'description': 'ELO ratings and win rates calculated excluding ties. Average scores are also calculated excluding ties.'
        }
    }
    
    # Sort models by ELO rating
    sorted_models = sorted(results['elo_ratings'].items(), key=lambda x: x[1], reverse=True)
    results['ranking'] = [
        {
            'model': name, 
            'elo': rating,
            'win_rate_excluding_ties': win_rates[name],
            'decisive_games': stats[name]['wins'] + stats[name]['losses'],
            'draws': stats[name]['draws']
        } 
        for name, rating in sorted_models
    ]
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def visualize_results(elo_ratings: Dict[str, float], stats: Dict[str, Dict], 
                     baseline_stats: Dict[str, Dict], output_prefix: str, logger: logging.Logger):
    """
    Visualize ELO ratings and statistics.
    
    Args:
        elo_ratings: Dictionary mapping model names to ELO ratings
        stats: Dictionary mapping model names to statistics
        baseline_stats: Dictionary mapping model names to baseline performance statistics
        output_prefix: Prefix for output files
        logger: Logger instance
    """
    try:
        # Sort models by ELO rating
        sorted_models = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
        model_names = [name for name, _ in sorted_models]
        ratings = [rating for _, rating in sorted_models]
        
        # Create ELO rating bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, ratings)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.title('ELO Ratings (Excluding Ties)')
        plt.xlabel('Model')
        plt.ylabel('ELO Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        elo_chart_path = f"{output_prefix}_elo_ratings.png"
        plt.savefig(elo_chart_path)
        logger.info(f"ELO ratings chart saved to {elo_chart_path}")
        plt.close()
        
        # Create win rate chart (excluding ties)
        win_rates = []
        for name in model_names:
            decisive_games = stats[name]['wins'] + stats[name]['losses']
            if decisive_games > 0:
                win_rate = stats[name]['wins'] / decisive_games * 100
            else:
                win_rate = 0
            win_rates.append(win_rate)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, win_rates)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Win Rates (Excluding Ties)')
        plt.xlabel('Model')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        win_rate_chart_path = f"{output_prefix}_win_rates.png"
        plt.savefig(win_rate_chart_path)
        logger.info(f"Win rates chart saved to {win_rate_chart_path}")
        plt.close()
        
        # Create average score chart
        avg_scores = [stats[name]['avg_score'] for name in model_names]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, avg_scores)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Average Scores (Lower is Better, Excluding Ties)')
        plt.xlabel('Model')
        plt.ylabel('Average Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        avg_score_chart_path = f"{output_prefix}_avg_scores.png"
        plt.savefig(avg_score_chart_path)
        logger.info(f"Average scores chart saved to {avg_score_chart_path}")
        plt.close()
        
        # Create baseline win rate chart if baseline stats are available
        if baseline_stats:
            baseline_win_rates = []
            for name in model_names:
                if name in baseline_stats and baseline_stats[name]['total_games'] > 0:
                    win_rate = baseline_stats[name]['win_rate']
                else:
                    win_rate = 0
                baseline_win_rates.append(win_rate)
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(model_names, baseline_win_rates)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.title('Win Rates Against Baseline (Reflex Agent, Excluding Ties)')
            plt.xlabel('Model')
            plt.ylabel('Win Rate (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            baseline_chart_path = f"{output_prefix}_baseline_win_rates.png"
            plt.savefig(baseline_chart_path)
            logger.info(f"Baseline win rates chart saved to {baseline_chart_path}")
            plt.close()
        
    except Exception as e:
        logger.error(f"Failed to visualize results: {e}")


def main():
    """Main function."""
    args = parse_args()
    logger = setup_logging(args.log_level)
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Load models
    models = load_models(args.models_dir, logger)
    
    if not models:
        logger.error("No models loaded, exiting")
        return
    
    # Evaluate against baseline if not skipped
    baseline_stats = {}
    if not args.skip_baseline:
        baseline_stats = evaluate_against_baseline(models, args.games_vs_baseline, logger)
    
    # Run tournament
    elo_ratings, stats = run_tournament(
        models, args.games_per_matchup, args.initial_elo, args.k_factor, logger
    )
    
    # Save results
    save_results(elo_ratings, stats, baseline_stats, args.output, logger)
    
    # Visualize results if requested
    if args.visualize:
        output_prefix = os.path.splitext(args.output)[0]
        visualize_results(elo_ratings, stats, baseline_stats, output_prefix, logger)


if __name__ == "__main__":
    main() 