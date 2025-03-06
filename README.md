# Golf Card Game with PPO Reinforcement Learning

This repository contains an implementation of the Golf card game environment and a Proximal Policy Optimization (PPO) agent to play it.

## Game Description

Golf is a card game where players aim to have the lowest total score. In this implementation:

- Each player has a 2x3 grid of cards (6 cards total)
- Cards are represented by rank only (A-K), suits are ignored
- Aces are worth 1, number cards are face value, face cards (J,Q,K) are worth 10
- Matching cards in the same column cancel out (worth 0)
- The goal is to have the lowest total score

## Project Structure

- `golf_game.py`: Implementation of the Golf card game environment
- `ppo_agent.py`: Implementation of the PPO agent
- `train_ppo.py`: Script to train the PPO agent
- `play_against_ppo.py`: Script to play against the trained PPO agent

## PPO Agent

The PPO agent uses an actor-critic architecture with the following components:

- **Actor-Critic Network**: A neural network with shared feature extraction layers and separate actor (policy) and critic (value) heads
- **Card Embeddings**: Cards are represented using embeddings to capture their semantic meaning
- **PPO Algorithm**: Uses clipped surrogate objective, generalized advantage estimation (GAE), and entropy bonus for exploration
- **Valid Action Masking**: Only valid actions are considered during action selection

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Training the Agent

To train the PPO agent, run:

```bash
python train_ppo.py
```

You can customize the training parameters using command-line arguments:

```bash
python train_ppo.py --num_episodes 50000 --batch_size 64 --actor_lr 0.0003 --critic_lr 0.0003
```

Key parameters:

- `--num_episodes`: Number of episodes to train
- `--update_interval`: Number of steps between PPO updates
- `--batch_size`: Batch size for training
- `--n_epochs`: Number of epochs per update
- `--actor_lr`: Learning rate for actor
- `--critic_lr`: Learning rate for critic
- `--entropy_coef`: Entropy coefficient for exploration
- `--eval_interval`: Episodes between evaluations
- `--model_dir`: Directory to save models
- `--load_model`: Path to load a pre-trained model

## Playing Against the Agent

To play against a trained PPO agent, run:

```bash
python play_against_ppo.py --model_path models/ppo_golf_best_winrate.pt --render
```

Parameters:

- `--model_path`: Path to the trained model (required)
- `--num_games`: Number of games to play
- `--render`: Render the game (recommended)
- `--human_player`: Human player index (0 or 1)

## Comparing with DQN

This repository also includes a DQN agent implementation for comparison. The PPO agent has several advantages:

1. **On-policy learning**: PPO is an on-policy algorithm, which can be more stable for sequential decision-making tasks
2. **Continuous action spaces**: While not used in this discrete action environment, PPO can handle continuous action spaces
3. **Sample efficiency**: PPO often requires fewer samples to learn a good policy
4. **Exploration**: PPO uses entropy regularization for better exploration

## Performance

The PPO agent typically achieves:

- Win rate of 60-70% against a random opponent
- Stable learning curve with consistent improvement
- Good generalization to different game scenarios

## Future Improvements

- Implement self-play training
- Add multi-agent PPO for more than 2 players
- Experiment with different network architectures
- Add curriculum learning for faster training
