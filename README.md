# Golf Card Game AI

This project implements a neural network (MLP) using reinforcement learning to play the card game Golf. It also supports LLM-based agents using Langchain.

## Game Rules

Golf is a card game where players try to minimize their score by strategically replacing cards in their hand. The basic rules include:

- Each player has a 2x3 grid of face-down cards (6 cards total)
- At the start, two cards are revealed for each player
- On their turn, a player can:
  1. Draw a card from the deck or discard pile
  2. Replace one of their cards with the drawn card or discard it
- Cards are from a standard deck (A-K in four suits)
- Aces are worth 1, number cards are face value, face cards (J,Q,K) are worth 10
- Matching cards in the same column cancel out (worth 0)
- Twos (2) are worth -2 points
- When a player reveals all their cards, the game enters its final round where each player gets one more turn
- The player with the lowest total score wins

## Project Structure

- `golf_game.py`: Implementation of the Golf card game environment
- `agent.py`: Neural network agent using reinforcement learning
- `llm_agent.py`: LLM-based agents using Langchain (Claude, GPT, DeepSeek)
- `train.py`: Script to train the DQN agent
- `play.py`: Script to play against or watch the trained agent
- `multi_agent_play.py`: Script to play with multiple agents competing against each other

## Setup

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Set up environment variables for LLM API access (if using LLM agents):

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

3. Train the DQN agent:

```
python train.py
```

4. Play against the trained agent:

```
python play.py
```

5. Play with multiple agents:

```
python multi_agent_play.py --agents dqn claude gpt human --games 3
```

## LLM Agents

The project supports the following LLM-based agents:

- **Claude**: Uses Anthropic's Claude models via Langchain
- **GPT**: Uses OpenAI's GPT models via Langchain
- **DeepSeek**: Uses DeepSeek models via Langchain

You can specify which LLM models to use with the `--llm-models` parameter:

```
python multi_agent_play.py --agents claude gpt --llm-models claude-3-sonnet-20240229 gpt-4
```

## Multi-Agent Play

The `multi_agent_play.py` script allows you to pit different agents against each other. Available agent types:

- `dqn`: The trained neural network agent
- `claude`: Claude LLM agent
- `gpt`: GPT LLM agent
- `deepseek`: DeepSeek LLM agent
- `human`: Human player

Example usage:

```
# Play a game with Claude vs GPT vs DQN
python multi_agent_play.py --agents claude gpt dqn --games 5

# Play as a human against Claude
python multi_agent_play.py --agents human claude

# Compare multiple LLM models
python multi_agent_play.py --agents claude claude --llm-models claude-3-sonnet-20240229 claude-3-opus-20240229

# Compare DeepSeek with other models
python multi_agent_play.py --agents deepseek gpt --llm-models deepseek-chat gpt-4
```

## Implementation Details

The project uses a Multi-Layer Perceptron (MLP) trained with reinforcement learning. The agent learns through self-play and experience replay to develop optimal strategies for the Golf card game.

For LLM agents, the game state is converted to a text representation and sent to the LLM with instructions on how to play. The LLM's response is parsed to extract the chosen action.
