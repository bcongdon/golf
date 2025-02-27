import os
import json
import random
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_anthropic import AnthropicLLM
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

from golf_game import GolfGame

# Load environment variables
load_dotenv()

class GolfAction(BaseModel):
    """Model for the Golf game action."""
    reasoning: str = Field(description="Reasoning behind the action")
    action: int = Field(description="The action to take (0-8)")

class LLMAgent:
    """Base class for LLM-based agents for the Golf card game."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        """Initialize the LLM agent with the specified model."""
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.parser = JsonOutputParser(pydantic_object=GolfAction)
        
        # Create the prompt template
        self.prompt = PromptTemplate(
            template="""
            You are playing the card game Golf. Your goal is to have the lowest score by strategically replacing cards.

            Game Rules:
            - Each player has a 2x3 grid of cards (6 cards total)
            - Cards are from a standard deck (A-K in four suits)
            - Aces are worth 1, number cards are face value, face cards (J,Q,K) are worth 10
            - Matching cards in the same column cancel out (worth 0)
            - Twos (2) are worth -2 points
            - The player with the lowest total score wins

            Current Game State:
            {game_state}

            Valid Actions:
            {valid_actions}

            Select the best action based on the current game state. Respond with a JSON object containing:
            - action: The action number to take
            - reasoning: Your reasoning for this action

            {format_instructions}
            """,
            input_variables=["game_state", "valid_actions"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on the model name."""
        if "gpt" in self.model_name.lower():
            return ChatOpenAI(model=self.model_name, temperature=self.temperature)
        elif "claude" in self.model_name.lower():
            return AnthropicLLM(model=self.model_name, temperature=self.temperature)
        elif "deepseek" in self.model_name.lower():
            return ChatDeepSeek(model=self.model_name, temperature=self.temperature)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _format_game_state(self, env: GolfGame) -> str:
        """Format the game state as a string for the LLM."""
        state_str = []
        
        # Current player info
        player_idx = env.current_player
        state_str.append(f"You are Player {player_idx}")
        
        # Player hands
        for p in range(env.num_players):
            hand_str = f"Player {p}'s hand:"
            for i, card in enumerate(env.player_hands[p]):
                if i in env.revealed_cards[p] or p == player_idx:
                    suit, rank = card
                    suit_symbol = ["♣", "♦", "♥", "♠"][suit]
                    rank_symbol = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"][rank]
                    card_str = f"{rank_symbol}{suit_symbol}"
                    value = env._card_value(card)
                    hand_str += f" {card_str}({value})"
                else:
                    hand_str += " ??"
            
            # Add score if all cards are revealed
            if len(env.revealed_cards[p]) == 6:
                score = env._calculate_score(p)
                hand_str += f" - Score: {score}"
            
            state_str.append(hand_str)
        
        # Discard pile
        if env.discard_pile:
            top_discard = env.discard_pile[-1]
            suit, rank = top_discard
            suit_symbol = ["♣", "♦", "♥", "♠"][suit]
            rank_symbol = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"][rank]
            state_str.append(f"Top of discard pile: {rank_symbol}{suit_symbol}")
        
        # Drawn card (if any)
        if env.drawn_card is not None:
            suit, rank = env.drawn_card
            suit_symbol = ["♣", "♦", "♥", "♠"][suit]
            rank_symbol = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"][rank]
            state_str.append(f"Drawn card: {rank_symbol}{suit_symbol}")
        
        # Final round info
        if env.final_round:
            state_str.append("FINAL ROUND: Game will end after each player gets one more turn")
        
        return "\n".join(state_str)
    
    def _format_valid_actions(self, valid_actions: List[int], env: GolfGame) -> str:
        """Format the valid actions as a string for the LLM."""
        action_str = []
        
        for action in valid_actions:
            if action == 0:
                action_str.append("0: Draw from deck")
            elif action == 1:
                action_str.append("1: Take from discard pile")
            elif 2 <= action <= 7:
                position = action - 2
                row = position // 3
                col = position % 3
                action_str.append(f"{action}: Replace card at position {position} (row {row}, column {col})")
            elif action == 8:
                action_str.append("8: Discard drawn card")
        
        return "\n".join(action_str)
    
    def select_action(self, state: np.ndarray, valid_actions: List[int], env: GolfGame, training: bool = False) -> int:
        """Select an action based on the current game state using the LLM."""
        # Format the game state and valid actions
        game_state = self._format_game_state(env)
        valid_actions_str = self._format_valid_actions(valid_actions, env)
        
        # Create the prompt
        prompt_value = self.prompt.format(
            game_state=game_state,
            valid_actions=valid_actions_str
        )
        
        # Get the LLM response
        try:
            # Stream the LLM response
            print("\nLLM thinking process:")
            response_content = ""
            for chunk in self.llm.stream(prompt_value):
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                print(chunk_content, end="", flush=True)
                response_content += chunk_content
            print("\n")  # Add newline after streaming completes
            
            # Try to parse the response as JSON first
            try:
                # Check if the response is already a JSON string
                if isinstance(response_content, str):
                    try:
                        # Try to parse as raw JSON first
                        json_response = json.loads(response_content)
                        action = json_response.get('action')
                        if action is not None:
                            print(f"action: {action}")
                            print(f"parsed_response: {json_response}")
                            
                            # Ensure the action is valid
                            if action not in valid_actions:
                                print(f"Warning: LLM selected invalid action {action}. Choosing randomly from valid actions.")
                                action = random.choice(valid_actions)
                            
                            return action
                    except json.JSONDecodeError:
                        # If not valid JSON, try using the parser
                        parsed_response = self.parser.parse(response_content)
                        action = parsed_response.action
                        
                        print(f"action: {action}")
                        print(f"parsed_response: {parsed_response}")
                        
                        # Ensure the action is valid
                        if action not in valid_actions:
                            print(f"Warning: LLM selected invalid action {action}. Choosing randomly from valid actions.")
                            action = random.choice(valid_actions)
                        
                        return action
            except Exception as parsing_error:
                print(f"Error parsing response: {parsing_error}")
                # Fallback to extracting action directly from text
                # Look for patterns like "action: 0" or "I choose action 0"
                import re
                action_match = re.search(r'action[:\s]+(\d+)', response_content, re.IGNORECASE)
                if action_match:
                    action = int(action_match.group(1))
                    print(f"Extracted action {action} from text")
                    if action in valid_actions:
                        return action
                
                # If all parsing attempts fail, choose randomly
                print("Failed to parse action from response. Choosing randomly.")
                return random.choice(valid_actions)
                
        except Exception as e:
            print(f"llm: {e}")
            # Fallback to random action
            return random.choice(valid_actions)
    
    def remember(self, *args, **kwargs):
        """Placeholder to match DQNAgent interface."""
        pass
    
    def learn(self):
        """Placeholder to match DQNAgent interface."""
        pass
    
    def save(self, path: str):
        """Placeholder to match DQNAgent interface."""
        pass
    
    def load(self, path: str):
        """Placeholder to match DQNAgent interface."""
        pass


class ClaudeAgent(LLMAgent):
    """Claude-specific implementation of the LLM agent."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", temperature: float = 0.0):
        super().__init__(model_name=model_name, temperature=temperature)


class GPTAgent(LLMAgent):
    """GPT-specific implementation of the LLM agent."""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        super().__init__(model_name=model_name, temperature=temperature)


class DeepSeekAgent(LLMAgent):
    """DeepSeek-specific implementation of the LLM agent."""
    
    def __init__(self, model_name: str = "deepseek-chat", temperature: float = 0.0):
        super().__init__(model_name=model_name, temperature=temperature) 