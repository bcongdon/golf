import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union

class GolfGame:
    """
    Implementation of the Golf card game environment.
    
    In this version of Golf:
    - Each player has a 2x3 grid of cards (6 cards total)
    - Cards are represented by rank only (A-K), suits are ignored
    - Aces are worth 1, number cards are face value, face cards (J,Q,K) are worth 10
    - Matching cards in the same column cancel out (worth 0)
    - The goal is to have the lowest total score
    """
    
    def __init__(self, num_players: int = 2, normalize_rewards: bool = True):
        self.num_players = num_players
        self.normalize_rewards = normalize_rewards
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state and return the initial observation."""
        # Initialize a deck of cards (ranks only, no suits)
        # We'll create 4 copies of each rank to simulate a standard deck
        self.deck = []
        for _ in range(4):  # 4 copies of each rank (simulating suits)
            for rank in range(13):  # 0-12 for A-K
                self.deck.append(rank)
        random.shuffle(self.deck)
        
        # Deal cards to players (2x3 grid for each player)
        self.player_hands = []
        for _ in range(self.num_players):
            hand = []
            for _ in range(6):  # 6 cards per player
                hand.append(self.deck.pop())
            self.player_hands.append(hand)
        
        # Initialize game state
        self.current_player = 0
        self.discard_pile = [self.deck.pop()]
        self.game_over = False
        self.revealed_cards = [set() for _ in range(self.num_players)]  # Track which cards are face up
        self.drawn_card = None  # Initialize drawn_card attribute
        self.drawn_from_discard = False  # Track if the drawn card came from the discard pile
        
        # Initially reveal two cards for each player
        for player in range(self.num_players):
            positions = random.sample(range(6), 2)
            for pos in positions:
                self.revealed_cards[player].add(pos)
        
        # Track if final round has been triggered
        self.final_round = False
        self.last_player = None  # The player who gets the last turn
        
        # Add turn counter to prevent infinite games
        self.turn_count = 0
        self.max_turns = 100  # Maximum number of turns before forcing game end
        
        return self._get_observation()
    
    def _card_value(self, rank: int) -> int:
        """Calculate the value of a card based on its rank."""
        if rank == 0:  # Ace
            return 1
        if rank == 1:  # 2 is worth -2
            return -2
        elif rank < 10:  # 3-10
            return rank + 1
        else:  # Jack, Queen, King
            return 10
    
    def _calculate_score(self, player: int) -> int:
        """Calculate the score for a player."""
        hand = self.player_hands[player]
        score = 0
        
        # Check columns (0,3), (1,4), (2,5) for matches
        for col in range(3):
            card1 = hand[col]
            card2 = hand[col + 3]
            
            # If ranks match, they cancel out
            if card1 == card2:
                # Special case for 2s
                if card1 == 1:
                    score -= 4
                continue
            else:
                score += self._card_value(card1) + self._card_value(card2)
        
        return score
    
    def _get_observation(self) -> np.ndarray:
        """
        Convert the game state to a neural network input representation using card indices.
        
        Each card is represented by its rank index (0-12 for A-K), with 13 representing unknown/hidden cards.
        
        The observation includes:
        - Current player's hand (6 card indices)
        - Opponent's visible cards (6 card indices, 13 for hidden)
        - Discard pile top card (1 card index)
        - Drawn card (1 card index)
        - Revealed flags for player (6 binary values)
        - Revealed flags for opponent (6 binary values)
        - Game state flags (2 binary values: drawn_from_discard, final_round)
        
        Total: 28 dimensions
        """
        # Initialize observation array with unknown card value (13) for all card positions
        # First 14 elements are card indices, last 14 are binary features
        obs = np.zeros(28, dtype=np.float32)
        obs[:14] = 13  # Set all card positions to "unknown" by default
        
        # Encode current player's hand (first 6 positions)
        for i, rank in enumerate(self.player_hands[self.current_player]):
            # For revealed cards, encode the actual rank
            if i in self.revealed_cards[self.current_player]:
                obs[i] = rank
                # Set revealed flag
                obs[14 + i] = 1.0
        
        # Encode opponent's visible cards (next 6 positions)
        opponent = (self.current_player + 1) % self.num_players
        for i, rank in enumerate(self.player_hands[opponent]):
            # Only encode revealed cards
            if i in self.revealed_cards[opponent]:
                obs[6 + i] = rank
                # Set revealed flag
                obs[20 + i] = 1.0
        
        # Encode discard pile top card (position 12)
        if self.discard_pile:
            obs[12] = self.discard_pile[-1]
        
        # Encode drawn card (position 13)
        if self.drawn_card is not None:
            obs[13] = self.drawn_card
            
            # Set drawn from discard flag (index 26)
            if self.drawn_from_discard:
                obs[26] = 1.0
        
        # Set final round flag (index 27)
        if self.final_round:
            obs[27] = 1.0
            
        return obs
    
    def _get_valid_actions(self) -> List[int]:
        """Return list of valid actions for current player."""
        # Actions:
        # 0: Draw from deck
        # 1: Take from discard pile
        # 2-7: Replace card at position 0-5 with drawn card
        # 8: Discard drawn card (was previously 9)
        
        if self.game_over:
            return []
            
        if self.drawn_card is None:
            # Can only draw from deck or discard
            valid_actions = [0]  # Always can draw from deck
            if self.discard_pile:  # Only add discard pile action if it's not empty
                valid_actions.append(1)
            return valid_actions
        else:
            # Can replace any card
            valid_actions = list(range(2, 8))  # Actions 2-7 for replacing cards
            
            # Can only discard if the card was drawn from the deck (not from discard pile)
            if not self.drawn_from_discard:
                valid_actions.append(8)  # Action 8 for discarding
                
            return valid_actions
    
    def _normalize_reward(self, reward: float, is_terminal: bool = False) -> float:
        """
        Normalize rewards to a consistent scale.
        
        For terminal rewards (end of game):
        - Win rewards are normalized to [0.5, 1.0]
        - Loss penalties are normalized to [-1.0, -0.5]
        
        For intermediate rewards:
        - All rewards are normalized to [-0.5, 0.5]
        
        Args:
            reward: The raw reward value
            is_terminal: Whether this is a terminal reward (end of game)
            
        Returns:
            Normalized reward value
        """
        if not self.normalize_rewards:
            return reward
            
        if is_terminal:
            if reward > 0:  # Win reward
                # Map positive terminal rewards to [0.5, 1.0]
                # Typical win reward is 5.0 + margin bonus (up to ~10)
                return 0.5 + min(0.5, reward / 10.0)
            else:  # Loss penalty
                # Map negative terminal rewards to [-1.0, -0.5]
                # Typical loss penalty is -0.3 * score_diff (up to ~-10)
                return -0.5 + max(-0.5, reward / 10.0)
        else:  # Intermediate rewards
            # Map intermediate rewards to [-0.5, 0.5]
            # Typical intermediate rewards range from -1.0 to 2.0
            return np.clip(reward / 4.0, -0.5, 0.5)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
                0: Draw from deck
                1: Take from discard pile
                2-7: Replace card at position 0-5 with drawn card
                8: Discard drawn card (was previously 9)
                
        Returns:
            observation: The new game state
            reward: The reward for the action
            done: Whether the game is over
            info: Additional information
        """
        if self.game_over:
            return self._get_observation(), 0, True, {"error": "Game already over"}
            
        reward = 0
        info = {}
        
        # Increment turn counter
        self.turn_count += 1
        
        # Force end game if max turns reached
        if self.turn_count >= self.max_turns:
            self.game_over = True
            info["max_turns_reached"] = True
            return self._get_observation(), 0, True, info
            
        # Handle drawing phase
        if self.drawn_card is None:
            if action == 0:  # Draw from deck
                if self.deck:
                    self.drawn_card = self.deck.pop()
                    self.drawn_from_discard = False
                else:
                    # Reshuffle discard pile if deck is empty
                    self.deck = self.discard_pile[:-1]
                    self.discard_pile = [self.discard_pile[-1]]
                    random.shuffle(self.deck)
                    self.drawn_card = self.deck.pop()
                    self.drawn_from_discard = False
            elif action == 1:  # Take from discard pile
                if self.discard_pile:
                    self.drawn_card = self.discard_pile.pop()
                    self.drawn_from_discard = True
                else:
                    # Normalize invalid action penalty
                    return self._get_observation(), self._normalize_reward(-1), False, {"error": "Discard pile is empty"}
            else:
                # Normalize invalid action penalty
                return self._get_observation(), self._normalize_reward(-1), False, {"error": "Invalid action"}
        else:
            # Handle card replacement phase
            if 2 <= action <= 7:  # Replace card
                position = action - 2
                old_card = self.player_hands[self.current_player][position]
                self.player_hands[self.current_player][position] = self.drawn_card
                self.discard_pile.append(old_card)
                self.revealed_cards[self.current_player].add(position)
                
                # Check if this player has revealed all cards
                if len(self.revealed_cards[self.current_player]) == 6 and not self.final_round:
                    self.final_round = True
                    self.last_player = (self.current_player + self.num_players - 1) % self.num_players
                    info["final_round"] = True
                    info["trigger_player"] = self.current_player
                
            elif action == 8:  # Discard drawn card (was previously 9)
                if self.drawn_from_discard:
                    # Normalize invalid action penalty
                    return self._get_observation(), self._normalize_reward(-1), False, {"error": "Cannot discard a card taken from the discard pile"}
                self.discard_pile.append(self.drawn_card)
            else:
                # Normalize invalid action penalty
                return self._get_observation(), self._normalize_reward(-1), False, {"error": "Invalid action"}
                
            self.drawn_card = None
            
            # Move to next player
            self.current_player = (self.current_player + 1) % self.num_players
            
            # Check if game should end (when we've reached the last player in final round)
            if self.final_round and self.current_player == self.last_player:
                self.game_over = True
        
        # Calculate reward based on relative score improvement and face-up cards
        if self.game_over:
            scores = [self._calculate_score(p) for p in range(self.num_players)]
            min_score = min(scores)
            current_player_score = scores[self.current_player]
            
            # Simplified terminal rewards
            if current_player_score == min_score:
                # Winning is rewarded with a fixed value
                reward = 5.0
                
                # Reduced margin-of-victory bonus
                margin = sum(scores) / len(scores) - current_player_score
                reward += 0.3 * margin  # Reduced from 0.5 to 0.3
            else:
                # Simplified losing penalty based on score difference
                score_diff = current_player_score - min_score
                reward = -0.3 * score_diff
            
            info["scores"] = scores
            # Store raw reward in info dict for debugging
            info["raw_reward"] = reward
            # Normalize terminal reward
            reward = self._normalize_reward(reward, is_terminal=True)
        else:
            # Simplified intermediate rewards
            reward = 0.0
            
            # If this was a card replacement action (actions 2-7)
            if 2 <= action <= 7:
                position = action - 2
                
                # Get the old and new card values
                old_card = self.discard_pile[-1]  # The card that was just discarded (the old card)
                new_card = self.player_hands[self.current_player][position]  # The card that replaced it
                
                old_value = self._card_value(old_card)
                new_value = self._card_value(new_card)
                
                # Simplified reward for replacing a high-value card with a lower-value card
                value_improvement = old_value - new_value
                if value_improvement > 0:
                    reward += 0.5 * value_improvement
                
                # Check if this created a match in the column
                col = position % 3
                other_position = (position + 3) % 6  # The other position in the same column
                other_card = self.player_hands[self.current_player][other_position]
                
                # Simplified reward for creating a match (same rank)
                if new_card == other_card:
                    # Only reward for matches if both cards are revealed
                    if position in self.revealed_cards[self.current_player] and other_position in self.revealed_cards[self.current_player]:
                        # Reduced reward for creating matches
                        reward += 1.0  # Reduced from 2.0 to 1.0
                
                # Simplified reward for revealing cards
                if position not in self.revealed_cards[self.current_player]:
                    # Reduced reward for revealing cards
                    reward += 0.1  # Reduced from 0.3 to 0.1
                    
                    # Removed early-game bonus for revealing cards
            
            # Simplified penalty for discarding a card
            elif action == 8:
                # Only penalize discarding an obvious match
                discarded_rank = self.discard_pile[-1]
                for i in range(3):  # Check each column
                    # Check if discarded card could match any revealed card
                    top_pos, bottom_pos = i, i + 3
                    if top_pos in self.revealed_cards[self.current_player] and bottom_pos in self.revealed_cards[self.current_player]:
                        # Only penalize if both cards in column are revealed (obvious match)
                        if self.player_hands[self.current_player][top_pos] == discarded_rank or \
                           self.player_hands[self.current_player][bottom_pos] == discarded_rank:
                            reward -= 0.5  # Penalty for discarding an obvious match
            
            # Simplified score-based penalty - only consider revealed cards
            visible_score = 0
            hand = self.player_hands[self.current_player]
            
            # Calculate score only using revealed cards
            for col in range(3):
                top_pos, bottom_pos = col, col + 3
                top_revealed = top_pos in self.revealed_cards[self.current_player]
                bottom_revealed = bottom_pos in self.revealed_cards[self.current_player]
                
                # If both cards in column are revealed, check for matches
                if top_revealed and bottom_revealed:
                    if hand[top_pos] == hand[bottom_pos]:
                        # Matching cards cancel out
                        if hand[top_pos] == 1:  # Special case for 2s
                            visible_score -= 4
                        continue
                    else:
                        visible_score += self._card_value(hand[top_pos]) + self._card_value(hand[bottom_pos])
                # If only one card is revealed, add its value
                elif top_revealed:
                    visible_score += self._card_value(hand[top_pos])
                elif bottom_revealed:
                    visible_score += self._card_value(hand[bottom_pos])
                
            # Apply penalty based only on visible score
            reward -= 0.01 * visible_score
            
            # Store raw reward in info dict for debugging
            info["raw_reward"] = reward
            # Normalize intermediate reward
            reward = self._normalize_reward(reward)
        
        return self._get_observation(), reward, self.game_over, info
    
    def render(self):
        """Print a text representation of the game state."""
        print(f"Player {self.current_player}'s turn")
        
        for player in range(self.num_players):
            print(f"Player {player}'s hand:")
            hand = self.player_hands[player]
            revealed = self.revealed_cards[player]
            
            # Print top row
            for i in range(3):
                if player == self.current_player or i in revealed:
                    rank = hand[i]
                    rank_str = "A23456789TJQK"[rank]
                    print(f"{rank_str} ", end="")
                else:
                    print("# ", end="")
            print()
            
            # Print bottom row
            for i in range(3, 6):
                if player == self.current_player or i in revealed:
                    rank = hand[i]
                    rank_str = "A23456789TJQK"[rank]
                    print(f"{rank_str} ", end="")
                else:
                    print("# ", end="")
            print()
            
            print(f"Cards revealed: {len(revealed)}/6")
            print(f"Score: {self._calculate_score(player)}")
            print()
        
        if self.discard_pile:
            rank = self.discard_pile[-1]
            rank_str = "A23456789TJQK"[rank]
            print(f"Discard pile top: {rank_str}")
        
        if self.drawn_card is not None:
            rank = self.drawn_card
            rank_str = "A23456789TJQK"[rank]
            print(f"Drawn card: {rank_str}")
        
        if self.final_round:
            print("Final round in progress!")
            
        print(f"Deck size: {len(self.deck)}")
        print() 