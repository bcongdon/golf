import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union

class GolfGame:
    """
    Implementation of the Golf card game environment.
    
    In this version of Golf:
    - Each player has a 2x3 grid of cards (6 cards total)
    - Cards are standard deck (A-K in four suits)
    - Aces are worth 1, number cards are face value, face cards (J,Q,K) are worth 10
    - Matching cards in the same column cancel out (worth 0)
    - The goal is to have the lowest total score
    """
    
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state and return the initial observation."""
        # Initialize a standard deck of cards
        self.deck = []
        for suit in range(4):  # 0-3 for clubs, diamonds, hearts, spades
            for rank in range(13):  # 0-12 for A-K
                self.deck.append((suit, rank))
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
    
    def _card_value(self, card: Tuple[int, int]) -> int:
        """Calculate the value of a card."""
        _, rank = card
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
            if card1[1] == card2[1]:
                # Special case for 2s
                if card1[1] == 1:
                    score -= 4
                continue
            else:
                score += self._card_value(card1) + self._card_value(card2)
        
        return score
    
    def _get_observation(self) -> np.ndarray:
        """
        Convert the game state to a neural network input representation.
        
        Each card is represented as a 3-element vector:
        - Rank (1-13): Ace=1, 2=2, ..., King=13 (normalized to [0,1])
        - Wild Flag (0 or 1): 1 if the card is a 2 (wild), otherwise 0
        - Point Value (-2 to 10): The actual score value of the card in the game (normalized to [0,1])
        
        The observation includes:
        - Current player's hand (6 cards * 3 values = 18 elements)
        - Opponent's visible cards (6 cards * 3 values = 18 elements)
        - Discard pile top card (3 elements)
        - Drawn card (3 elements)
        - Revealed flags for player (6 elements)
        - Revealed flags for opponent (6 elements)
        - Game state flags (2 elements: drawn_from_discard, final_round)
        
        Total: 56 dimensions
        """
        # Initialize observation array
        obs = np.zeros((56,), dtype=np.float32)
        
        # Helper function to encode a single card with normalization
        def encode_card(card, start_idx):
            suit, rank = card
            # Convert from 0-indexed to 1-indexed rank and normalize to [0,1]
            # Rank range: 1-13 -> normalized to [0,1]
            rank_value = (rank + 1) / 13.0
            
            # Set wild flag (1 if it's a 2, which is rank 1 in 0-indexed)
            wild_flag = 1.0 if rank == 1 else 0.0
            
            # Calculate point value
            if rank == 0:  # Ace
                point_value = 1.0
            elif rank == 1:  # 2 (wild)
                point_value = -2.0
            elif rank < 10:  # 3-10
                point_value = float(rank + 1)
            else:  # Face cards (J, Q, K)
                point_value = 10.0
                
            # Normalize point value from [-2,10] to [0,1]
            normalized_point_value = (point_value + 2) / 12.0
                
            # Set the values in the observation array
            obs[start_idx] = rank_value
            obs[start_idx + 1] = wild_flag
            obs[start_idx + 2] = normalized_point_value
        
        # Encode current player's hand
        for i, card in enumerate(self.player_hands[self.current_player]):
            start_idx = i * 3
            # For revealed cards, encode the actual card
            if i in self.revealed_cards[self.current_player]:
                encode_card(card, start_idx)
                # Set revealed flag
                obs[36 + i] = 1.0
            # For hidden cards, leave as zeros (unknown)
        
        # Encode opponent's visible cards
        opponent = (self.current_player + 1) % self.num_players
        for i, card in enumerate(self.player_hands[opponent]):
            # Only encode revealed cards
            if i in self.revealed_cards[opponent]:
                start_idx = 18 + (i * 3)  # Start after player's 18 elements
                encode_card(card, start_idx)
                # Set revealed flag
                obs[42 + i] = 1.0
        
        # Encode discard pile top card
        if self.discard_pile:
            encode_card(self.discard_pile[-1], 48)
        
        # Encode drawn card
        if self.drawn_card is not None:
            encode_card(self.drawn_card, 51)
            
            # Set drawn from discard flag
            if self.drawn_from_discard:
                obs[54] = 1.0
        
        # Set final round flag
        if self.final_round:
            obs[55] = 1.0
            
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
                    return self._get_observation(), -1, False, {"error": "Discard pile is empty"}
            else:
                return self._get_observation(), -1, False, {"error": "Invalid action"}
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
                    return self._get_observation(), -1, False, {"error": "Cannot discard a card taken from the discard pile"}
                self.discard_pile.append(self.drawn_card)
            else:
                return self._get_observation(), -1, False, {"error": "Invalid action"}
                
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
            
            # Enhanced terminal rewards
            if current_player_score == min_score:
                # Winning is highly rewarded
                if len(self.revealed_cards[self.current_player]) == 6:
                    # Extra reward for winning with all cards revealed
                    reward = 15.0
                else:
                    reward = 10.0
                
                # Additional reward based on margin of victory
                margin = sum(scores) / len(scores) - current_player_score
                reward += 0.5 * margin  # Bigger wins are better
            else:
                # Negative reward based on how far from winning, but less punishing
                score_diff = current_player_score - min_score
                reward = -0.5 * score_diff
                
                # Less penalty if the player revealed more cards (encouraging exploration)
                revealed_ratio = len(self.revealed_cards[self.current_player]) / 6.0
                reward *= (1.0 - 0.3 * revealed_ratio)
            
            info["scores"] = scores
        else:
            # Enhanced intermediate rewards
            reward = 0.0
            
            # If this was a card replacement action (actions 2-7)
            if 2 <= action <= 7:
                position = action - 2
                
                # Get the old and new card values
                old_card = self.discard_pile[-1]  # The card that was just discarded (the old card)
                new_card = self.player_hands[self.current_player][position]  # The card that replaced it
                
                old_value = self._card_value(old_card)
                new_value = self._card_value(new_card)
                
                # Stronger reward for replacing a high-value card with a lower-value card
                value_improvement = old_value - new_value
                reward += 0.5 * value_improvement  # Increased from 0.2 to 0.5
                
                # Check if this created a match in the column
                col = position % 3
                other_position = (position + 3) % 6  # The other position in the same column
                other_card = self.player_hands[self.current_player][other_position]
                
                # Extra reward for creating a match (same rank)
                if new_card[1] == other_card[1]:
                    # Significantly higher reward for creating matches
                    reward += 2.0  # Increased from 1.0 to 2.0
                    
                    # Even more reward if it's a 2 (worth -2 points)
                    if new_card[1] == 1:  # 2 is rank 1 (0-indexed)
                        reward += 1.0  # Increased from 0.5 to 1.0
                
                # Reward for revealing cards (encouraging exploration)
                if position not in self.revealed_cards[self.current_player]:
                    # Higher reward for revealing cards
                    reward += 0.3  # Increased from 0.1 to 0.3
                    
                    # Extra reward for revealing cards early in the game
                    if len(self.revealed_cards[self.current_player]) < 3:
                        reward += 0.2  # Additional reward for early exploration
                
                # Reward for strategic play - replacing cards in columns with high total value
                if other_position in self.revealed_cards[self.current_player]:
                    # If the other card in column is revealed, reward replacing high-value columns
                    column_value = self._card_value(new_card) + self._card_value(other_card)
                    if column_value > 10:  # If column has high value
                        reward += 0.2  # Reward for targeting high-value columns
            
            # Penalty for discarding a potentially useful card
            elif action == 8:
                card_value = self._card_value(self.discard_pile[-1])
                if card_value <= 5:  # If it's a low-value card (A, 2, 3, 4, 5)
                    reward -= 0.2 * (6 - card_value)  # Increased penalty for discarding valuable cards
                
                # Additional penalty for discarding a card that could create a match
                discarded_rank = self.discard_pile[-1][1]
                for i in range(3):  # Check each column
                    # Check if discarded card could match any revealed card
                    top_pos, bottom_pos = i, i + 3
                    if top_pos in self.revealed_cards[self.current_player]:
                        if self.player_hands[self.current_player][top_pos][1] == discarded_rank:
                            reward -= 0.5  # Penalty for discarding a potential match
                    if bottom_pos in self.revealed_cards[self.current_player]:
                        if self.player_hands[self.current_player][bottom_pos][1] == discarded_rank:
                            reward -= 0.5  # Penalty for discarding a potential match
            
            # Apply a penalty proportional to the current score
            # This encourages the agent to keep the score low throughout the game
            current_score = self._calculate_score(self.current_player)
            reward -= 0.02 * current_score  # Increased from 0.01 to 0.02
            
            # Reward for making progress in revealing cards
            revealed_count = len(self.revealed_cards[self.current_player])
            if revealed_count >= 4 and not self.final_round:
                # Encourage revealing more cards as game progresses
                reward += 0.1 * revealed_count
        
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
                    suit, rank = hand[i]
                    rank_str = "A23456789TJQK"[rank]
                    suit_str = "♣♦♥♠"[suit]
                    print(f"{rank_str}{suit_str} ", end="")
                else:
                    print("## ", end="")
            print()
            
            # Print bottom row
            for i in range(3, 6):
                if player == self.current_player or i in revealed:
                    suit, rank = hand[i]
                    rank_str = "A23456789TJQK"[rank]
                    suit_str = "♣♦♥♠"[suit]
                    print(f"{rank_str}{suit_str} ", end="")
                else:
                    print("## ", end="")
            print()
            
            print(f"Cards revealed: {len(revealed)}/6")
            print(f"Score: {self._calculate_score(player)}")
            print()
        
        if self.discard_pile:
            suit, rank = self.discard_pile[-1]
            rank_str = "A23456789TJQK"[rank]
            suit_str = "♣♦♥♠"[suit]
            print(f"Discard pile top: {rank_str}{suit_str}")
        
        if self.drawn_card:
            suit, rank = self.drawn_card
            rank_str = "A23456789TJQK"[rank]
            suit_str = "♣♦♥♠"[suit]
            print(f"Drawn card: {rank_str}{suit_str}")
        
        if self.final_round:
            print("Final round in progress!")
            
        print(f"Deck size: {len(self.deck)}")
        print() 