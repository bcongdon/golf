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
        
        The observation includes:
        - Current player's hand (with hidden cards masked - only shows revealed cards)
        - Opponent's visible cards
        - Top card of the discard pile
        - Card positions and revealed flags
        - Whether the game is in final round
        
        Simplified to only include rank information (not suits) since suits don't matter for scoring.
        """
        # New observation space:
        # 13 ranks for player's cards + 13 ranks for opponent's cards + 
        # 6 positions + 6 revealed flags + 6 opponent revealed flags + 
        # 13 ranks for discard pile + 1 discard indicator + 1 drawn card indicator + 
        # 13 ranks for drawn card + 1 final round indicator
        # Total: 60 dimensions (reduced from 119)
        obs = np.zeros((60,), dtype=np.float32)
        
        # Encode current player's hand (only revealed cards)
        for i, card in enumerate(self.player_hands[self.current_player]):
            # Mark position
            obs[26 + i] = 1.0
            
            # Only include card value if it's revealed
            if i in self.revealed_cards[self.current_player]:
                _, rank = card  # Ignore suit, only use rank
                obs[rank] = 1.0  # Set card in the first 13 positions
                obs[32 + i] = 1.0  # Mark as revealed
        
        # Encode opponent's visible cards
        opponent = (self.current_player + 1) % self.num_players
        for i, card in enumerate(self.player_hands[opponent]):
            # Only include opponent's revealed cards
            if i in self.revealed_cards[opponent]:
                _, rank = card  # Ignore suit, only use rank
                # Store opponent cards in second 13 positions
                obs[13 + rank] = 1.0
                # Mark opponent card as revealed
                obs[38 + i] = 1.0
                
        # Encode discard pile top card
        if self.discard_pile:
            _, rank = self.discard_pile[-1]  # Ignore suit
            obs[44] = 1.0  # Set flag that discard pile has a card
            obs[45 + rank] = 1.0  # Store rank in dedicated discard section
        
        # Encode if drawn card exists
        if self.drawn_card is not None:
            _, rank = self.drawn_card  # Ignore suit
            obs[58] = 1.0  # Flag indicating drawn card exists
            obs[rank] = 1.0  # Show drawn card in player's card section
            
        # Encode if in final round
        if self.final_round:
            obs[59] = 1.0
            
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
            
            # Reward is positive if player has lowest score (or tied for lowest)
            if current_player_score == min_score:
                reward = 10.0
            else:
                # Negative reward based on how far from winning
                reward = -1.0 * (current_player_score - min_score)
            
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
                
                # Reward for replacing a high-value card with a lower-value card
                value_improvement = old_value - new_value
                reward += 0.2 * value_improvement
                
                # Check if this created a match in the column
                col = position % 3
                other_position = (position + 3) % 6  # The other position in the same column
                other_card = self.player_hands[self.current_player][other_position]
                
                # Extra reward for creating a match (same rank)
                if new_card[1] == other_card[1]:
                    reward += 1.0
                    
                    # Even more reward if it's a 2 (worth -2 points)
                    if new_card[1] == 1:  # 2 is rank 1 (0-indexed)
                        reward += 0.5
                
                # Reward for revealing cards (encouraging exploration)
                if position not in self.revealed_cards[self.current_player]:
                    reward += 0.1
            
            # Small penalty for discarding a potentially useful card
            elif action == 8:
                card_value = self._card_value(self.discard_pile[-1])
                if card_value <= 5:  # If it's a low-value card (A, 2, 3, 4, 5)
                    reward -= 0.1 * (6 - card_value)  # More penalty for lower values
            
            # Apply a small penalty proportional to the current score
            # This encourages the agent to keep the score low throughout the game
            current_score = self._calculate_score(self.current_player)
            reward -= 0.01 * current_score
        
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