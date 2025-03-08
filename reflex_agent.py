import numpy as np
import random
from typing import List, Dict

class ReflexAgent:
    """
    A reflex agent for the Golf card game that makes decisions based on
    the current game state using predefined heuristics.
    
    This version is designed to be more aggressive and take more risks,
    leading to fewer stalemates.
    """
    
    def __init__(self, player_id: int = 0):
        # Store player ID (0 or 1)
        self.player_id = player_id
        
        # Card values lookup (0=Ace, 1=2, ..., 12=King)
        # In Golf, LOWER scores are better
        self.card_values = {
            0: 1,    # Ace
            1: -2,   # 2 (special negative value)
            2: 3,    # 3
            3: 4,    # 4
            4: 5,    # 5
            5: 6,    # 6
            6: 7,    # 7
            7: 8,    # 8
            8: 9,    # 9
            9: 10,   # 10
            10: 10,  # Jack
            11: 10,  # Queen
            12: 10,  # King
            13: 5    # Unknown card - assume average value
        }
        
        # Expected value of an unknown card (average of all possible values)
        self.unknown_card_value = sum(self.card_values.values()) / len(self.card_values)
        
        # Value of a match (should be better than any single card improvement)
        self.match_value = -15  # Negative because lower scores are better
        
        # Aggressiveness factor (higher = more aggressive)
        self.aggressiveness = 0.8
        
        # Randomness factor (higher = more random decisions)
        self.randomness = 0.2
        
        # Turn counter to track game progress
        self.turn_counter = 0
        
        # Flag to track if we're in the end game
        self.end_game = False
    
    def _get_hand_and_revealed_indices(self) -> tuple[slice, slice]:
        """Get the correct indices for hand and revealed flags based on player_id."""
        if self.player_id == 0:
            return slice(0, 6), slice(14, 20)  # Player 0: hand 0-5, revealed 14-19
        else:
            return slice(6, 12), slice(20, 26)  # Player 1: hand 6-11, revealed 20-25
    
    def _evaluate_swap(self, hand: List[int], pos: int, new_card: int) -> float:
        """
        Evaluate the value of swapping a card at a given position.
        Returns the expected value change (negative is better in Golf).
        """
        # Convert float card values to integers
        old_card = int(hand[pos])
        pair_pos = (pos + 3) % 6 if pos < 3 else pos - 3
        pair_card = int(hand[pair_pos])
        new_card = int(new_card)
        
        # Calculate current value of the column
        current_value = 0
        if old_card == pair_card and old_card != 13:  # Only count matches for known cards
            if old_card == 1:  # Special case for 2s
                current_value = -4  # Matching 2s are worth -4
            else:
                current_value = 0  # Matching cards cancel out
        else:
            # Handle unknown cards in current value calculation
            old_card_value = self.card_values[old_card]
            pair_card_value = self.card_values[pair_card]
            current_value = old_card_value + pair_card_value
        
        # Calculate new value if we make the swap
        new_value = 0
        if new_card == pair_card and new_card != 13:  # Only count matches for known cards
            if new_card == 1:  # Special case for 2s
                new_value = -4  # Matching 2s are worth -4
            else:
                new_value = self.match_value  # Strong bonus for making a match
        else:
            # Handle unknown cards in new value calculation
            new_card_value = self.card_values[new_card]
            pair_card_value = self.card_values[pair_card]
            new_value = new_card_value + pair_card_value
        
        # Return change in value (negative is better)
        # If new_value is less than current_value, this will be negative (good)
        # Apply aggressiveness factor to encourage more swaps
        value_change = new_value - current_value
        
        # Increase aggressiveness as the game progresses
        turn_factor = min(1.0, self.turn_counter / 20.0) * 0.2
        effective_aggressiveness = self.aggressiveness + turn_factor
        
        return value_change * (1.0 - effective_aggressiveness)
    
    def _should_draw_from_discard(self, state: np.ndarray, valid_actions: List[int]) -> bool:
        """Decide whether to draw from the discard pile."""
        # Add randomness to decision
        if random.random() < self.randomness:
            return random.choice([True, False])
            
        # Extract the discard card (index 12 in the observation)
        discard_card = int(state[12])  # Discard pile top card is in position 12
        
        # Get correct indices for this player's hand and revealed flags
        hand_slice, revealed_slice = self._get_hand_and_revealed_indices()
        
        # Convert hand to integers and mask unrevealed cards as 13
        hand = []
        revealed_count = 0
        for i in range(6):
            card = int(state[hand_slice][i])
            revealed = bool(state[revealed_slice][i])
            if revealed:
                revealed_count += 1
            hand.append(card if revealed else 13)
        
        # Always take a 2 (-2 points)
        if discard_card == 1:
            return True
            
        # Check if discard would make any matches
        for i in range(6):
            if hand[i] == 13:  # Skip unrevealed cards
                continue
            pair_pos = (i + 3) % 6 if i < 3 else i - 3
            if hand[pair_pos] == discard_card:  # Would make a match
                return True
        
        # In end game (most cards revealed), be more selective
        if revealed_count >= 4:
            self.end_game = True
            # Only take very good cards in end game
            return self.card_values[discard_card] <= 1  # Only take Aces and 2s
        
        # If no matches possible, take low value cards
        # Be more selective about taking cards from discard to encourage more deck draws
        card_value = self.card_values[discard_card]
        
        # Dynamic threshold based on turn counter and aggressiveness
        base_threshold = 3
        turn_adjustment = min(2.0, self.turn_counter / 10.0)
        threshold = base_threshold - turn_adjustment - (random.random() * self.aggressiveness * 2)
        
        return card_value <= threshold
    
    def _calculate_hand_score(self, hand: List[int]) -> int:
        """Calculate the score of the current hand."""
        score = 0
        # Calculate score for each column
        for i in range(3):
            card1 = hand[i]
            card2 = hand[i + 3]
            
            # If cards match, they cancel out (except 2s which are -2 each)
            if card1 == card2 and card1 != 13:
                if card1 == 1:  # 2s
                    score += -4  # Two 2s are worth -4
                else:
                    score += 0  # Matching cards cancel out
            else:
                # Add individual card values
                score += self.card_values[card1] if card1 != 13 else self.unknown_card_value
                score += self.card_values[card2] if card2 != 13 else self.unknown_card_value
        
        return score
    
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Select an action based on the current game state using heuristics.
        
        Actions:
        0: Draw from deck
        1: Draw from discard pile
        2-7: Replace card at position 0-5 with drawn card
        8: Discard drawn card
        """
        # Increment turn counter
        self.turn_counter += 1
            
        # Get correct indices for this player's hand and revealed flags
        hand_slice, revealed_slice = self._get_hand_and_revealed_indices()
        
        # Convert hand to integers and mask unrevealed cards as 13
        hand = []
        revealed = []
        for i in range(6):
            card = int(state[hand_slice][i])
            is_revealed = bool(state[revealed_slice][i])
            hand.append(card if is_revealed else 13)
            revealed.append(is_revealed)
        
        # Check if we're in the end game (most cards revealed)
        revealed_count = sum(revealed)
        if revealed_count >= 4:
            self.end_game = True
        
        # If we haven't drawn a card yet
        if 0 in valid_actions and 1 in valid_actions:
            # More likely to draw from deck to increase variance
            # Increase deck drawing probability as game progresses
            deck_draw_prob = self.aggressiveness * 0.5
            if self.end_game:
                deck_draw_prob = self.aggressiveness * 0.3  # Be more conservative in end game
            
            if random.random() < deck_draw_prob:
                return 0  # Draw from deck
            
            if self._should_draw_from_discard(state, valid_actions):
                return 1  # Draw from discard
            return 0  # Draw from deck
        
        # If we need to decide whether to keep or swap a drawn card
        if 8 in valid_actions:  # Can only discard if we drew from deck
            drawn_card = int(state[13])  # Drawn card is in position 13
            
            best_swap_value = float('inf')  # Initialize to worst possible value
            best_swap_pos = None
            
            # Evaluate all possible swaps
            for pos in range(6):
                action = pos + 2  # Convert position to action (2-7)
                if action in valid_actions:
                    swap_value = self._evaluate_swap(hand, pos, drawn_card)
                    
                    # Prefer swapping unrevealed cards if the value is similar
                    if not revealed[pos]:
                        swap_value -= 1.0  # Stronger preference for unrevealed cards
                        
                    # Be more eager to swap high-value cards (10, J, Q, K) since they're worth more points
                    if hand[pos] >= 9:  # 10, J, Q, K are all worth 10 points
                        swap_value -= self.aggressiveness * 1.5  # Stronger incentive to swap high-value cards
                        
                    if swap_value < best_swap_value:  # Lower is better
                        best_swap_value = swap_value
                        best_swap_pos = action
            
            # More aggressive threshold for swapping
            # Even make slightly negative swaps sometimes to increase variance
            base_threshold = 0.5
            turn_adjustment = min(1.0, self.turn_counter / 15.0)
            swap_threshold = base_threshold - turn_adjustment - (self.aggressiveness * 2.0)
            
            # In end game, be more selective about swaps
            if self.end_game:
                swap_threshold = -0.5  # Only make clearly beneficial swaps in end game
            
            # Make the swap if it meets our threshold
            if best_swap_value < swap_threshold and best_swap_pos is not None:
                return best_swap_pos
            
            # Sometimes make a random swap even if it doesn't meet the threshold
            # Less likely to do this in end game
            random_swap_prob = self.aggressiveness * 0.3
            if self.end_game:
                random_swap_prob = self.aggressiveness * 0.1
                
            if random.random() < random_swap_prob and best_swap_pos is not None:
                return best_swap_pos
            
            # Otherwise, discard if possible
            if 8 in valid_actions:
                return 8
        
        # If we're in an unexpected state, take the first valid action
        return valid_actions[0] 