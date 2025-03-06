import numpy as np
from typing import List, Dict

class ReflexAgent:
    """
    A simple reflex agent for the Golf card game that makes decisions based on
    the current game state using predefined heuristics.
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
        return new_value - current_value
    
    def _should_draw_from_discard(self, state: np.ndarray, valid_actions: List[int]) -> bool:
        """Decide whether to draw from the discard pile."""
        # Extract the discard card (index 12 in the observation)
        discard_card = int(state[12])  # Discard pile top card is in position 12
        
        # Get correct indices for this player's hand and revealed flags
        hand_slice, revealed_slice = self._get_hand_and_revealed_indices()
        
        # Convert hand to integers and mask unrevealed cards as 13
        hand = []
        for i in range(6):
            card = int(state[hand_slice][i])
            revealed = bool(state[revealed_slice][i])
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
        
        # If no matches possible, take low value cards
        card_value = self.card_values[discard_card]
        return card_value <= 4  # Take cards worth 4 or less
    
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Select an action based on the current game state using heuristics.
        
        Actions:
        0: Draw from deck
        1: Draw from discard pile
        2-7: Replace card at position 0-5 with drawn card
        8: Discard drawn card
        """
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
        
        # If we haven't drawn a card yet
        if 0 in valid_actions and 1 in valid_actions:
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
                        swap_value -= 0.5
                    if swap_value < best_swap_value:  # Lower is better
                        best_swap_value = swap_value
                        best_swap_pos = action
            
            # Make the swap if it improves our position
            if best_swap_value < -0.5 and best_swap_pos is not None:
                return best_swap_pos
            
            # Otherwise, discard if possible
            if 8 in valid_actions:
                return 8
        
        # If we're in an unexpected state, take the first valid action
        return valid_actions[0] 