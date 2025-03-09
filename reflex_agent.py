import numpy as np
import random
from typing import List, Dict, Tuple

class ReflexAgent:
    """
    A deterministic reflex agent for the Golf card game that makes decisions based on
    the current game state using strong strategic heuristics.
    
    This version focuses on optimal card management with minimal randomness.
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
            13: 6    # Unknown card - assume slightly above average value
        }
        
        # Priorities for card ranks (lower is better)
        self.card_priorities = {
            1: 1,    # 2 (highest priority)
            0: 2,    # Ace
            2: 3,    # 3
            3: 4,    # 4
            4: 5,    # 5
            5: 6,    # 6
            6: 7,    # 7
            7: 8,    # 8
            8: 9,    # 9
            9: 10,   # 10
            10: 11,  # Jack
            11: 12,  # Queen
            12: 13,  # King (lowest priority)
        }
        
        # Expected value of an unknown card (weighted average based on actual distribution)
        self.unknown_card_value = 6.0
        
        # Value of a match (should be better than any single card improvement)
        self.match_value = -20  # Negative because lower scores are better
        
        # Bonus for a pair of 2s
        self.twos_pair_bonus = -10
        
        # Minimal randomness for tiebreaking only
        self.randomness = 0.05
        
        # Turn counter to track game progress
        self.turn_counter = 0
        
        # Flag to track if we're in the end game
        self.end_game = False
        
        # Game state tracking
        self.opponent_score = None
        self.my_score = None
        self.cards_in_deck = 52  # Initial estimate
    
    def _get_hand_and_revealed_indices(self) -> Tuple[slice, slice]:
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
        # Get card values
        old_card = int(hand[pos])
        pair_pos = (pos + 3) % 6 if pos < 3 else pos - 3
        pair_card = int(hand[pair_pos])
        new_card = int(new_card)
        
        # Calculate current value
        current_value = self._calculate_column_value(old_card, pair_card)
        
        # Calculate new value if we make the swap
        new_value = self._calculate_column_value(new_card, pair_card)
        
        # Calculate the difference (negative is better)
        value_change = new_value - current_value
        
        # Add bonus for creating matches with special cards
        if new_card == pair_card:
            if new_card == 1:  # Pair of 2s
                value_change -= 2  # Extra incentive for 2s
            elif new_card <= 2:  # Pair of Aces or 3s
                value_change -= 1  # Slightly more incentive for low cards
        
        # Add penalty for breaking existing matches
        if old_card == pair_card and old_card != 13:
            if old_card == 1:  # Breaking a pair of 2s
                value_change += 5  # Strong penalty
            else:  # Breaking any other pair
                value_change += 3  # Moderate penalty
        
        # Prioritize replacing high-value cards
        if old_card >= 9 and old_card != 13:  # Face cards or 10
            value_change -= 0.5  # Extra incentive
            
        # Extra incentive for replacing unknown cards
        if old_card == 13:
            value_change -= 1.0
            
        return value_change
    
    def _calculate_column_value(self, card1: int, card2: int) -> float:
        """Calculate the score value of a column with two cards."""
        # Handle unknown cards
        card1_value = self.card_values[card1] if card1 != 13 else self.unknown_card_value
        card2_value = self.card_values[card2] if card2 != 13 else self.unknown_card_value
        
        # If both cards are known and matching
        if card1 == card2 and card1 != 13:
            if card1 == 1:  # Pair of 2s
                return -4  # Special rule for 2s
            return 0  # Regular matches cancel out
            
        # Regular case - sum of card values
        return card1_value + card2_value
    
    def _is_good_discard_card(self, card: int, hand: List[int], revealed: List[bool]) -> bool:
        """Determine if a card would be good to draw from the discard pile."""
        # Always take a 2 (-2 points)
        if card == 1:
            return True
            
        # Always take an Ace (1 point) in early game
        if card == 0 and self.turn_counter < 8:
            return True
        
        # Check if it would create a match with any revealed card
        for i in range(6):
            if not revealed[i]:
                continue
                
            pair_pos = (i + 3) % 6 if i < 3 else i - 3
            if hand[i] == card:
                # Would create a match
                return True
                
            # Check if it would improve a high-value position
            if hand[i] >= 9:  # Face card or 10
                card_value = self.card_values[card]
                current_value = self.card_values[hand[i]]
                if card_value < current_value - 2:  # Significant improvement
                    return True
        
        # Determine threshold based on game stage
        card_value = self.card_values[card]
        if self.end_game:
            # Very strict in end game
            return card_value <= 1  # Only Aces and 2s
        elif self.turn_counter > 10:
            # Stricter in mid-game
            return card_value <= 3  # Up to 3s
        else:
            # More lenient in early game
            return card_value <= 5  # Up to 5s
    
    def _should_draw_from_discard(self, state: np.ndarray, valid_actions: List[int], 
                                 hand: List[int], revealed: List[bool]) -> bool:
        """Decide whether to draw from the discard pile."""
        # Minimal randomness for exploration only in early game
        if self.turn_counter < 5 and random.random() < self.randomness:
            return random.choice([True, False])
            
        # Extract the discard card (index 12 in the observation)
        discard_card = int(state[12])  # Discard pile top card is in position 12
        
        return self._is_good_discard_card(discard_card, hand, revealed)
    
    def _calculate_hand_score(self, hand: List[int], revealed: List[bool]) -> int:
        """Calculate the score of the current hand."""
        score = 0
        # Calculate score for each column
        for i in range(3):
            card1_pos = i
            card2_pos = i + 3
            
            # Use revealed information if available
            card1 = hand[card1_pos] if revealed[card1_pos] else 13
            card2 = hand[card2_pos] if revealed[card2_pos] else 13
            
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
    
    def _evaluate_all_possible_swaps(self, hand: List[int], revealed: List[bool], drawn_card: int, 
                                     valid_actions: List[int]) -> Tuple[int, float]:
        """Evaluate all possible swaps and return the best one with its value."""
        best_swap_value = float('inf')
        best_swap_pos = None
        
        # Evaluate all possible swaps
        for pos in range(6):
            action = pos + 2  # Convert position to action (2-7)
            if action in valid_actions:
                swap_value = self._evaluate_swap(hand, pos, drawn_card)
                
                # Prefer swapping unrevealed cards
                if not revealed[pos]:
                    swap_value -= 1.5  # Strong preference for unrevealed cards
                    
                if swap_value < best_swap_value:  # Lower is better
                    best_swap_value = swap_value
                    best_swap_pos = action
        
        return best_swap_pos, best_swap_value
    
    def _should_swap_card(self, swap_value: float) -> bool:
        """Determine if a swap should be made based on its value and game stage."""
        # Always make beneficial swaps
        if swap_value < -0.5:
            return True
            
        # In early game, be more aggressive with swaps
        if self.turn_counter < 8:
            return swap_value < 1.0
            
        # In mid game, be somewhat selective
        if not self.end_game:
            return swap_value < 0.2
            
        # In end game, be very selective
        return swap_value < -1.0
    
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Select an action based on the current game state using deterministic heuristics.
        
        Actions:
        0: Draw from deck
        1: Draw from discard pile
        2-7: Replace card at position 0-5 with drawn card
        8: Discard drawn card
        """
        if not valid_actions:
            raise ValueError("No valid actions provided")
            
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
            
        # Check for final round indicator (third last element in state)
        if len(state) >= 28 and state[-2] == 1.0:
            self.end_game = True
        
        # If we haven't drawn a card yet (draw phase)
        if 0 in valid_actions and 1 in valid_actions:
            # Calculate hand score for strategic decisions
            self.my_score = self._calculate_hand_score(hand, revealed)
            
            if self._should_draw_from_discard(state, valid_actions, hand, revealed):
                return 1  # Draw from discard pile
            else:
                return 0  # Draw from deck
        
        # If we need to decide whether to keep or swap a drawn card (swap phase)
        if 8 in valid_actions:  # Can only discard if we drew from deck
            drawn_card = int(state[13])  # Drawn card is in position 13
            
            # Get the best possible swap
            best_swap_pos, best_swap_value = self._evaluate_all_possible_swaps(hand, revealed, drawn_card, valid_actions)
            
            # Decide whether to swap or discard
            if best_swap_pos is not None and self._should_swap_card(best_swap_value):
                return best_swap_pos
            
            # Otherwise, discard
            return 8
        
        # If we're in an unexpected state or only one action is valid, take the first valid action
        return valid_actions[0] 