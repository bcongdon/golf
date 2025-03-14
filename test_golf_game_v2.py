import unittest
import numpy as np
from golf_game_v2 import GolfGame, GameConfig, Action, Card, GameInfo

class TestGolfGameV2(unittest.TestCase):
    def setUp(self):
        """Set up a new game instance before each test."""
        self.config = GameConfig(
            num_players=2,
            grid_rows=2,
            grid_cols=3,
            initial_revealed=2,
            max_turns=100,
            normalize_rewards=True,
            copies_per_rank=4
        )
        self.game = GolfGame(self.config)

    def test_initialization(self):
        """Test game initialization and configuration."""
        self.assertEqual(self.game.config.num_players, 2)
        self.assertEqual(self.game.config.grid_rows, 2)
        self.assertEqual(self.game.config.grid_cols, 3)
        
        # Test initial game state
        self.assertFalse(self.game.game_over)
        self.assertEqual(self.game.current_player, 0)
        self.assertIsNone(self.game.drawn_card)
        self.assertFalse(self.game.final_round)
        self.assertEqual(self.game.turn_count, 0)

        # Test deck initialization
        total_cards = len(Card) * self.config.copies_per_rank
        cards_dealt = self.config.num_players * self.config.grid_rows * self.config.grid_cols + 1  # +1 for discard
        self.assertEqual(len(self.game.deck), total_cards - cards_dealt)

        # Test player hands
        for hand in self.game.player_hands:
            self.assertEqual(len(hand), self.config.grid_rows * self.config.grid_cols)

        # Test revealed cards
        for revealed in self.game.revealed_cards:
            self.assertEqual(len(revealed), self.config.initial_revealed)

    def test_card_values(self):
        """Test card value calculations."""
        test_cases = [
            (Card.ACE, 1),
            (Card.TWO, -2),
            (Card.THREE, 3),
            (Card.FOUR, 4),
            (Card.FIVE, 5),
            (Card.SIX, 6),
            (Card.SEVEN, 7),
            (Card.EIGHT, 8),
            (Card.NINE, 9),
            (Card.TEN, 10),
            (Card.JACK, 10),
            (Card.QUEEN, 10),
            (Card.KING, 10),
        ]
        for card, expected_value in test_cases:
            self.assertEqual(card.point_value, expected_value)

    def test_card_symbols(self):
        """Test card symbol representations."""
        test_cases = [
            (Card.ACE, 'A'),
            (Card.TWO, '2'),
            (Card.TEN, 'T'),
            (Card.JACK, 'J'),
            (Card.QUEEN, 'Q'),
            (Card.KING, 'K'),
        ]
        for card, expected_symbol in test_cases:
            self.assertEqual(card.symbol, expected_symbol)

    def test_valid_actions(self):
        """Test valid action generation in different game states."""
        # Initial valid actions
        valid_actions = self.game._get_valid_actions()
        self.assertEqual(set(valid_actions), {Action.DRAW_FROM_DECK, Action.DRAW_FROM_DISCARD})

        # After drawing from deck
        self.game.drawn_card = Card.ACE
        self.game.drawn_from_discard = False
        valid_actions = self.game._get_valid_actions()
        expected_actions = set(range(Action.REPLACE_0, Action.DISCARD + 1))
        self.assertEqual(set(valid_actions), expected_actions)

        # After drawing from discard
        self.game.drawn_from_discard = True
        valid_actions = self.game._get_valid_actions()
        expected_actions = set(range(Action.REPLACE_0, Action.REPLACE_5 + 1))
        self.assertEqual(set(valid_actions), expected_actions)

        # When game is over
        self.game.game_over = True
        valid_actions = self.game._get_valid_actions()
        self.assertEqual(len(valid_actions), 0)

    def test_score_calculation(self):
        """Test score calculation with various hand configurations."""
        # Test matching cards
        self.game.player_hands[0] = [
            Card.ACE, Card.FIVE, Card.JACK,
            Card.ACE, Card.SIX, Card.JACK
        ]
        score = self.game._calculate_score(0)
        # ACE matches ACE (0), FIVE(5)+SIX(6)=11, JACK matches JACK (0)
        self.assertEqual(score, 11)

        # Test matching twos
        self.game.player_hands[0] = [
            Card.TWO, Card.FIVE, Card.JACK,
            Card.TWO, Card.SIX, Card.JACK
        ]
        score = self.game._calculate_score(0)
        # TWO matches TWO (-4), FIVE(5)+SIX(6)=11, JACK matches JACK (0)
        self.assertEqual(score, 7)

        # Test no matches
        self.game.player_hands[0] = [
            Card.ACE, Card.TWO, Card.THREE,
            Card.FOUR, Card.FIVE, Card.SIX
        ]
        score = self.game._calculate_score(0)
        # ACE=1, TWO=-2, THREE=3, FOUR=4, FIVE=5, SIX=6
        self.assertEqual(score, 17)

    def test_step_draw_actions(self):
        """Test draw actions and their effects."""
        # Test drawing from deck
        obs, reward, done, info = self.game.step(Action.DRAW_FROM_DECK)
        self.assertIsNotNone(self.game.drawn_card)
        self.assertFalse(self.game.drawn_from_discard)
        self.assertFalse(done)
    
        # Reset and test drawing from discard
        self.game.reset()
        obs, reward, done, info = self.game.step(Action.DRAW_FROM_DISCARD)
        self.assertIsNotNone(self.game.drawn_card)
        self.assertTrue(self.game.drawn_from_discard)
        self.assertFalse(done)
    
        # Test invalid action
        self.game.reset()
        obs, reward, done, info = self.game.step(Action.DISCARD)
        self.assertEqual(reward, -0.25)  # Normalized penalty
        self.assertNotEqual(info.error, "")

    def test_step_replace_actions(self):
        """Test card replacement actions."""
        # Draw a card first
        self.game.step(Action.DRAW_FROM_DECK)
        drawn_card = self.game.drawn_card
        
        # Replace card at position 0
        old_card = self.game.player_hands[0][0]
        obs, reward, done, info = self.game.step(Action.REPLACE_0)
        
        # Verify card replacement
        self.assertEqual(self.game.player_hands[0][0], drawn_card)
        self.assertEqual(self.game.discard_pile[-1], old_card)
        self.assertIsNone(self.game.drawn_card)
        self.assertEqual(self.game.current_player, 1)

    def test_final_round_trigger(self):
        """Test final round triggering conditions."""
        # Reveal all but one card for player 0
        self.game.revealed_cards[0] = set(range(5))
    
        # Draw and replace last card
        self.game.step(Action.DRAW_FROM_DECK)
        obs, reward, done, info = self.game.step(Action.REPLACE_5)
    
        self.assertTrue(self.game.final_round)
        self.assertEqual(self.game.last_player, 1)
        self.assertTrue(info.final_round)

    def test_game_end_conditions(self):
        """Test various game ending conditions."""
        # Test max turns limit
        self.game.config.max_turns = 5
    
        # Do 4 complete turns (each turn is a draw + replace)
        for _ in range(4):
            self.game.step(Action.DRAW_FROM_DECK)
            obs, reward, done, info = self.game.step(Action.REPLACE_0)
            self.assertFalse(done)
    
        # Start the 5th turn (should trigger max turns)
        obs, reward, done, info = self.game.step(Action.DRAW_FROM_DECK)
        self.assertTrue(done)
        self.assertTrue(info.max_turns_reached)
        self.assertEqual(info.game_end_reason, "Game ended after reaching maximum turns (5)")

    def test_reward_calculation(self):
        """Test reward calculation in various scenarios."""
        # Test terminal rewards
        self.game.game_over = True
        self.game.player_hands = [
            [Card.ACE] * 6,  # Score = 0 (all match)
            [Card.KING] * 6  # Score = 0 (all match)
        ]
    
        info = GameInfo()
        info.scores = [0, 0]  # Both players have perfect scores (tied for first)
        reward = self.game._calculate_terminal_reward(info)
        self.assertEqual(reward, 2.5)  # Reward for tie for first place
        
        # Test with specified scores where current player wins
        info.scores = [5, 10]  # Current player has better score
        reward = self.game._calculate_terminal_reward(info)
        self.assertGreater(reward, 5.0)  # Base win reward (clear winner) plus margin bonus
        
        # Test when current player loses
        info.scores = [10, 5]  # Current player has worse score
        reward = self.game._calculate_terminal_reward(info)
        self.assertLess(reward, 0)  # Negative reward for losing

    def test_reward_for_revealing_card(self):
        """Test that a reward is granted for revealing a card for the first time."""
        self.game.reset()
    
        # Draw a card 
        self.game.step(Action.DRAW_FROM_DECK)
        
        # Set up an unrevealed card position
        position = 1  # Use position 1 (not initially revealed)
        if position in self.game.revealed_cards[0]:
            self.game.revealed_cards[0].remove(position)
        
        # Save original card values for reference
        original_card = self.game.player_hands[0][position]
        drawn_card = self.game.drawn_card
        
        # Directly check the intermediate reward calculation difference with and without reveal bonus
        action = Action.REPLACE_0 + position
        
        # Calculate reward without revealing bonus
        info_without_reveal = GameInfo()
        info_without_reveal.was_previously_revealed = True
        reward_without_reveal = self.game._calculate_intermediate_reward(action, info_without_reveal)
        
        # Calculate reward with revealing bonus
        info_with_reveal = GameInfo()
        info_with_reveal.was_previously_revealed = False
        reward_with_reveal = self.game._calculate_intermediate_reward(action, info_with_reveal)
        
        # The difference should be exactly 0.2 (the revealing bonus)
        self.assertAlmostEqual(reward_with_reveal - reward_without_reveal, 0.2)

    def test_observation_space(self):
        """Test observation space structure and values."""
        obs = self.game._get_observation()
        cards_per_player = self.config.grid_rows * self.config.grid_cols
        expected_size = (2 * cards_per_player  # Player hands
                        + 2  # Discard and drawn card
                        + 2 * cards_per_player  # Revealed flags
                        + 3)  # Drawn from discard, final round flags, and turn progress
        
        self.assertEqual(len(obs), expected_size)
        
        # Test observation ranges for cards (player hands, discard, and drawn card)
        self.assertTrue(all(0 <= x <= len(Card) for x in obs[:2 * cards_per_player + 2]))
        
        # Test observation ranges for revealed flags and game state flags
        # First 2*cards_per_player + 2 elements are cards, rest are binary flags except the last one
        self.assertTrue(all(x in [0, 1] for x in obs[2 * cards_per_player + 2:-1]))
        
        # Last element is turn progress (between 0 and 1)
        self.assertTrue(0 <= obs[-1] <= 1)

    def test_unrevealed_cards_encoding(self):
        """Test that unrevealed cards are properly encoded with the unknown_card value."""
        # Create a game with known card values and revealed status
        game = GolfGame(self.config)
        cards_per_player = self.config.grid_rows * self.config.grid_cols
        unknown_card = len(Card)
        
        # Get the observation
        obs = game._get_observation()
        
        # Check that unrevealed cards have the unknown_card value
        for i in range(cards_per_player):
            if i not in game.revealed_cards[game.current_player]:
                self.assertEqual(obs[i], unknown_card, 
                                f"Unrevealed card at index {i} should have value {unknown_card}")
        
        # Check opponent's unrevealed cards
        opponent = (game.current_player + 1) % game.config.num_players
        for i in range(cards_per_player):
            if i not in game.revealed_cards[opponent]:
                self.assertEqual(obs[cards_per_player + i], unknown_card,
                                f"Opponent's unrevealed card at index {i} should have value {unknown_card}")
        
        # Check that revealed cards have their actual values
        for i in game.revealed_cards[game.current_player]:
            self.assertEqual(obs[i], game.player_hands[game.current_player][i],
                            f"Revealed card at index {i} should have its actual value")
            # Revealed flag indices should be offset by 2 (for discard and drawn card)
            self.assertEqual(obs[2 * cards_per_player + 2 + i], 1.0,
                            f"Revealed flag at index {i} should be 1.0")

    def test_render(self):
        """Test game state rendering."""
        rendered = self.game.render()
        self.assertIsInstance(rendered, str)
        
        # Check for key elements in rendered output
        self.assertIn("Player 0's turn", rendered)
        self.assertIn("Player 0's hand", rendered)
        self.assertIn("Player 1's hand", rendered)
        self.assertIn("Deck size:", rendered)
        self.assertIn("Cards revealed:", rendered)
        self.assertIn("Score:", rendered)

    def test_normalize_reward(self):
        """Test reward normalization in different scenarios."""
        # Test non-normalized rewards
        self.game.config.normalize_rewards = False
        self.assertEqual(self.game._normalize_reward(10), 10)
        self.assertEqual(self.game._normalize_reward(-10), -10)
        
        # Test normalized non-terminal rewards
        self.game.config.normalize_rewards = True
        self.assertEqual(self.game._normalize_reward(4), 0.5)  # Clipped to 0.5
        self.assertEqual(self.game._normalize_reward(-4), -0.5)  # Clipped to -0.5
        self.assertEqual(self.game._normalize_reward(2), 0.5)
        self.assertEqual(self.game._normalize_reward(-2), -0.5)
        
        # Test normalized terminal rewards
        self.assertEqual(self.game._normalize_reward(10, is_terminal=True), 1.0)  # 0.5 + 0.5
        self.assertEqual(self.game._normalize_reward(5, is_terminal=True), 1.0)  # 0.5 + 0.5
        self.assertEqual(self.game._normalize_reward(2, is_terminal=True), 0.7)  # 0.5 + 0.2
        self.assertEqual(self.game._normalize_reward(-5, is_terminal=True), -1.0)  # -0.5 - 0.5
        self.assertEqual(self.game._normalize_reward(-2, is_terminal=True), -0.7)  # -0.5 - 0.2

    def test_calculate_visible_score(self):
        """Test calculation of visible score (only revealed cards)."""
        # Set up a specific hand with some revealed cards
        self.game.player_hands[0] = [
            Card.ACE, Card.FIVE, Card.JACK,  # Top row
            Card.ACE, Card.SIX, Card.JACK    # Bottom row
        ]
        
        # No cards revealed
        self.game.revealed_cards[0] = set()
        self.assertEqual(self.game._calculate_visible_score(), 0)
        
        # Reveal one card
        self.game.revealed_cards[0] = {0}  # ACE in top row
        self.assertEqual(self.game._calculate_visible_score(), 1)
        
        # Reveal matching cards in a column
        self.game.revealed_cards[0] = {0, 3}  # ACE in top and bottom
        self.assertEqual(self.game._calculate_visible_score(), 0)  # Matching cards = 0
        
        # Reveal matching TWOs
        self.game.player_hands[0][0] = Card.TWO
        self.game.player_hands[0][3] = Card.TWO
        self.assertEqual(self.game._calculate_visible_score(), -4)  # Matching TWOs = -4
        
        # Reveal non-matching cards in a column
        self.game.player_hands[0] = [
            Card.KING, Card.FIVE, Card.JACK,  # Top row
            Card.ACE, Card.SIX, Card.JACK    # Bottom row
        ]
        self.game.revealed_cards[0] = {0, 3}  # KING in top, ACE in bottom
        self.assertEqual(self.game._calculate_visible_score(), 11)  # KING(10) + ACE(1) = 11
        
    def test_reset(self):
        """Test game reset functionality."""
        # Make some changes to the game state
        self.game.current_player = 1
        self.game.game_over = True
        self.game.final_round = True
        self.game.turn_count = 50
        self.game.drawn_card = Card.ACE
        
        # Reset the game
        obs = self.game.reset()
        
        # Verify game state is reset
        self.assertEqual(self.game.current_player, 0)
        self.assertFalse(self.game.game_over)
        self.assertFalse(self.game.final_round)
        self.assertEqual(self.game.turn_count, 0)
        self.assertIsNone(self.game.drawn_card)
        
        # Verify observation is returned
        self.assertIsInstance(obs, np.ndarray)
        
        # Verify deck is shuffled
        total_cards = len(Card) * self.config.copies_per_rank
        cards_dealt = self.config.num_players * self.config.grid_rows * self.config.grid_cols + 1
        self.assertEqual(len(self.game.deck), total_cards - cards_dealt)
        
    def test_discard_action(self):
        """Test discarding a drawn card."""
        # Draw a card first
        self.game.step(Action.DRAW_FROM_DECK)
        drawn_card = self.game.drawn_card
        
        # Discard the drawn card
        obs, reward, done, info = self.game.step(Action.DISCARD)
        
        # Verify card was discarded
        self.assertEqual(self.game.discard_pile[-1], drawn_card)
        self.assertIsNone(self.game.drawn_card)
        self.assertEqual(self.game.current_player, 1)  # Turn passed to next player
        
    def test_multiple_players(self):
        """Test game with more than 2 players."""
        # Create a new game with 3 players
        config = GameConfig(
            num_players=3,
            grid_rows=2,
            grid_cols=3,
            initial_revealed=2,
            max_turns=100
        )
        game = GolfGame(config)
        
        # Verify initialization
        self.assertEqual(len(game.player_hands), 3)
        self.assertEqual(len(game.revealed_cards), 3)
        
        # Test player rotation
        self.assertEqual(game.current_player, 0)
        game.step(Action.DRAW_FROM_DECK)
        game.step(Action.DISCARD)
        self.assertEqual(game.current_player, 1)
        game.step(Action.DRAW_FROM_DECK)
        game.step(Action.DISCARD)
        self.assertEqual(game.current_player, 2)
        game.step(Action.DRAW_FROM_DECK)
        game.step(Action.DISCARD)
        self.assertEqual(game.current_player, 0)  # Back to first player
    
    def test_flip_state_perspective(self):
        """Test that the state perspective can be flipped between players."""
        game = GolfGame()
        state = game._get_observation()
        
        # Manually create a state with known values
        cards_per_player = game.config.grid_rows * game.config.grid_cols
        test_state = np.zeros_like(state)
        
        # Set player 0's cards to values 0-5
        for i in range(cards_per_player):
            test_state[i] = i
            
        # Set player 1's cards to values 10-15
        for i in range(cards_per_player):
            test_state[cards_per_player + i] = 10 + i
            
        # Set revealed flags
        for i in range(3):  # First 3 cards of player 0 are revealed
            test_state[2*cards_per_player + 2 + i] = 1
            
        for i in range(2):  # First 2 cards of player 1 are revealed
            test_state[2*cards_per_player + 2 + cards_per_player + i] = 1
        
        # Flip the perspective
        flipped_state = game.flip_state_perspective(test_state)
        
        # Check that player hands are swapped
        for i in range(cards_per_player):
            self.assertEqual(flipped_state[i], 10 + i)  # Player 1's cards are now player 0's
            self.assertEqual(flipped_state[cards_per_player + i], i)  # Player 0's cards are now player 1's
            
        # Check that revealed flags are swapped
        for i in range(2):  # First 2 cards should be revealed (from player 1)
            self.assertEqual(flipped_state[2*cards_per_player + 2 + i], 1)
        
        for i in range(3):  # First 3 cards should be revealed (from player 0)
            self.assertEqual(flipped_state[2*cards_per_player + 2 + cards_per_player + i], 1)

if __name__ == '__main__':
    unittest.main() 