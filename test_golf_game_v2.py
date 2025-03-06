import unittest
import numpy as np
from golf_game_v2 import GolfGame, GameConfig, Action, Card

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
        self.assertIn('error', info)

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
        self.assertIn('final_round', info)
        self.assertEqual(info['trigger_player'], 0)

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
        self.assertTrue(info['max_turns_reached'])

        # Test final round ending
        self.game.reset()
        self.game.final_round = True
        self.game.last_player = 1
        self.game.current_player = 1
        
        # Make the final move
        self.game.step(Action.DRAW_FROM_DECK)
        obs, reward, done, info = self.game.step(Action.REPLACE_0)
        
        self.assertTrue(done)
        self.assertTrue(self.game.game_over)
        self.assertIn('scores', info)

    def test_reward_calculation(self):
        """Test reward calculation in various scenarios."""
        # Test terminal rewards
        self.game.game_over = True
        self.game.player_hands = [
            [Card.ACE] * 6,  # Score = 0 (all match)
            [Card.KING] * 6  # Score = 0 (all match)
        ]
        
        reward = self.game._calculate_terminal_reward({})
        self.assertEqual(reward, 5.0)  # Base win reward (tied for lowest score)

        # Test intermediate rewards
        self.game.reset()
        self.game.step(Action.DRAW_FROM_DECK)
        self.game.drawn_card = Card.ACE  # Low value card
        
        # Replace a KING (high value) with an ACE (low value)
        self.game.player_hands[0][0] = Card.KING
        self.game.revealed_cards[0].add(0)  # Reveal the card
        obs, reward, done, info = self.game.step(Action.REPLACE_0)
        
        # Should get positive reward for:
        # 1. Value improvement (KING=10 to ACE=1)
        # 2. Potential match creation
        self.assertGreater(reward, 0)

    def test_observation_space(self):
        """Test observation space structure and values."""
        obs = self.game._get_observation()
        cards_per_player = self.config.grid_rows * self.config.grid_cols
        expected_size = (2 * cards_per_player  # Player hands
                        + 2  # Discard and drawn card
                        + 2 * cards_per_player  # Revealed flags
                        + 2)  # Drawn from discard and final round flags
        
        self.assertEqual(len(obs), expected_size)
        
        # Test observation ranges
        self.assertTrue(all(0 <= x <= len(Card) for x in obs[:2 * cards_per_player + 2]))
        self.assertTrue(all(x in [0, 1] for x in obs[2 * cards_per_player + 2:]))

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

if __name__ == '__main__':
    unittest.main() 