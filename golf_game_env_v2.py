import numpy as np
import random
from typing import Optional, Dict, Any, Tuple
import gymnasium
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers


def env(num_players=2, render_mode=None):
    """
    Public API for creating a Six Card Golf environment with two-step turns:
      1) Draw (stock or discard)
      2) Decide (swap or discard).
    """
    raw = CardGolfEnv(num_players=num_players, render_mode=render_mode)
    if render_mode == "human":
        raw = wrappers.CaptureStdoutWrapper(raw)
    raw = wrappers.TerminateIllegalWrapper(raw, illegal_reward=-100)
    raw = wrappers.AssertOutOfBoundsWrapper(raw)
    raw = wrappers.OrderEnforcingWrapper(raw)
    return raw


class CardGolfEnv(AECEnv):
    """
    - Each turn has 2 actions:
        Phase 0: draw from stock (action=0) or discard (action=1)
        Phase 1: either place card in slot 0..5 (actions=2..7) or discard it (action=8)
    - The drawn card is visible to the acting player in the observation after the first action.
    - Round ends as soon as any player has all 6 face-up cards.
    - Final reward = -score, where score uses standard 6-card golf scoring.
    """

    metadata = {"render_modes": ["human"], "name": "six_card_golf_v0", "is_parallelizable": True}

    def __init__(self, num_players=2, render_mode: Optional[str] = None):
        super().__init__()
        self.num_players = num_players
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = []

        # 9 discrete actions for the two-step turn.
        #  Phase 0 => {0=drawStock, 1=drawDiscard}
        #  Phase 1 => {2..7=swapSlot0..5, 8=discard}
        self._action_spaces = {
            agent: spaces.Discrete(9) for agent in self.possible_agents
        }

        # Observation shape:
        #   - 6 ranks for self cards (face-down = -1)
        #   - (num_players-1)*6 ranks for opponents' face-up (face-down = -1)
        #   - discard top rank (or -1 if empty)
        #   - stock size
        #   - drawn card rank (or -1 if none drawn)
        # total = 6 + 6*(num_players-1) + 3
        self.obs_size = 6 + 6*(num_players-1) + 3
        obs_box = spaces.Box(low=-1, high=52, shape=(self.obs_size,), dtype=np.int32)
        mask_box = spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8)

        self._observation_spaces = {
            agent: spaces.Dict({
                "observation": obs_box, 
                "action_mask": mask_box
            }) for agent in self.possible_agents
        }
        
        # Initialize attributes required for PettingZoo compatibility
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminal_rewards = {}

    # Methods to get observation and action spaces for compatibility with RLlib and PettingZoo
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state.
        
        Args:
            seed: The random seed to use
            options: Additional options for reset
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agents = self.possible_agents[:]
        self.has_terminated = False

        # For tracking previously revealed cards (used in reward calculation)
        self.prev_face_up = {agent: set() for agent in self.agents}
        
        # Deck of 52 => ranks 0..12 each repeated 4 times
        deck = [r for r in range(13) for _ in range(4)]
        random.shuffle(deck)

        # Deal each player 6 cards, flip 2 random face-up
        self.player_cards = {}
        for agent in self.agents:
            hand = [{"rank": deck.pop(), "face_up": False} for _ in range(6)]
            up_idxs = random.sample(range(6), 2)
            for idx in up_idxs:
                hand[idx]["face_up"] = True
                self.prev_face_up[agent].add(idx)
            self.player_cards[agent] = hand

        # Discard top card
        self.discard_pile = [deck.pop()]
        self.stock_pile = deck

        # Two-step turn logic:
        #  self.turn_phase in {0=need to draw, 1=decide swap/discard}
        #  self.tmp_drawn_card is None or the rank we just drew
        self.turn_phase = 0
        self.tmp_drawn_card = None

        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()

        # Initialize rewards and state dictionaries
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminal_rewards = {}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"was_previously_revealed": False} for agent in self.agents}
        
        # Return first observations and empty info dict
        return self.observe(self.agent_selection), {}

    def step(self, action):
        # Reset immediate rewards for all agents at the beginning of each step
        self._clear_rewards()
        self._cumulative_rewards[self.agent_selection] = 0

        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        current_agent = self.agent_selection

        if self.terminations[current_agent]:
            # Handle terminated agents
            if current_agent in self.terminal_rewards:
                self.rewards[current_agent] = self.terminal_rewards[current_agent]
                del self.terminal_rewards[current_agent]
            else:
                self.rewards[current_agent] = 0.0
            self._accumulate_rewards()
            self.agent_selection = self.agent_selector.next()
            return

        # Validate action
        mask = self._make_action_mask(current_agent)
        if action < 0 or action >= 9 or mask[action] == 0:
            self.terminations[current_agent] = True
            self.has_terminated = True
            return

        immediate_reward = 0.0
        if self.turn_phase == 0:
            # Draw phase
            if action == 0:
                card = self._draw_from_stock()
                self.tmp_drawn_card = card if card is not None else None
            else:
                self.tmp_drawn_card = self.discard_pile.pop() if self.discard_pile else None
            immediate_reward = self._calculate_draw_reward(current_agent, action)
            self.rewards[current_agent] = immediate_reward
            self.turn_phase = 1
        else:
            # Decide phase
            self.infos[current_agent]["was_previously_revealed"] = False
            if action in range(2, 8):
                slot = action - 2
                self.infos[current_agent]["was_previously_revealed"] = slot in self.prev_face_up[current_agent]
                immediate_reward = self._calculate_swap_reward(current_agent, slot, self.tmp_drawn_card)
                self._swap_card(current_agent, slot, self.tmp_drawn_card)
                self.prev_face_up[current_agent].add(slot)
            else:
                immediate_reward = self._calculate_discard_reward(current_agent) if self.tmp_drawn_card else 0.0
                if self.tmp_drawn_card is not None:
                    self.discard_pile.append(self.tmp_drawn_card)

            if not all(c["face_up"] for c in self.player_cards[current_agent]):
                visible_score = self._calculate_visible_score(current_agent)
                immediate_reward -= 0.005 * visible_score
            self.rewards[current_agent] = immediate_reward

            if all(c["face_up"] for c in self.player_cards[current_agent]):
                self._end_round(current_agent)
            else:
                # After decide phase, reset phase and move to next agent if not terminated
                self.turn_phase = 0
                self.tmp_drawn_card = None
                self.agent_selection = self.agent_selector.next()

        # Accumulate rewards once, after all updates
        self._accumulate_rewards()

    def _end_round(self, ending_agent):
        """End the current round and set terminal rewards."""
        self.has_terminated = True
        scores = {ag: self._score_hand(ag) for ag in self.agents}
        for ag in self.agents:
            terminal_reward = self._calculate_terminal_reward(ag, scores[ag])
            self.rewards[ag] = terminal_reward
            self.terminations[ag] = True
            self.truncations[ag] = False

    def observe(self, agent):
        """
        Return agent observation according to the format expected by RLlib wrappers.
        Also checks if the agent is valid.
        """
        if agent not in self.agents:
            return None
        
        obs = self._make_obs(agent)
        mask = self._make_action_mask(agent)
        return {"observation": obs, "action_mask": mask}

    def render(self):
        if self.render_mode != "human":
            return
        print("\n--- Six Card Golf State ---")
        print(f"Current agent: {self.agent_selection}; Phase: {self.turn_phase}")
        for ag in self.possible_agents:
            cards_str = []
            for c in self.player_cards[ag]:
                if c["face_up"]:
                    cards_str.append(self._rank_to_str(c["rank"]))
                else:
                    cards_str.append("XX")
            print(f"{ag}: {cards_str}")
        disc_top = "Empty" if not self.discard_pile else self._rank_to_str(self.discard_pile[-1])
        print(f"Discard top: {disc_top}, Stock: {len(self.stock_pile)}")
        if self.tmp_drawn_card is not None:
            print(f"Drawn card (for {self.agent_selection}): {self._rank_to_str(self.tmp_drawn_card)}")
        else:
            print(f"No drawn card currently. Phase={self.turn_phase}")

    def close(self):
        pass

    def _score_hand(self, agent):
        """
        Score with 3 columns: (0,3), (1,4), (2,5).
        If ranks match => 0 points for that column, else sum card values.
        """
        cards = self.player_cards[agent]
        pairs = [(0, 3), (1, 4), (2, 5)]
        total = 0
        for (i, j) in pairs:
            r1 = cards[i]["rank"]
            r2 = cards[j]["rank"]
            if r1 == r2:
                continue
            total += self._card_value(r1) + self._card_value(r2)
        return total

    def _swap_card(self, agent, slot, new_rank):
        old_card = self.player_cards[agent][slot]
        self.discard_pile.append(old_card["rank"])
        self.player_cards[agent][slot] = {"rank": new_rank, "face_up": True}

    def _draw_from_stock(self):
        if len(self.stock_pile) == 0:
            return None
        return self.stock_pile.pop()

    def _make_obs(self, agent):
        # [my6, other6*(N-1), discardTop, stockSize, drawnCard]
        obs = []
        # my cards
        for c in self.player_cards[agent]:
            obs.append(c["rank"] if c["face_up"] else -1)
        # opponents
        for other in self.agents:
            if other == agent:
                continue
            for c in self.player_cards[other]:
                obs.append(c["rank"] if c["face_up"] else -1)
        # discard top
        disc_top = self.discard_pile[-1] if len(self.discard_pile) > 0 else -1
        obs.append(disc_top)
        # stock size
        obs.append(len(self.stock_pile))
        # drawn card rank or -1
        obs.append(self.tmp_drawn_card if self.tmp_drawn_card is not None else -1)
        return np.array(obs, dtype=np.int32)

    def _make_action_mask(self, agent):
        # 9-dim => depends on self.turn_phase
        # Phase 0 => only actions {0,1} valid, plus checks if stock/discard empty
        # Phase 1 => only actions {2..8} valid
        mask = np.zeros(9, dtype=np.int8)
        
        # Return empty mask if it's not this agent's turn
        if agent != self.agent_selection:
            return mask
            
        if self.turn_phase == 0:
            # we can do draw from stock (0) if stock not empty
            if len(self.stock_pile) > 0:
                mask[0] = 1
            # we can do draw from discard (1) if discard not empty
            if len(self.discard_pile) > 0:
                mask[1] = 1
        else:
            # turn_phase == 1 => 2..8 are valid
            # but we need a drawn card to swap or discard
            if self.tmp_drawn_card is not None:
                # can swap with any of 6 slots
                mask[2:8] = 1
                # can also discard
                mask[8] = 1
            else:
                # if no card drawn => can't do anything
                pass
        return mask

    @staticmethod
    def _rank_to_str(r):
        mapping = {
            0: "A", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7",
            7: "8", 8: "9", 9: "10", 10: "J", 11: "Q", 12: "K"
        }
        return mapping[r] if (r >= 0 and r <= 12) else "??"

    @staticmethod
    def _card_value(rank):
        if rank == 0:  # Ace
            return 1
        elif rank == 1:  # 2
            return -2
        elif 2 <= rank <= 8:
            return rank + 1  # 3->4,4->5,...,8->9
        else:
            return 10  # 9->10,10->J->10,11->Q->10,12->K->10
    
    def _calculate_visible_score(self, agent):
        """Calculate score only considering revealed cards."""
        score = 0
        cards = self.player_cards[agent]
        pairs = [(0, 3), (1, 4), (2, 5)]
        
        for (i, j) in pairs:
            i_revealed = cards[i]["face_up"]
            j_revealed = cards[j]["face_up"]
            
            if i_revealed and j_revealed:
                if cards[i]["rank"] == cards[j]["rank"]:
                    # Matched pair
                    continue
                score += self._card_value(cards[i]["rank"]) + self._card_value(cards[j]["rank"])
            elif i_revealed:
                score += self._card_value(cards[i]["rank"])
            elif j_revealed:
                score += self._card_value(cards[j]["rank"])
        
        return score
    
    def _calculate_terminal_reward(self, agent, score):
        """Calculate terminal reward based on final scores."""
        # Get all player scores
        scores = {ag: self._score_hand(ag) for ag in self.agents}
        min_score = min(scores.values())
        current_player_score = scores[agent]
        
        # Count how many players have the minimum score (tied for first)
        num_tied_for_first = sum(1 for s in scores.values() if s == min_score)
        
        if current_player_score == min_score:
            # If there's a tie for first place
            if num_tied_for_first > 1:
                # Reduced reward for ties - still positive but less than a clear win
                avg_score = sum(scores.values()) / len(scores)
                margin = avg_score - current_player_score
                return 2.5 + 0.3 * margin  # Half the normal winning reward
            else:
                # Clear winner gets full reward
                avg_score = sum(scores.values()) / len(scores)
                margin = avg_score - current_player_score
                return 5.0 + 0.3 * margin
        else:
            # Losing player's reward is based on how far they are from the minimum score
            return -0.3 * (current_player_score - min_score)
    
    def _calculate_draw_reward(self, agent, action):
        """Calculate reward for drawing a card."""
        reward = 0.0
        
        # Small reward for drawing from discard if it could lead to a match
        if action == 1 and len(self.discard_pile) > 0:
            drawn_rank = self.tmp_drawn_card
            for col in range(3):  # 3 columns
                i, j = col, col + 3  # Top and bottom positions in column
                if self.player_cards[agent][i]["face_up"] and self.player_cards[agent][i]["rank"] == drawn_rank:
                    reward += 0.3
                if self.player_cards[agent][j]["face_up"] and self.player_cards[agent][j]["rank"] == drawn_rank:
                    reward += 0.3
                    
        return reward
    
    def _calculate_swap_reward(self, agent, slot, new_rank):
        """Calculate reward for swapping a card."""
        reward = 0.0
        
        # Value of the old card
        old_card = self.player_cards[agent][slot]
        old_value = self._card_value(old_card["rank"])
        new_value = self._card_value(new_rank)
        
        # Reward for value improvement
        value_improvement = old_value - new_value
        if value_improvement > 0:
            reward += value_improvement
        
        # Check if this creates a match
        col = slot % 3
        other_slot = (slot + 3) % 6 if slot < 3 else slot - 3
        other_card = self.player_cards[agent][other_slot]
        
        if other_card["face_up"] and other_card["rank"] == new_rank:
            # Bonus for creating a match
            reward += 2.0
        
        # Reward for revealing a new card
        if not old_card["face_up"]:
            reward += 0.2
            
        return reward
    
    def _calculate_discard_reward(self, agent):
        """Calculate reward for discarding a card."""
        reward = 0.0
        
        # Penalty for discarding potential matches
        discarded_rank = self.tmp_drawn_card
        for col in range(3):
            top_slot = col
            bottom_slot = col + 3
            top_card = self.player_cards[agent][top_slot]
            bottom_card = self.player_cards[agent][bottom_slot]
            
            if top_card["face_up"] and top_card["rank"] == discarded_rank:
                reward -= 1.5
            if bottom_card["face_up"] and bottom_card["rank"] == discarded_rank:
                reward -= 1.5
                
        return reward
