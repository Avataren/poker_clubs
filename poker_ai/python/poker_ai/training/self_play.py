"""Batched self-play data generation."""

import numpy as np
import torch
from typing import Optional

from poker_ai.env.poker_env import BatchPokerEnv
from poker_ai.model.network import BestResponseNet, AverageStrategyNet
from poker_ai.model.state_encoder import (
    HOLE_CARDS_START, HOLE_CARDS_END,
    COMMUNITY_START, COMMUNITY_END,
    GAME_STATE_START, GAME_STATE_END,
    HAND_STRENGTH_START, HAND_STRENGTH_END,
    LSTM_PLACEHOLDER_START, LSTM_PLACEHOLDER_END,
)
from poker_ai.training.circular_buffer import CircularBuffer, Transition
from poker_ai.training.reservoir import ReservoirBuffer, SLTransition
from poker_ai.config.hyperparams import NFSPConfig


def extract_static_features(obs: np.ndarray) -> np.ndarray:
    """Extract static features (441-dim) from full observation (569-dim).

    Strips out the LSTM placeholder which is handled separately.
    """
    cards = obs[HOLE_CARDS_START:COMMUNITY_END]       # 364
    game_state = obs[GAME_STATE_START:GAME_STATE_END]  # 25
    hand_str = obs[HAND_STRENGTH_START:HAND_STRENGTH_END]  # 52
    return np.concatenate([cards, game_state, hand_str])


def pad_action_history(history: list[np.ndarray], max_len: int = 200) -> tuple[np.ndarray, int]:
    """Pad action history to fixed length.

    Returns:
        (padded_array of shape (max_len, 7), actual_length)
    """
    length = min(len(history), max_len)
    padded = np.zeros((max_len, 7), dtype=np.float32)
    for i in range(length):
        padded[i] = history[i]
    return padded, length


class SelfPlayWorker:
    """Runs batched self-play episodes and populates replay buffers."""

    def __init__(
        self,
        config: NFSPConfig,
        br_net: BestResponseNet,
        as_net: AverageStrategyNet,
        br_buffer: CircularBuffer,
        as_buffer: ReservoirBuffer,
        device: torch.device,
    ):
        self.config = config
        self.br_net = br_net
        self.as_net = as_net
        self.br_buffer = br_buffer
        self.as_buffer = as_buffer
        self.device = device
        self.env = BatchPokerEnv(
            num_envs=config.num_envs,
            num_players=config.num_players,
            starting_stack=config.starting_stack,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
        )

        # Track which strategy each player uses per episode
        # True = use AS (average strategy), False = use BR (best response)
        self.use_as = np.zeros((config.num_envs, config.num_players), dtype=bool)

        # Per-env state tracking
        self.prev_obs = [None] * config.num_envs
        self.prev_mask = [None] * config.num_envs
        self.prev_action = [None] * config.num_envs
        self.prev_player = [None] * config.num_envs
        self.action_histories: list[list[list[np.ndarray]]] = [
            [[] for _ in range(config.num_players)] for _ in range(config.num_envs)
        ]

    def reset_episode_strategies(self):
        """Randomly assign AS vs BR strategy for each player in each env."""
        self.use_as = np.random.random((self.config.num_envs, self.config.num_players)) < self.config.eta

    def run_episodes(self, epsilon: float) -> int:
        """Run one batch of episodes to completion.

        Returns number of steps taken.
        """
        self.reset_episode_strategies()
        results = self.env.reset_all()
        total_steps = 0

        # Initialize per-env tracking
        env_done = [False] * self.config.num_envs
        for i in range(self.config.num_envs):
            self.action_histories[i] = [[] for _ in range(self.config.num_players)]
            player, obs, mask = results[i]
            self.prev_obs[i] = obs
            self.prev_mask[i] = mask
            self.prev_player[i] = player
            self.prev_action[i] = None

        while not all(env_done):
            for env_idx in range(self.config.num_envs):
                if env_done[env_idx]:
                    continue

                player = self.prev_player[env_idx]
                obs = self.prev_obs[env_idx]
                static_obs = extract_static_features(obs)

                # Get action history for current player
                history = self.action_histories[env_idx][player]
                ah_padded, ah_len = pad_action_history(history)

                # Convert to tensors
                obs_t = torch.tensor(static_obs, device=self.device).unsqueeze(0)
                ah_t = torch.tensor(ah_padded, device=self.device).unsqueeze(0)
                ah_len_t = torch.tensor([ah_len], device=self.device)

                # Get legal mask from tracked state
                mask = self.prev_mask[env_idx]
                mask_t = torch.tensor(mask, device=self.device).unsqueeze(0)

                # Select action based on strategy assignment
                with torch.no_grad():
                    if self.use_as[env_idx, player]:
                        action = self.as_net.select_action(obs_t, ah_t, ah_len_t, mask_t).item()
                    else:
                        action = self.br_net.select_action(
                            obs_t, ah_t, ah_len_t, mask_t, epsilon
                        ).item()

                # Step environment
                next_player, next_obs, next_mask, rewards, done = self.env.step(env_idx, action)
                total_steps += 1

                # Store transitions
                next_static = extract_static_features(next_obs)
                next_history = self.action_histories[env_idx][next_player] if not done else []
                next_ah_padded, next_ah_len = pad_action_history(next_history)

                # BR buffer: store (s, a, r, s') for current player
                if not self.use_as[env_idx, player]:
                    self.br_buffer.push(Transition(
                        obs=static_obs,
                        action_history=ah_padded,
                        history_length=ah_len,
                        action=action,
                        reward=rewards[player],
                        next_obs=next_static,
                        next_action_history=next_ah_padded,
                        next_history_length=next_ah_len,
                        next_legal_mask=next_mask,
                        done=done,
                        legal_mask=mask,
                    ))

                # AS buffer: store (s, a) for average strategy players
                if self.use_as[env_idx, player]:
                    self.as_buffer.push(SLTransition(
                        obs=static_obs,
                        action_history=ah_padded,
                        history_length=ah_len,
                        action=action,
                        legal_mask=mask,
                    ))

                # Update action history for all players in this env
                # (simplified: just track action taken by current player)
                action_record = np.zeros(7, dtype=np.float32)
                action_record[0] = player / max(1, self.config.num_players - 1)
                if action == 0:
                    action_record[1] = 1.0  # fold
                elif action == 1:
                    action_record[2] = 1.0  # check/call
                elif action <= 3:
                    action_record[3] = 1.0  # small raise
                elif action <= 5:
                    action_record[4] = 1.0  # medium raise
                else:
                    action_record[5] = 1.0  # large raise / all-in
                # bet ratio approximation
                action_record[6] = min(action / 7.0, 1.0)

                for p in range(self.config.num_players):
                    self.action_histories[env_idx][p].append(action_record.copy())

                if done:
                    env_done[env_idx] = True
                    # Prepare env for next call (reset state)
                    player, obs, mask = self.env.reset_env(env_idx)
                    self.action_histories[env_idx] = [[] for _ in range(self.config.num_players)]
                    self.prev_obs[env_idx] = obs
                    self.prev_mask[env_idx] = mask
                    self.prev_player[env_idx] = player
                    self.prev_action[env_idx] = None
                else:
                    self.prev_obs[env_idx] = next_obs
                    self.prev_mask[env_idx] = next_mask
                    self.prev_player[env_idx] = next_player
                    self.prev_action[env_idx] = action

        return total_steps
