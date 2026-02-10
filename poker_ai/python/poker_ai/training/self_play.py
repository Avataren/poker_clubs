"""Batched self-play data generation."""

import numpy as np
import torch

from poker_ai.env.poker_env import BatchPokerEnv
from poker_ai.model.network import BestResponseNet, AverageStrategyNet
from poker_ai.model.state_encoder import (
    HOLE_CARDS_START, COMMUNITY_END,
    GAME_STATE_START, GAME_STATE_END,
    HAND_STRENGTH_START, HAND_STRENGTH_END,
)
from poker_ai.training.circular_buffer import CircularBuffer, Transition
from poker_ai.training.reservoir import ReservoirBuffer, SLTransition
from poker_ai.config.hyperparams import NFSPConfig


# Pre-compute slice indices for extract_static_features
_CARD_SLICE = slice(HOLE_CARDS_START, COMMUNITY_END)          # 364
_GAME_SLICE = slice(GAME_STATE_START, GAME_STATE_END)          # 25
_HAND_SLICE = slice(HAND_STRENGTH_START, HAND_STRENGTH_END)    # 52
_STATIC_DIM = (COMMUNITY_END - HOLE_CARDS_START) + (GAME_STATE_END - GAME_STATE_START) + (HAND_STRENGTH_END - HAND_STRENGTH_START)


def extract_static_features(obs: np.ndarray) -> np.ndarray:
    """Extract static features (441-dim) from full observation (569-dim)."""
    return np.concatenate([obs[_CARD_SLICE], obs[_GAME_SLICE], obs[_HAND_SLICE]])


def extract_static_features_batch(obs_batch: np.ndarray) -> np.ndarray:
    """Vectorized: extract static features from (n, 569) -> (n, 441)."""
    return np.concatenate([
        obs_batch[:, _CARD_SLICE],
        obs_batch[:, _GAME_SLICE],
        obs_batch[:, _HAND_SLICE],
    ], axis=1)


def pad_action_history(history: list[np.ndarray], max_len: int = 200) -> tuple[np.ndarray, int]:
    """Pad action history to fixed length. Returns (padded[max_len, 7], length)."""
    length = min(len(history), max_len)
    padded = np.zeros((max_len, 7), dtype=np.float32)
    for i in range(length):
        padded[i] = history[i]
    return padded, length


def make_action_record(player: int, action: int, num_players: int) -> np.ndarray:
    """Create a 7-dim action record."""
    rec = np.zeros(7, dtype=np.float32)
    rec[0] = player / max(1, num_players - 1)
    if action == 0:
        rec[1] = 1.0
    elif action == 1:
        rec[2] = 1.0
    elif action <= 3:
        rec[3] = 1.0
    elif action <= 5:
        rec[4] = 1.0
    else:
        rec[5] = 1.0
    rec[6] = min(action / 7.0, 1.0)
    return rec


class SelfPlayWorker:
    """Runs batched self-play episodes and populates replay buffers.

    Optimizations:
    - GPU inference batched: one forward pass per network per step
    - Env stepping batched: single Rust FFI call for all active envs
    """

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

        self.use_as = np.zeros((config.num_envs, config.num_players), dtype=bool)
        self.prev_obs = np.zeros((config.num_envs, 569), dtype=np.float32)
        self.prev_mask = np.zeros((config.num_envs, 8), dtype=bool)
        self.prev_player = np.zeros(config.num_envs, dtype=np.intp)
        self.action_histories: list[list[list[np.ndarray]]] = [
            [[] for _ in range(config.num_players)] for _ in range(config.num_envs)
        ]

    def reset_episode_strategies(self):
        self.use_as = np.random.random((self.config.num_envs, self.config.num_players)) < self.config.eta

    def _prepare_batch(self, env_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batched tensors for a list of env indices."""
        n = len(env_indices)
        # Extract static features from stored observations
        obs_batch = extract_static_features_batch(self.prev_obs[env_indices])

        ah_array = np.zeros((n, 200, 7), dtype=np.float32)
        ah_lens = np.zeros(n, dtype=np.int64)
        mask_batch = self.prev_mask[env_indices]

        for i, idx in enumerate(env_indices):
            player = self.prev_player[idx]
            history = self.action_histories[idx][player]
            length = min(len(history), 200)
            ah_lens[i] = length
            for j in range(length):
                ah_array[i, j] = history[j]

        obs_t = torch.from_numpy(obs_batch).to(self.device)
        ah_t = torch.from_numpy(ah_array).to(self.device)
        ah_len_t = torch.from_numpy(ah_lens).to(self.device)
        mask_t = torch.from_numpy(mask_batch.copy()).to(self.device)

        return obs_t, ah_t, ah_len_t, mask_t

    def run_episodes(self, epsilon: float) -> int:
        """Run one batch of episodes to completion. Returns number of steps."""
        self.reset_episode_strategies()
        results = self.env.reset_all()
        total_steps = 0

        env_done = np.zeros(self.config.num_envs, dtype=bool)
        for i in range(self.config.num_envs):
            self.action_histories[i] = [[] for _ in range(self.config.num_players)]
            player, obs, mask = results[i]
            self.prev_obs[i] = obs
            self.prev_mask[i] = mask
            self.prev_player[i] = player

        num_players = self.config.num_players

        while not env_done.all():
            # Partition active envs into AS vs BR groups
            active_mask = ~env_done
            active_indices = np.where(active_mask)[0]

            # Classify by strategy
            active_players = self.prev_player[active_indices]
            is_as = self.use_as[active_indices, active_players]
            as_envs = active_indices[is_as].tolist()
            br_envs = active_indices[~is_as].tolist()

            # Batched GPU inference
            actions_map = {}
            with torch.no_grad():
                if as_envs:
                    obs_b, ah_b, ah_len_b, mask_b = self._prepare_batch(as_envs)
                    batch_acts = self.as_net.select_action(obs_b, ah_b, ah_len_b, mask_b).cpu().numpy()
                    for i, idx in enumerate(as_envs):
                        actions_map[idx] = int(batch_acts[i])

                if br_envs:
                    obs_b, ah_b, ah_len_b, mask_b = self._prepare_batch(br_envs)
                    batch_acts = self.br_net.select_action(obs_b, ah_b, ah_len_b, mask_b, epsilon).cpu().numpy()
                    for i, idx in enumerate(br_envs):
                        actions_map[idx] = int(batch_acts[i])

            # Build action pairs for batch step
            action_pairs = [(int(idx), actions_map[idx]) for idx in active_indices]

            # Snapshot pre-step state for transitions
            pre_obs = self.prev_obs[active_indices].copy()
            pre_mask = self.prev_mask[active_indices].copy()
            pre_players = self.prev_player[active_indices].copy()

            # Batch step all active envs in one Rust call
            next_players, next_obs_batch, next_masks_batch, rewards_batch, dones = (
                self.env.step_batch(action_pairs)
            )

            total_steps += len(active_indices)

            # Process results and store transitions
            done_indices = []
            for i, idx in enumerate(active_indices):
                idx = int(idx)
                player = int(pre_players[i])
                action = actions_map[idx]
                obs = pre_obs[i]
                mask = pre_mask[i]
                static_obs = extract_static_features(obs)
                history = self.action_histories[idx][player]
                ah_padded, ah_len = pad_action_history(history)

                done = dones[i]
                next_obs = next_obs_batch[i]
                next_mask = next_masks_batch[i]
                next_player = next_players[i]
                rewards = rewards_batch[i]

                next_static = extract_static_features(next_obs)
                next_history = self.action_histories[idx][next_player] if not done else []
                next_ah_padded, next_ah_len = pad_action_history(next_history)

                if not self.use_as[idx, player]:
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

                if self.use_as[idx, player]:
                    self.as_buffer.push(SLTransition(
                        obs=static_obs,
                        action_history=ah_padded,
                        history_length=ah_len,
                        action=action,
                        legal_mask=mask,
                    ))

                action_record = make_action_record(player, action, num_players)
                for p in range(num_players):
                    self.action_histories[idx][p].append(action_record.copy())

                if done:
                    env_done[idx] = True
                    done_indices.append(idx)
                else:
                    self.prev_obs[idx] = next_obs
                    self.prev_mask[idx] = next_mask
                    self.prev_player[idx] = next_player

            # Batch-reset all done envs
            if done_indices:
                reset_players, reset_obs, reset_masks = self.env.reset_batch(done_indices)
                for i, idx in enumerate(done_indices):
                    self.action_histories[idx] = [[] for _ in range(num_players)]
                    self.prev_obs[idx] = reset_obs[i]
                    self.prev_mask[idx] = reset_masks[i]
                    self.prev_player[idx] = reset_players[i]

        return total_steps
