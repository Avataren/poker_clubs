"""Batched self-play data generation with continuous episodes."""

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


# Pre-compute slice indices
_CARD_SLICE = slice(HOLE_CARDS_START, COMMUNITY_END)          # 364
_GAME_SLICE = slice(GAME_STATE_START, GAME_STATE_END)          # 25
_HAND_SLICE = slice(HAND_STRENGTH_START, HAND_STRENGTH_END)    # 52
# Column indices for vectorized extraction
_STATIC_COLS = np.concatenate([
    np.arange(HOLE_CARDS_START, COMMUNITY_END),
    np.arange(GAME_STATE_START, GAME_STATE_END),
    np.arange(HAND_STRENGTH_START, HAND_STRENGTH_END),
])


def extract_static_features_batch(obs_batch: np.ndarray) -> np.ndarray:
    """Vectorized: extract static features from (n, 569) -> (n, 441)."""
    return obs_batch[:, _STATIC_COLS]


def pad_action_history(history: list[np.ndarray], max_len: int = 30) -> tuple[np.ndarray, int]:
    """Pad action history to fixed length. Returns (padded[max_len, 7], length)."""
    length = min(len(history), max_len)
    padded = np.zeros((max_len, 7), dtype=np.float32)
    if length > 0:
        padded[:length] = history[-length:]  # take most recent entries
    return padded, length


# Pre-computed action record templates
_ACTION_TEMPLATES = np.zeros((8, 7), dtype=np.float32)
_ACTION_TEMPLATES[0, 1] = 1.0  # fold
_ACTION_TEMPLATES[1, 2] = 1.0  # check/call
_ACTION_TEMPLATES[2, 3] = 1.0  # small raise
_ACTION_TEMPLATES[3, 3] = 1.0  # small raise
_ACTION_TEMPLATES[4, 4] = 1.0  # medium raise
_ACTION_TEMPLATES[5, 4] = 1.0  # medium raise
_ACTION_TEMPLATES[6, 5] = 1.0  # large raise
_ACTION_TEMPLATES[7, 5] = 1.0  # all-in
for i in range(8):
    _ACTION_TEMPLATES[i, 6] = min(i / 7.0, 1.0)


def make_action_record(player: int, action: int, num_players: int) -> np.ndarray:
    """Create a 7-dim action record."""
    rec = _ACTION_TEMPLATES[action].copy()
    rec[0] = player / max(1, num_players - 1)
    return rec


class SelfPlayWorker:
    """Continuous batched self-play: all envs always active, instant resets."""

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

        self.num_players = config.num_players
        self.num_envs = config.num_envs
        self.max_hist = config.max_history_len

        # Per-env state (always active)
        self.use_as = np.zeros((config.num_envs, config.num_players), dtype=bool)
        self.prev_obs = np.zeros((config.num_envs, 569), dtype=np.float32)
        self.prev_mask = np.zeros((config.num_envs, 8), dtype=bool)
        self.prev_player = np.zeros(config.num_envs, dtype=np.intp)
        # Action histories stored as fixed-size ring buffers per env per player
        self.ah_arrays = np.zeros(
            (config.num_envs, config.num_players, config.max_history_len, 7),
            dtype=np.float32,
        )
        self.ah_lens = np.zeros(
            (config.num_envs, config.num_players), dtype=np.int64
        )
        self._initialized = False

    def _get_history(self, env: int, player: int) -> tuple[np.ndarray, int]:
        """Get (padded_history, length) for an env/player."""
        return self.ah_arrays[env, player], int(self.ah_lens[env, player])

    def _append_actions_all(self, action_records: np.ndarray):
        """Append action records for all envs. action_records: (n, 7)."""
        max_hist = self.max_hist
        num_players = self.num_players
        for i in range(len(action_records)):
            rec = action_records[i]
            for p in range(num_players):
                length = int(self.ah_lens[i, p])
                if length < max_hist:
                    self.ah_arrays[i, p, length] = rec
                    self.ah_lens[i, p] = length + 1
                else:
                    self.ah_arrays[i, p, :-1] = self.ah_arrays[i, p, 1:]
                    self.ah_arrays[i, p, -1] = rec

    def _reset_history(self, env: int):
        """Reset action histories for an env."""
        self.ah_arrays[env] = 0.0
        self.ah_lens[env] = 0

    def _init_envs(self):
        results = self.env.reset_all()
        self.use_as[:] = np.random.random((self.num_envs, self.num_players)) < self.config.eta
        self.ah_arrays[:] = 0.0
        self.ah_lens[:] = 0
        for i in range(self.num_envs):
            player, obs, mask = results[i]
            self.prev_obs[i] = obs
            self.prev_mask[i] = mask
            self.prev_player[i] = player
        self._initialized = True

    def run_episodes(self, epsilon: float) -> int:
        """Run continuous self-play until at least num_envs episodes complete."""
        if not self._initialized:
            self._init_envs()

        n = self.num_envs
        num_players = self.num_players
        max_hist = self.max_hist
        total_steps = 0
        completed_episodes = 0

        while completed_episodes < n:
            # Prepare full batch tensors
            players = self.prev_player
            static_obs = extract_static_features_batch(self.prev_obs)

            # Build action history batch: (n, max_hist, 7)
            ah_batch = self.ah_arrays[np.arange(n), players]  # (n, max_hist, 7)
            ah_lens_batch = self.ah_lens[np.arange(n), players]  # (n,)

            obs_t = torch.from_numpy(static_obs).to(self.device)
            ah_t = torch.from_numpy(ah_batch).to(self.device)
            ah_len_t = torch.from_numpy(ah_lens_batch).to(self.device)
            mask_t = torch.from_numpy(self.prev_mask.copy()).to(self.device)

            # Classify AS vs BR
            is_as = self.use_as[np.arange(n), players]
            as_idx = np.where(is_as)[0]
            br_idx = np.where(~is_as)[0]

            # Batched GPU inference
            actions_np = np.empty(n, dtype=np.int64)
            with torch.no_grad():
                if len(as_idx) > 0:
                    actions_np[as_idx] = self.as_net.select_action(
                        obs_t[as_idx], ah_t[as_idx], ah_len_t[as_idx], mask_t[as_idx]
                    ).cpu().numpy()
                if len(br_idx) > 0:
                    actions_np[br_idx] = self.br_net.select_action(
                        obs_t[br_idx], ah_t[br_idx], ah_len_t[br_idx], mask_t[br_idx], epsilon
                    ).cpu().numpy()

            # Snapshot pre-step state
            pre_static = static_obs  # already computed
            pre_mask = self.prev_mask.copy()
            pre_players = players.copy()
            pre_ah = ah_batch.copy()
            pre_ah_lens = ah_lens_batch.copy()

            # Batch step ALL envs
            action_pairs = [(i, int(actions_np[i])) for i in range(n)]
            next_players_list, next_obs_batch, next_masks_batch, rewards_batch, dones = (
                self.env.step_batch(action_pairs)
            )
            total_steps += n
            next_players_arr = np.array(next_players_list, dtype=np.intp)

            # Compute next static features in batch
            next_static = extract_static_features_batch(next_obs_batch)

            # Build next action history arrays (vectorized where possible)
            dones_arr = np.array(dones, dtype=bool)
            next_ah_all = np.zeros((n, max_hist, 7), dtype=np.float32)
            next_ah_lens_all = np.zeros(n, dtype=np.int64)
            alive = ~dones_arr
            if alive.any():
                alive_idx = np.where(alive)[0]
                alive_next_p = next_players_arr[alive_idx]
                next_ah_all[alive_idx] = self.ah_arrays[alive_idx, alive_next_p]
                next_ah_lens_all[alive_idx] = self.ah_lens[alive_idx, alive_next_p]

            # Gather per-player rewards
            player_rewards = rewards_batch[np.arange(n), pre_players]

            # Push BR transitions in batch (for non-AS envs)
            br_mask = ~is_as
            if br_mask.any():
                br_sel = np.where(br_mask)[0]
                self.br_buffer.push_batch(
                    obs=pre_static[br_sel],
                    action_history=pre_ah[br_sel],
                    history_length=pre_ah_lens[br_sel],
                    actions=actions_np[br_sel],
                    rewards=player_rewards[br_sel].astype(np.float32),
                    next_obs=next_static[br_sel],
                    next_action_history=next_ah_all[br_sel],
                    next_history_length=next_ah_lens_all[br_sel],
                    next_legal_mask=next_masks_batch[br_sel],
                    dones=dones_arr[br_sel].astype(np.float32),
                    legal_mask=pre_mask[br_sel],
                )

            # Push AS transitions in batch
            as_mask_arr = is_as
            if as_mask_arr.any():
                as_sel = np.where(as_mask_arr)[0]
                self.as_buffer.push_batch(
                    obs=pre_static[as_sel],
                    action_history=pre_ah[as_sel],
                    history_length=pre_ah_lens[as_sel],
                    actions=actions_np[as_sel],
                    legal_mask=pre_mask[as_sel],
                )

            # Update action histories for all envs (vectorized)
            action_records = _ACTION_TEMPLATES[actions_np].copy()  # (n, 7)
            action_records[:, 0] = pre_players / max(1, num_players - 1)
            self._append_actions_all(action_records)

            # Update state for alive envs
            if alive.any():
                alive_idx = np.where(alive)[0]
                self.prev_obs[alive_idx] = next_obs_batch[alive_idx]
                self.prev_mask[alive_idx] = next_masks_batch[alive_idx]
                self.prev_player[alive_idx] = next_players_arr[alive_idx]

            # Batch-reset done envs
            done_indices = np.where(dones_arr)[0]
            if len(done_indices) > 0:
                completed_episodes += len(done_indices)
                reset_players, reset_obs, reset_masks = self.env.reset_batch(done_indices.tolist())
                for j, idx in enumerate(done_indices):
                    self._reset_history(idx)
                    self.prev_obs[idx] = reset_obs[j]
                    self.prev_mask[idx] = reset_masks[j]
                    self.prev_player[idx] = reset_players[j]
                    self.use_as[idx] = np.random.random(num_players) < self.config.eta

        return total_steps
