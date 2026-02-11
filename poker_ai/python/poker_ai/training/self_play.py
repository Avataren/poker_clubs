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
from poker_ai.training.circular_buffer import CircularBuffer
from poker_ai.training.reservoir import ReservoirBuffer
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
    """Vectorized: extract static features from (n, 579) -> (n, 451)."""
    return obs_batch[:, _STATIC_COLS]


def pad_action_history(history: list[np.ndarray], max_len: int = 30) -> tuple[np.ndarray, int]:
    """Pad action history to fixed length. Returns (padded[max_len, 11], length)."""
    length = min(len(history), max_len)
    padded = np.zeros((max_len, 11), dtype=np.float32)
    if length > 0:
        padded[:length] = history[-length:]  # take most recent entries
    return padded, length


# Pre-computed action record templates (9 actions × 11 dims)
# Format: [seat_norm, 9× action one-hot, bet_size/pot]
_ACTION_TEMPLATES = np.zeros((9, 11), dtype=np.float32)
for _i in range(9):
    _ACTION_TEMPLATES[_i, 1 + _i] = 1.0  # one-hot at position 1+action_idx
    _ACTION_TEMPLATES[_i, 10] = min(_i / 8.0, 1.0)  # rough bet ratio


def make_action_record(player: int, action: int, num_players: int) -> np.ndarray:
    """Create an 11-dim action record."""
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
        self.use_cuda_transfer = self.device.type == "cuda"

        # Per-env state (always active)
        self.use_as = np.zeros((config.num_envs, config.num_players), dtype=bool)
        self.prev_obs = np.zeros((config.num_envs, config.input_dim), dtype=np.float32)
        self.prev_mask = np.zeros((config.num_envs, config.num_actions), dtype=bool)
        self.prev_player = np.zeros(config.num_envs, dtype=np.intp)
        # Action histories stored as fixed-size ring buffers per env per player
        self.ah_arrays = np.zeros(
            (config.num_envs, config.num_players, config.max_history_len, config.history_input_dim),
            dtype=np.float32,
        )
        self.ah_lens = np.zeros(
            (config.num_envs, config.num_players), dtype=np.int64
        )
        # Next write slot per env/player in the ring.
        self.ah_pos = np.zeros(
            (config.num_envs, config.num_players), dtype=np.int64
        )
        # Reusable scratch buffers for the hot self-play loop.
        static_dim = len(_STATIC_COLS)
        self.env_idx = np.arange(self.num_envs, dtype=np.intp)
        self.hist_offsets = np.arange(self.max_hist, dtype=np.int64)
        self.static_obs = np.empty((self.num_envs, static_dim), dtype=np.float32)
        self.next_static = np.empty((self.num_envs, static_dim), dtype=np.float32)
        self.hist_dim = config.history_input_dim
        self.pre_mask = np.empty((self.num_envs, config.num_actions), dtype=bool)
        self.pre_players = np.empty(self.num_envs, dtype=np.intp)
        self.ah_batch = np.empty((self.num_envs, self.max_hist, self.hist_dim), dtype=np.float32)
        self.ah_lens_batch = np.empty(self.num_envs, dtype=np.int64)
        self.next_ah_all = np.empty((self.num_envs, self.max_hist, self.hist_dim), dtype=np.float32)
        self.next_ah_lens_all = np.empty(self.num_envs, dtype=np.int64)
        self.actions_np = np.empty(self.num_envs, dtype=np.int64)
        self.action_records = np.empty((self.num_envs, self.hist_dim), dtype=np.float32)
        self._initialized = False

    def _get_history(self, env: int, player: int) -> tuple[np.ndarray, int]:
        """Get (padded_history, length) for an env/player."""
        env_idx = np.array([env], dtype=np.intp)
        player_idx = np.array([player], dtype=np.intp)
        histories, lengths = self._gather_histories(env_idx, player_idx)
        return histories[0], int(lengths[0])

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        """Move a numpy array to device, enabling async copies on CUDA via pinned memory."""
        tensor = torch.from_numpy(array)
        if self.use_cuda_transfer:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.use_cuda_transfer)

    def _gather_histories(
        self, env_indices: np.ndarray, players: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather chronological [max_hist, hist_dim] histories for env/player pairs."""
        n = len(env_indices)
        max_hist = self.max_hist
        out = np.empty((n, max_hist, self.hist_dim), dtype=np.float32)
        lengths = np.empty(n, dtype=np.int64)
        self._gather_histories_into(env_indices, players, out, lengths)
        return out, lengths

    def _gather_histories_into(
        self,
        env_indices: np.ndarray,
        players: np.ndarray,
        out: np.ndarray,
        lengths: np.ndarray,
    ):
        """Gather chronological [max_hist, hist_dim] histories into preallocated outputs."""
        n = len(env_indices)
        max_hist = self.max_hist
        out.fill(0.0)
        if n == 0:
            lengths.fill(0)
            return

        np.copyto(lengths, self.ah_lens[env_indices, players], casting="unsafe")
        starts = (self.ah_pos[env_indices, players] - lengths) % max_hist
        offsets = self.hist_offsets
        gather_idx = (starts[:, None] + offsets[None, :]) % max_hist
        ring = self.ah_arrays[env_indices, players]
        gathered = ring[np.arange(n)[:, None], gather_idx]
        valid = offsets[None, :] < lengths[:, None]
        out[valid] = gathered[valid]

    def _append_actions_all(self, action_records: np.ndarray):
        """Append action records for all envs in vectorized O(players * envs) time."""
        n = len(action_records)
        max_hist = self.max_hist
        num_players = self.num_players
        env_idx = self.env_idx[:n]
        for p in range(num_players):
            pos = self.ah_pos[env_idx, p]
            self.ah_arrays[env_idx, p, pos] = action_records
            self.ah_lens[env_idx, p] = np.minimum(self.ah_lens[env_idx, p] + 1, max_hist)
            self.ah_pos[env_idx, p] = (pos + 1) % max_hist

    def _reset_history(self, env: int):
        """Reset action histories for an env."""
        self.ah_arrays[env] = 0.0
        self.ah_lens[env] = 0
        self.ah_pos[env] = 0

    def _init_envs(self, eta: float | None = None):
        if eta is None:
            eta = self.config.eta_start
        results = self.env.reset_all()
        self.use_as[:] = np.random.random((self.num_envs, self.num_players)) < eta
        self.ah_arrays[:] = 0.0
        self.ah_lens[:] = 0
        self.ah_pos[:] = 0
        for i in range(self.num_envs):
            player, obs, mask = results[i]
            self.prev_obs[i] = obs
            self.prev_mask[i] = mask
            self.prev_player[i] = player
        self._initialized = True

    def run_episodes(self, epsilon: float, eta: float | None = None) -> int:
        """Run continuous self-play until at least num_envs episodes complete."""
        if eta is None:
            eta = self.config.eta_start
        if not self._initialized:
            self._init_envs(eta)

        n = self.num_envs
        num_players = self.num_players
        env_idx = self.env_idx
        total_steps = 0
        completed_episodes = 0

        while completed_episodes < n:
            # Prepare full batch tensors
            players = self.prev_player
            np.take(self.prev_obs, _STATIC_COLS, axis=1, out=self.static_obs)

            # Build action history batch: (n, max_hist, hist_dim)
            self._gather_histories_into(env_idx, players, self.ah_batch, self.ah_lens_batch)
            np.copyto(self.pre_mask, self.prev_mask)
            np.copyto(self.pre_players, players)

            obs_t = self._to_device(self.static_obs)
            ah_t = self._to_device(self.ah_batch)
            ah_len_t = self._to_device(self.ah_lens_batch)
            mask_t = self._to_device(self.pre_mask)

            # Classify AS vs BR
            is_as = self.use_as[env_idx, players]
            as_idx = np.where(is_as)[0]
            br_idx = np.where(~is_as)[0]

            # Batched GPU inference
            actions_np = self.actions_np
            with torch.inference_mode():
                if len(as_idx) > 0:
                    actions_np[as_idx] = self.as_net.select_action(
                        obs_t[as_idx], ah_t[as_idx], ah_len_t[as_idx], mask_t[as_idx]
                    ).cpu().numpy()
                if len(br_idx) > 0:
                    actions_np[br_idx] = self.br_net.select_action(
                        obs_t[br_idx], ah_t[br_idx], ah_len_t[br_idx], mask_t[br_idx], epsilon
                    ).cpu().numpy()

            # Snapshot pre-step state
            pre_static = self.static_obs
            pre_mask = self.pre_mask
            pre_players = self.pre_players
            pre_ah = self.ah_batch
            pre_ah_lens = self.ah_lens_batch

            # Batch step ALL envs
            next_players_arr, next_obs_batch, next_masks_batch, rewards_batch, dones_arr = (
                self.env.step_batch_dense(actions_np)
            )
            total_steps += n

            # Compute next static features in batch
            np.take(next_obs_batch, _STATIC_COLS, axis=1, out=self.next_static)

            # Update action histories with the just-taken action before computing next state histories.
            np.take(_ACTION_TEMPLATES, actions_np, axis=0, out=self.action_records)
            self.action_records[:, 0] = pre_players / max(1, num_players - 1)
            self._append_actions_all(self.action_records)

            # Build next action history arrays (vectorized where possible)
            self.next_ah_all.fill(0.0)
            self.next_ah_lens_all.fill(0)
            alive = ~dones_arr
            if alive.any():
                alive_idx = np.where(alive)[0]
                alive_next_p = next_players_arr[alive_idx]
                alive_histories, alive_lengths = self._gather_histories(
                    alive_idx, alive_next_p
                )
                self.next_ah_all[alive_idx] = alive_histories
                self.next_ah_lens_all[alive_idx] = alive_lengths

            # Gather per-player rewards (normalize to big blinds)
            player_rewards = rewards_batch[env_idx, pre_players] / self.config.big_blind

            # Push BR transitions in batch (for non-AS envs)
            br_mask = ~is_as
            if br_mask.any():
                br_sel = np.where(br_mask)[0]
                self.br_buffer.push_batch(
                    obs=pre_static[br_sel],
                    action_history=pre_ah[br_sel],
                    history_length=pre_ah_lens[br_sel],
                    actions=actions_np[br_sel],
                    rewards=player_rewards[br_sel],
                    next_obs=self.next_static[br_sel],
                    next_action_history=self.next_ah_all[br_sel],
                    next_history_length=self.next_ah_lens_all[br_sel],
                    next_legal_mask=next_masks_batch[br_sel],
                    dones=dones_arr[br_sel].astype(np.float32),
                    legal_mask=pre_mask[br_sel],
                )

                # NFSP supervised buffer should train on BR behavior-policy actions.
                self.as_buffer.push_batch(
                    obs=pre_static[br_sel],
                    action_history=pre_ah[br_sel],
                    history_length=pre_ah_lens[br_sel],
                    actions=actions_np[br_sel],
                    legal_mask=pre_mask[br_sel],
                )

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
                reset_players, reset_obs, reset_masks = self.env.reset_batch_dense(done_indices)
                self.ah_arrays[done_indices] = 0.0
                self.ah_lens[done_indices] = 0
                self.ah_pos[done_indices] = 0
                self.prev_obs[done_indices] = reset_obs
                self.prev_mask[done_indices] = reset_masks
                self.prev_player[done_indices] = reset_players
                self.use_as[done_indices] = (
                    np.random.random((len(done_indices), num_players)) < eta
                )

        return total_steps
