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
_GAME_SLICE = slice(GAME_STATE_START, GAME_STATE_END)          # 86
_HAND_SLICE = slice(HAND_STRENGTH_START, HAND_STRENGTH_END)    # 52
# Column indices for vectorized extraction
_STATIC_COLS = np.concatenate([
    np.arange(HOLE_CARDS_START, COMMUNITY_END),
    np.arange(GAME_STATE_START, GAME_STATE_END),
    np.arange(HAND_STRENGTH_START, HAND_STRENGTH_END),
])


def extract_static_features_batch(obs_batch: np.ndarray) -> np.ndarray:
    """Vectorized: extract static features from (n, 630) -> (n, 502)."""
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
        br_inference: BestResponseNet | None = None,
        as_inference: AverageStrategyNet | None = None,
        pause_check: callable = None,
        checkpoint_pool=None,
    ):
        self.config = config
        # Use separate inference copies if provided (async training),
        # otherwise use the training networks directly (sync training).
        self.br_net = br_inference if br_inference is not None else br_net
        self.as_net = as_inference if as_inference is not None else as_net
        self.br_buffer = br_buffer
        self.as_buffer = as_buffer
        self.device = device
        self._pause_check = pause_check

        # Historical opponent pool (Pluribus-inspired)
        self._checkpoint_pool = checkpoint_pool
        self._historical_net: AverageStrategyNet | None = None
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
        self.hist_dim = config.history_input_dim
        self.pre_mask = np.empty((self.num_envs, config.num_actions), dtype=bool)
        self.pre_players = np.empty(self.num_envs, dtype=np.intp)
        self.ah_batch = np.empty((self.num_envs, self.max_hist, self.hist_dim), dtype=np.float32)
        self.ah_lens_batch = np.empty(self.num_envs, dtype=np.int64)
        self.actions_np = np.empty(self.num_envs, dtype=np.int64)
        self.action_records = np.empty((self.num_envs, self.hist_dim), dtype=np.float32)
        # Per-player pending transition tracking for correct BR transitions.
        # Instead of storing (obs, action, next_player_obs) on every step,
        # we defer BR buffer pushes until the same player acts again (non-terminal)
        # or the hand ends (terminal), giving correct same-player next_obs.
        self.pending_obs = np.zeros((config.num_envs, config.num_players, static_dim), dtype=np.float32)
        self.pending_ah = np.zeros(
            (config.num_envs, config.num_players, config.max_history_len, config.history_input_dim),
            dtype=np.float32,
        )
        self.pending_ah_len = np.zeros((config.num_envs, config.num_players), dtype=np.int64)
        self.pending_mask = np.zeros((config.num_envs, config.num_players, config.num_actions), dtype=bool)
        self.pending_action = np.zeros((config.num_envs, config.num_players), dtype=np.int64)
        self.pending_valid = np.zeros((config.num_envs, config.num_players), dtype=bool)
        self.pending_is_br = np.zeros((config.num_envs, config.num_players), dtype=bool)
        # Pre-allocated zero buffers for terminal BR flushes (avoids per-flush alloc)
        self._term_zero_obs = np.zeros((config.num_envs, static_dim), dtype=np.float32)
        self._term_zero_ah = np.zeros(
            (config.num_envs, config.max_history_len, config.history_input_dim), dtype=np.float32
        )
        self._term_zero_ah_len = np.zeros(config.num_envs, dtype=np.int64)
        self._term_zero_mask = np.zeros((config.num_envs, config.num_actions), dtype=bool)

        # Diverse opponent tracking: which seats use fixed exploit strategies
        self.exploit_prob = config.exploit_opponent_prob
        # Per-env per-seat: 0=learning, 1=always-raise, 2=calling-station,
        # 3=tight-fold, 4=historical checkpoint opponent
        self.exploit_type = np.zeros((config.num_envs, config.num_players), dtype=np.int8)
        # Remaining hands before exploit bot reverts to learning agent.
        # Exploit bots persist for 50-200 hands so EMA stats can reflect their behavior.
        self.exploit_remaining = np.zeros((config.num_envs, config.num_players), dtype=np.int32)
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
        self.pending_valid[:] = False
        self._assign_exploit_types(np.arange(self.num_envs))
        for i in range(self.num_envs):
            player, obs, mask = results[i]
            self.prev_obs[i] = obs
            self.prev_mask[i] = mask
            self.prev_player[i] = player
        self._initialized = True

    def _assign_exploit_types(self, env_indices: np.ndarray):
        """Manage exploit bot lifecycle: decrement counters and assign new ones.

        Exploit bots persist for 50-200 hands so EMA stats (α=0.02, ~50 hand
        window) have time to reflect their extreme behavior. The per-hand roll
        probability is adjusted so that the effective fraction of hands with an
        exploit opponent matches exploit_opponent_prob.

        Types: 1=always-raise, 2=calling-station, 3=tight-fold,
               4=historical checkpoint opponent (Pluribus-inspired).
        When the checkpoint pool has entries, ~50% of new stints use type 4.
        """
        n = len(env_indices)
        if self.exploit_prob <= 0.0 or self.num_players < 2:
            return

        remaining = self.exploit_remaining[env_indices]  # (n, num_players)
        types = self.exploit_type[env_indices]            # (n, num_players)

        # Decrement counters for active exploit bots
        active = remaining > 0
        remaining[active] -= 1

        # Seats that just expired revert to learning
        just_expired = (remaining == 0) & (types > 0)
        types[just_expired] = 0

        # For inactive seats (type==0), roll for new exploit assignment
        # Seat 0 is always learning
        inactive = types == 0
        inactive[:, 0] = False

        # Per-hand probability adjusted for duration so effective_fraction ≈ exploit_prob
        # effective_frac ≈ p_roll * avg_duration, so p_roll = target / avg_duration
        avg_duration = 125.0  # midpoint of [50, 200]
        p_roll = self.exploit_prob / avg_duration
        rolls = np.random.random((n, self.num_players))
        new_exploit = inactive & (rolls < p_roll)

        if new_exploit.any():
            pool_available = (
                self._checkpoint_pool is not None and len(self._checkpoint_pool) > 1
            )
            if pool_available:
                # 50% historical, 50% scripted (types 1-3)
                type_rolls = np.random.random(new_exploit.shape)
                hist_mask = new_exploit & (type_rolls < 0.5)
                script_mask = new_exploit & ~hist_mask

                if script_mask.any():
                    scripted = np.random.randint(1, 4, size=(n, self.num_players))
                    types[script_mask] = scripted[script_mask]
                if hist_mask.any():
                    types[hist_mask] = 4
                    self._load_historical_opponent()
            else:
                # No pool yet — all scripted
                new_types = np.random.randint(1, 4, size=(n, self.num_players))
                types[new_exploit] = new_types[new_exploit]

            # Duration: 50-200 hands (uniform), matching EMA effective window
            new_duration = np.random.randint(50, 201, size=(n, self.num_players))
            remaining[new_exploit] = new_duration[new_exploit]

        self.exploit_type[env_indices] = types
        self.exploit_remaining[env_indices] = remaining

    def _load_historical_opponent(self):
        """Load random checkpoint weights into the historical opponent network."""
        sd = self._checkpoint_pool.sample()
        if sd is None:
            return
        if self._historical_net is None:
            self._historical_net = AverageStrategyNet(self.config).to(self.device)
            self._historical_net.eval()
        self._historical_net.load_state_dict(sd)

    @staticmethod
    def _exploit_action(exploit_type: int, mask: np.ndarray) -> int:
        """Pick action for a fixed exploit strategy.
        Actions: 0=fold, 1=call/check, 2=min, 3=0.5x, 4=0.75x, 5=1x, 6=1.5x, 7=2x, 8=allin
        """
        if exploit_type == 1:  # always-raise: biggest raise available
            for a in (7, 6, 5, 4, 3, 2, 8):  # 2x, 1.5x, 1x, ..., allin
                if mask[a]:
                    return a
            return 1 if mask[1] else 0  # fall back to call/fold
        elif exploit_type == 2:  # calling-station: always call
            return 1 if mask[1] else 0
        elif exploit_type == 3:  # tight-fold: fold if facing bet, else check
            if mask[0]:  # fold available means facing a bet
                return 0
            return 1  # check
        return 1  # default: call

    def run_episodes(self, epsilon: float, eta: float | None = None) -> int:
        """Run continuous self-play until at least num_envs episodes complete.

        Uses per-player pending transition tracking so that BR buffer entries
        always pair the same player's obs with that player's next obs (or a
        terminal reward), instead of the wrong-player obs returned by the env.
        """
        if eta is None:
            eta = self.config.eta_start
        if not self._initialized:
            self._init_envs(eta)

        n = self.num_envs
        num_players = self.num_players
        env_idx = self.env_idx
        static_dim = self.static_obs.shape[1]
        total_steps = 0
        completed_episodes = 0

        while completed_episodes < n:
            # Check for pause request (async training: weight sync, eval)
            if self._pause_check is not None:
                self._pause_check()

            # Prepare full batch tensors for the current player in each env
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

            # Classify AS vs BR for the current player in each env
            is_as = self.use_as[env_idx, players]
            # Check which current actors are exploit bots
            cur_exploit = self.exploit_type[env_idx, players]
            is_exploit = cur_exploit > 0
            is_scripted = (cur_exploit >= 1) & (cur_exploit <= 3)
            is_historical = cur_exploit == 4
            is_learning = ~is_exploit
            as_idx = np.where(is_as & is_learning)[0]
            br_idx = np.where(~is_as & is_learning)[0]
            scripted_idx = np.where(is_scripted)[0]
            historical_idx = np.where(is_historical)[0]

            # Batched GPU inference — single CPU sync
            actions_np = self.actions_np
            with torch.inference_mode():
                actions_gpu = torch.empty(n, dtype=torch.long, device=self.device)
                if len(as_idx) > 0:
                    actions_gpu[as_idx] = self.as_net.select_action(
                        obs_t[as_idx], ah_t[as_idx], ah_len_t[as_idx], mask_t[as_idx]
                    )
                if len(br_idx) > 0:
                    # Single forward pass: get both epsilon-greedy (for play/BR buffer)
                    # and greedy (for AS buffer — no exploration noise)
                    eps_greedy_br, greedy_br = self.br_net.select_action_with_greedy(
                        obs_t[br_idx], ah_t[br_idx], ah_len_t[br_idx], mask_t[br_idx], epsilon
                    )
                    actions_gpu[br_idx] = eps_greedy_br
                    greedy_br_np = greedy_br.cpu().numpy()
                # Historical opponents: GPU inference via historical net
                if len(historical_idx) > 0 and self._historical_net is not None:
                    actions_gpu[historical_idx] = self._historical_net.select_action(
                        obs_t[historical_idx], ah_t[historical_idx],
                        ah_len_t[historical_idx], mask_t[historical_idx],
                    )
                elif len(historical_idx) > 0:
                    # Fallback: pool empty or net not loaded — use AS net
                    actions_gpu[historical_idx] = self.as_net.select_action(
                        obs_t[historical_idx], ah_t[historical_idx],
                        ah_len_t[historical_idx], mask_t[historical_idx],
                    )
                # Scripted exploit bots get placeholder (overwritten below)
                if len(scripted_idx) > 0:
                    actions_gpu[scripted_idx] = 1
                actions_np[:] = actions_gpu.cpu().numpy()

            # Override actions for scripted exploit bot seats
            for ei in scripted_idx:
                etype = int(self.exploit_type[ei, players[ei]])
                actions_np[ei] = self._exploit_action(etype, self.pre_mask[ei])

            pre_players = self.pre_players  # safe copy from above

            # --- Flush pending non-terminal BR transitions ---
            # When the same player acts again, their pending transition gets
            # the correct next_state (this player's current obs).
            # Skip exploit seats — their transitions don't go to buffers.
            has_pending_br = (
                self.pending_valid[env_idx, pre_players]
                & self.pending_is_br[env_idx, pre_players]
                & is_learning
            )
            if has_pending_br.any():
                sel = np.where(has_pending_br)[0]
                sel_p = pre_players[sel]
                n_sel = len(sel)
                self.br_buffer.push_batch(
                    obs=self.pending_obs[sel, sel_p],
                    action_history=self.pending_ah[sel, sel_p],
                    history_length=self.pending_ah_len[sel, sel_p],
                    actions=self.pending_action[sel, sel_p],
                    rewards=np.zeros(n_sel, dtype=np.float32),
                    next_obs=self.static_obs[sel],
                    next_action_history=self.ah_batch[sel],
                    next_history_length=self.ah_lens_batch[sel],
                    next_legal_mask=self.pre_mask[sel],
                    dones=np.zeros(n_sel, dtype=np.float32),
                    legal_mask=self.pending_mask[sel, sel_p],
                )

            # --- Store current state as pending for the acting player ---
            # Exploit seats still get tracked for pending_valid (for env stepping)
            # but marked so their transitions are skipped at flush time.
            self.pending_obs[env_idx, pre_players] = self.static_obs
            self.pending_ah[env_idx, pre_players] = self.ah_batch
            self.pending_ah_len[env_idx, pre_players] = self.ah_lens_batch
            self.pending_mask[env_idx, pre_players] = self.pre_mask
            self.pending_action[env_idx, pre_players] = actions_np
            self.pending_valid[env_idx, pre_players] = True
            self.pending_is_br[env_idx, pre_players] = ~is_as & is_learning

            # --- Push to AS buffer for learning BR-policy actions ---
            learning_br_idx = br_idx
            if learning_br_idx.size > 0:
                self.as_buffer.push_batch(
                    obs=self.static_obs[learning_br_idx],
                    action_history=self.ah_batch[learning_br_idx],
                    history_length=self.ah_lens_batch[learning_br_idx],
                    actions=greedy_br_np,
                    legal_mask=self.pre_mask[learning_br_idx],
                )

            # --- Step all envs ---
            next_players_arr, next_obs_batch, next_masks_batch, rewards_batch, dones_arr = (
                self.env.step_batch_dense(actions_np)
            )
            total_steps += n

            # Update action histories with the just-taken action
            np.take(_ACTION_TEMPLATES, actions_np, axis=0, out=self.action_records)
            self.action_records[:, 0] = pre_players / max(1, num_players - 1)
            self._append_actions_all(self.action_records)

            # --- Flush terminal BR transitions for done envs ---
            # Every player who acted this hand (pending_valid) and used BR
            # gets a terminal transition with their actual reward.
            done_indices = np.where(dones_arr)[0]
            if len(done_indices) > 0:
                completed_episodes += len(done_indices)
                for p in range(num_players):
                    # Skip exploit bot seats for buffer pushes
                    is_exploit_p = self.exploit_type[done_indices, p] > 0
                    has_p = (
                        self.pending_valid[done_indices, p]
                        & self.pending_is_br[done_indices, p]
                        & ~is_exploit_p
                    )
                    if has_p.any():
                        sel = done_indices[has_p]
                        n_sel = len(sel)
                        self.br_buffer.push_batch(
                            obs=self.pending_obs[sel, p],
                            action_history=self.pending_ah[sel, p],
                            history_length=self.pending_ah_len[sel, p],
                            actions=self.pending_action[sel, p],
                            rewards=rewards_batch[sel, p],
                            next_obs=self._term_zero_obs[:n_sel],
                            next_action_history=self._term_zero_ah[:n_sel],
                            next_history_length=self._term_zero_ah_len[:n_sel],
                            next_legal_mask=self._term_zero_mask[:n_sel],
                            dones=np.ones(n_sel, dtype=np.float32),
                            legal_mask=self.pending_mask[sel, p],
                        )
                # Clear pending for done envs
                self.pending_valid[done_indices] = False

                # Reset done envs
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
                self._assign_exploit_types(done_indices)

            # Update state for alive envs
            alive = ~dones_arr
            if alive.any():
                alive_idx = np.where(alive)[0]
                self.prev_obs[alive_idx] = next_obs_batch[alive_idx]
                self.prev_mask[alive_idx] = next_masks_batch[alive_idx]
                self.prev_player[alive_idx] = next_players_arr[alive_idx]

        return total_steps
