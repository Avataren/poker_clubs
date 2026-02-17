"""NFSP (Neural Fictitious Self-Play) training loop."""

import copy
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.model.network import BestResponseNet, AverageStrategyNet
from poker_ai.training.circular_buffer import CircularBuffer
from poker_ai.training.reservoir import ReservoirBuffer
from poker_ai.training.self_play import SelfPlayWorker, extract_static_features_batch, pad_action_history, make_action_record, _STATIC_COLS


class CheckpointPool:
    """Pool of historical network weights for diverse opponent training.

    Stores AS network state_dicts (CPU tensors) from periodic training
    snapshots. Self-play workers sample from this pool to create historical
    opponents, providing realistic diversity without scripted bots.
    Inspired by Pluribus (Brown & Sandholm, 2019).
    """

    def __init__(self, max_size: int = 10):
        self._pool: deque[dict] = deque(maxlen=max_size)

    def snapshot(self, as_net: nn.Module):
        """Save a copy of the AS network's current weights to the pool."""
        sd = {k: v.cpu().clone() for k, v in as_net.state_dict().items()}
        self._pool.append(sd)

    def sample(self) -> dict | None:
        """Return a random state_dict from the pool, or None if empty."""
        if not self._pool:
            return None
        idx = np.random.randint(len(self._pool))
        return self._pool[idx]

    def __len__(self) -> int:
        return len(self._pool)
from poker_ai.env.poker_env import PokerEnv, BatchPokerEnv
from poker_ai.model.state_encoder import HAND_STRENGTH_START, STATIC_FEATURE_SIZE


# --- Card augmentation (suit perm × hole swap × flop perm) ---
# Card encoding: 7 slots × 52 one-hot (indices 0-363 in static features).
#   Slots: [hole0, hole1, flop0, flop1, flop2, turn, river]
# Within each 52-dim slot, layout is suit*13 + rank_offset (4 suits × 13 ranks).
#
# Valid symmetries (all independent, composable):
#   1. Suit permutation (24): suits are interchangeable in Hold'em
#   2. Hole card swap (2): hole card order doesn't matter
#   3. Flop permutation (6): flop card order doesn't matter
# Total: 24 × 2 × 6 = 288 augmentations (minus identity = 287)
#
# We precompute all 288 index arrays for the card region at import time.

def _build_card_augmentation_indices() -> np.ndarray:
    """Build (288, 364) index arrays for all card augmentations."""
    from itertools import permutations
    suit_perms = list(permutations(range(4)))       # 24
    hole_perms = [(0, 1), (1, 0)]                   # 2
    flop_perms = list(permutations(range(3)))        # 6

    num_aug = len(suit_perms) * len(hole_perms) * len(flop_perms)  # 288
    num_card_features = 364  # 7 slots × 52
    indices = np.zeros((num_aug, num_card_features), dtype=np.int64)

    idx = 0
    for suit_perm in suit_perms:
        # Build suit remapping for one 52-dim slot
        slot_remap = np.zeros(52, dtype=np.int64)
        for old_suit in range(4):
            new_suit = suit_perm[old_suit]
            for rank in range(13):
                slot_remap[new_suit * 13 + rank] = old_suit * 13 + rank

        for hole_perm in hole_perms:
            for flop_perm in flop_perms:
                # Slot reordering: [hole0, hole1, flop0, flop1, flop2, turn, river]
                slot_order = [
                    hole_perm[0], hole_perm[1],             # hole cards
                    2 + flop_perm[0], 2 + flop_perm[1], 2 + flop_perm[2],  # flop
                    5, 6,                                    # turn, river (fixed)
                ]
                for new_slot, old_slot in enumerate(slot_order):
                    dst_base = new_slot * 52
                    src_base = old_slot * 52
                    indices[idx, dst_base:dst_base + 52] = src_base + slot_remap
                idx += 1

    return indices

_CARD_AUG_INDICES = _build_card_augmentation_indices()  # (288, 364)


def apply_card_augmentation(obs: torch.Tensor, aug_idx: int | None = None) -> torch.Tensor:
    """Apply a random card augmentation (suit perm × hole swap × flop perm).

    Args:
        obs: (batch, STATIC_FEATURE_SIZE) tensor
        aug_idx: optional augmentation index (1-287). If None, picks randomly.
                 Pass the same value for obs and next_obs in DQN transitions
                 to preserve temporal consistency.
    Returns:
        augmented obs with permuted card features (in-place safe)
    """
    if aug_idx is None:
        aug_idx = np.random.randint(1, 288)  # skip identity (0)
    idx = _CARD_AUG_INDICES[aug_idx]  # (364,)
    aug = obs.clone()
    aug[:, :364] = obs[:, idx]
    return aug


@dataclass
class EvalStats:
    """Summary statistics for heads-up evaluation."""
    bb100: float
    ci95: float
    std_bb100: float
    num_hands: int
    seat0_bb100: float
    seat1_bb100: float
    bluff_pct: float = 0.0         # raises with hand_strength < 0.3
    thin_value_pct: float = 0.0    # raises with 0.3 <= strength < 0.6
    value_bet_pct: float = 0.0     # raises with strength >= 0.6
    # HUD stats
    vpip: float = 0.0              # voluntarily put $ in pot (%)
    pfr: float = 0.0               # preflop raise (%)
    aggression: float = 0.0        # raises / (raises + calls + checks)
    wtsd: float = 0.0              # went to showdown (%)
    cbet: float = 0.0              # continuation bet (%)
    # Per-street action distributions
    flop_fold_pct: float = 0.0
    flop_call_pct: float = 0.0
    flop_raise_pct: float = 0.0
    turn_fold_pct: float = 0.0
    turn_call_pct: float = 0.0
    turn_raise_pct: float = 0.0
    river_fold_pct: float = 0.0
    river_call_pct: float = 0.0
    river_raise_pct: float = 0.0
    # Bet sizing
    avg_bet_size: float = 0.0      # mean bet/pot ratio when raising
    # Per-street bluff rates
    flop_bluff_pct: float = 0.0
    turn_bluff_pct: float = 0.0
    river_bluff_pct: float = 0.0
    # Showdown & fold-to-bet
    showdown_pct: float = 0.0      # hands reaching showdown (%)
    fold_to_preflop_bet: float = 0.0  # fold when facing preflop raise (%)
    fold_to_flop_bet: float = 0.0  # fold when facing flop bet (%)
    fold_to_turn_bet: float = 0.0
    fold_to_river_bet: float = 0.0


@dataclass
class MultiwayEvalStats:
    """Summary statistics for multiway (6-max / 9-ring) evaluation."""
    bb100: float
    ci95: float
    std_bb100: float
    num_hands: int
    position_bb100: dict[str, float]   # per-position winrate
    vpip: float                        # voluntarily put $ in pot (%)
    pfr: float                         # preflop raise (%)
    three_bet: float                   # 3-bet preflop (%)
    steal_attempt: float               # open-raise from CO/BTN (%)
    fold_to_steal: float               # fold in blinds vs steal (%)
    action_pcts: dict[str, float]      # action distribution (%)


# Position labels by table size, indexed by relative position from dealer
# rel=0 is dealer (BTN), rel=1 is SB, rel=2 is BB, rel=3+ are early/middle positions
_POSITION_LABELS = {
    2: {0: "BTN/SB", 1: "BB"},
    6: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"},
    9: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "UTG+1", 5: "UTG+2",
        6: "LJ", 7: "HJ", 8: "CO"},
}

# Canonical display order (early → late → blinds)
_POSITION_ORDER = {
    6: ["UTG", "MP", "CO", "BTN", "SB", "BB"],
    9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}


def _get_position_names(num_players: int) -> list[str]:
    """Get position labels in display order (early → late → blinds)."""
    if num_players in _POSITION_ORDER:
        return _POSITION_ORDER[num_players]
    return [f"Seat{i}" for i in range(num_players)]


def _seat_to_position_name(seat: int, dealer: int, num_players: int) -> str:
    """Convert absolute seat to position name given current dealer."""
    rel = (seat - dealer) % num_players
    labels = _POSITION_LABELS.get(num_players)
    if labels and rel in labels:
        return labels[rel]
    return f"Seat{rel}"


class NFSPTrainer:
    """NFSP training manager."""

    def __init__(self, config: NFSPConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.use_cuda_transfer = self.device.type == "cuda"
        if config.use_amp is None:
            self.use_amp = self.device.type == "cuda"
        else:
            self.use_amp = bool(config.use_amp) and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        )
        torch.set_float32_matmul_precision("high")
        print(f"Using device: {self.device}")
        if self.use_amp:
            print(f"AMP enabled: dtype={self.amp_dtype}")
        print(f"Model: hidden={config.hidden_dim}, residual={config.residual_dim}, "
              f"history_hidden={config.history_hidden_dim}, batch={config.batch_size}")
        if config.freeze_as:
            print("AS network FROZEN permanently — skipping all AS gradient updates")
        elif config.as_freeze_duration > 0:
            print(f"AS network frozen for first {config.as_freeze_duration:,} episodes after resume")

        # AS freeze tracking (set actual unfreeze episode in load_checkpoint)
        self._as_unfreeze_episode = 0  # 0 = no freeze active
        self._as_optimizer_reset_pending = False  # reset optimizer on unfreeze
        self._as_warmup_start_episode = 0  # episode at which AS LR warmup begins
        self._as_warmup_end_episode = 0    # episode at which AS LR warmup ends

        # Networks
        self.br_net = BestResponseNet(config).to(self.device)
        self.br_target = copy.deepcopy(self.br_net).to(self.device)
        self.br_target.eval()
        self.as_net = AverageStrategyNet(config).to(self.device)

        br_params = sum(p.numel() for p in self.br_net.parameters())
        as_params = sum(p.numel() for p in self.as_net.parameters())
        print(f"Parameters: BR={br_params:,}, AS={as_params:,}, Total={br_params+as_params:,}")

        # torch.compile for kernel fusion (ROCm/CUDA)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision('high')
            try:
                self.br_net = torch.compile(self.br_net)
                self.br_target = torch.compile(self.br_target)
                self.as_net = torch.compile(self.as_net)
                print("torch.compile enabled for all networks")
            except Exception as e:
                print(f"torch.compile not available: {e}")

        # Optimizers
        self.br_optimizer = torch.optim.Adam(self.br_net.parameters(), lr=config.br_lr)
        self.as_optimizer = torch.optim.Adam(self.as_net.parameters(), lr=config.as_lr)
        # GradScaler is only needed for float16 (narrow exponent range).
        # bfloat16 has the same exponent range as float32, so scaling is unnecessary
        # and can actually cause NaN by producing inf scaled gradients.
        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        scaler_device = "cuda" if use_scaler else "cpu"
        self.br_scaler = torch.amp.GradScaler(scaler_device, enabled=use_scaler)
        self.as_scaler = torch.amp.GradScaler(scaler_device, enabled=use_scaler)

        # LR schedulers (cosine decay with linear warmup)
        self._lr_warmup_steps = config.lr_warmup_steps
        self._lr_min_factor = config.lr_min_factor
        # Total training steps approximation for cosine period
        # (episodes * ~10 steps/episode is a rough estimate; we use env steps directly)
        self._lr_total_steps = config.epsilon_decay_steps  # reuse as LR schedule horizon

        # Replay buffers
        from poker_ai.model.state_encoder import STATIC_FEATURE_SIZE
        self.br_buffer = CircularBuffer(
            config.br_buffer_size,
            obs_dim=STATIC_FEATURE_SIZE,
            max_seq_len=config.max_history_len,
            num_actions=config.num_actions,
            history_dim=config.history_input_dim,
        )
        self.as_buffer = ReservoirBuffer(
            config.as_buffer_size,
            obs_dim=STATIC_FEATURE_SIZE,
            max_seq_len=config.max_history_len,
            num_actions=config.num_actions,
            history_dim=config.history_input_dim,
        )

        # Self-play worker
        self.checkpoint_pool = CheckpointPool(max_size=10)
        self.worker = SelfPlayWorker(
            config, self.br_net, self.as_net, self.br_buffer, self.as_buffer, self.device,
            checkpoint_pool=self.checkpoint_pool,
        )

        # Logging
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)

        # Counters
        self.total_steps = 0
        self.total_episodes = 0
        self.br_updates = 0
        self.as_updates = 0
        self._schedule_step_offset = 0  # for warm restarts: schedules use (total_steps - offset)
        self._last_br_grad_norm = 0.0
        self._last_as_grad_norm = 0.0

    def is_as_frozen(self) -> bool:
        """Check if AS training is currently frozen."""
        if self.config.freeze_as:
            return True
        if self._as_unfreeze_episode > 0 and self.total_episodes < self._as_unfreeze_episode:
            return True
        # Reset AS optimizer on unfreeze transition so stale Adam momentum
        # from the checkpoint doesn't cause catastrophic updates on the new
        # buffer data that accumulated during the freeze.
        if self._as_optimizer_reset_pending:
            self._as_optimizer_reset_pending = False
            self.as_optimizer = torch.optim.Adam(
                self.as_net.parameters(), lr=self.config.as_lr
            )
            # Start AS LR warmup: ramp from 1% to 100% over warmup_episodes
            if self.config.as_warmup_episodes > 0:
                self._as_warmup_start_episode = self.total_episodes
                self._as_warmup_end_episode = self.total_episodes + self.config.as_warmup_episodes
                print(f"  AS optimizer reset + LR warmup over {self.config.as_warmup_episodes:,} episodes")
            else:
                print("  AS optimizer reset (stale Adam state discarded)")
            # Apply LR schedule immediately so the first training step uses
            # the correct (warmup-scaled) LR, not the raw base LR.
            self._update_lr()
        return False

    def restart_schedules(self, reset_optimizers: bool = False):
        """Warm restart: reset LR, epsilon, and eta schedules to their start values.

        Sets the schedule offset to current total_steps so all schedules
        recompute from step 0. The model weights and buffers are preserved.
        Optionally resets Adam optimizer state (clears momentum).
        """
        self._schedule_step_offset = self.total_steps
        print(f"  Schedule warm restart at step {self.total_steps:,} "
              f"(eps={self.config.epsilon_start}, eta={self.config.eta_start}, "
              f"lr factor=1.0)")
        if reset_optimizers:
            self.br_optimizer = torch.optim.Adam(
                self.br_net.parameters(), lr=self.config.br_lr
            )
            self.as_optimizer = torch.optim.Adam(
                self.as_net.parameters(), lr=self.config.as_lr
            )
            print("  Optimizers reset (Adam momentum cleared)")
        self._update_lr()

    def _schedule_steps(self) -> int:
        """Steps for schedule computations, accounting for warm restart offset."""
        return self.total_steps - self._schedule_step_offset

    def get_epsilon(self) -> float:
        """Get current epsilon for exploration."""
        progress = min(self._schedule_steps() / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def get_eta(self) -> float:
        """Get current eta (anticipatory parameter) with linear ramp.

        eta = P(use AS policy) during self-play. A player uses AS with
        probability eta, BR with probability (1-eta). BR transitions
        are collected only from BR-policy actions.
        """
        progress = min(self._schedule_steps() / max(self.config.eta_ramp_steps, 1), 1.0)
        return self.config.eta_start + progress * (
            self.config.eta_end - self.config.eta_start
        )

    def _get_lr_factor(self) -> float:
        """Cosine decay with linear warmup. Returns multiplier in [min_factor, 1.0]."""
        steps = self._schedule_steps()
        warmup = self._lr_warmup_steps
        if warmup > 0 and steps < warmup:
            return max(steps / warmup, 0.01)  # linear warmup from 1% to 100%
        # Cosine decay from 1.0 to min_factor
        progress = min((steps - warmup) / max(self._lr_total_steps - warmup, 1), 1.0)
        min_f = self._lr_min_factor
        return min_f + 0.5 * (1.0 - min_f) * (1.0 + math.cos(math.pi * progress))

    def _update_lr(self):
        """Apply LR schedule to both optimizers."""
        factor = self._get_lr_factor()
        for pg in self.br_optimizer.param_groups:
            pg["lr"] = self.config.br_lr * factor
        # AS LR warmup after unfreeze: ramp from 1% to 100% over warmup period
        as_factor = factor
        if self._as_warmup_end_episode > 0 and self.total_episodes < self._as_warmup_end_episode:
            warmup_progress = max(
                (self.total_episodes - self._as_warmup_start_episode)
                / max(self._as_warmup_end_episode - self._as_warmup_start_episode, 1),
                0.0,
            )
            as_factor = factor * (0.01 + 0.99 * warmup_progress)
        for pg in self.as_optimizer.param_groups:
            pg["lr"] = self.config.as_lr * as_factor

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        """Move numpy array to training device with optional pinned-memory staging."""
        tensor = torch.from_numpy(array)
        if self.use_cuda_transfer:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.use_cuda_transfer)

    def train_br_step(self) -> float:
        """One DQN training step on Best Response network."""
        # Require 5% of buffer capacity before training to avoid noisy updates
        # after resume when buffers are nearly empty
        min_samples = max(self.config.batch_size, int(0.05 * self.config.br_buffer_size))
        if len(self.br_buffer) < min_samples:
            return 0.0

        # Sample directly as numpy arrays, convert to tensors (no Python loop)
        (obs_np, ah_np, ah_len_np, actions_np, rewards_np, next_obs_np,
         next_ah_np, next_ah_len_np, next_mask_np, dones_np, masks_np
        ) = self.br_buffer.sample_arrays(self.config.batch_size)

        obs = self._to_device(obs_np)
        ah = self._to_device(ah_np)
        ah_len = self._to_device(ah_len_np)
        actions = self._to_device(actions_np)
        rewards = self._to_device(rewards_np)
        next_obs = self._to_device(next_obs_np)
        next_ah = self._to_device(next_ah_np)
        next_ah_len = self._to_device(next_ah_len_np)
        next_mask = self._to_device(next_mask_np)
        dones = self._to_device(dones_np)
        masks = self._to_device(masks_np)

        # Suit permutation augmentation — same permutation for obs and next_obs
        # to preserve temporal consistency within each DQN transition.
        br_aug_idx = np.random.randint(1, 288)
        obs = apply_card_augmentation(obs, br_aug_idx)
        next_obs = apply_card_augmentation(next_obs, br_aug_idx)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            # Current Q-values
            q_values = self.br_net(obs, ah, ah_len, masks)
            q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q-values (Double DQN)
            with torch.no_grad():
                next_q_online = self.br_net(next_obs, next_ah, next_ah_len, next_mask)
                next_actions = next_q_online.argmax(dim=-1)
                next_q_target = self.br_target(next_obs, next_ah, next_ah_len, next_mask)
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                # Terminal states have no bootstrap term; avoid (-inf * 0) -> NaN.
                next_q = torch.where(dones > 0.5, torch.zeros_like(next_q), next_q)
                target = rewards + self.config.gamma * next_q
                # Clamp targets to prevent float16 overflow under AMP
                target = target.clamp(-10000, 10000)

            loss = F.smooth_l1_loss(q_taken, target, beta=self.config.huber_delta)

        # Skip optimizer step on NaN/inf loss to avoid poisoning weights
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            self.br_updates += 1
            # Diagnose source of NaN
            nan_sources = []
            if torch.isnan(q_taken).any():
                nan_sources.append(f"q_taken({torch.isnan(q_taken).sum().item()})")
            if torch.isnan(target).any():
                nan_sources.append(f"target({torch.isnan(target).sum().item()})")
            if torch.isnan(rewards).any():
                nan_sources.append(f"rewards({torch.isnan(rewards).sum().item()})")
            if torch.isnan(obs).any():
                nan_sources.append(f"obs({torch.isnan(obs).sum().item()})")
            if torch.isnan(next_obs).any():
                nan_sources.append(f"next_obs({torch.isnan(next_obs).sum().item()})")
            if torch.isinf(q_taken).any():
                nan_sources.append(f"q_taken_inf({torch.isinf(q_taken).sum().item()})")
            src = ", ".join(nan_sources) if nan_sources else "loss computation only"
            if self.br_updates % 50 == 0 or self.br_updates <= 10:
                print(f"  [WARNING] BR loss is {loss_val} at update {self.br_updates}, "
                      f"skipping optimizer step | NaN in: {src}")
            return loss_val

        self.br_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.br_scaler.scale(loss).backward()
            self.br_scaler.unscale_(self.br_optimizer)
            self._last_br_grad_norm = torch.nn.utils.clip_grad_norm_(self.br_net.parameters(), 10.0).item()
            self.br_scaler.step(self.br_optimizer)
            self.br_scaler.update()
        else:
            loss.backward()
            self._last_br_grad_norm = torch.nn.utils.clip_grad_norm_(self.br_net.parameters(), 10.0).item()
            self.br_optimizer.step()

        self.br_updates += 1
        return loss_val

    def train_as_step(self) -> float:
        """One supervised learning step on Average Strategy network."""
        # Require 5% of buffer capacity before training to avoid noisy updates
        # after resume when buffers are nearly empty
        min_samples = max(self.config.batch_size, int(0.05 * self.config.as_buffer_size))
        if len(self.as_buffer) < min_samples:
            return 0.0

        # Sample directly as numpy arrays
        obs_np, ah_np, ah_len_np, actions_np, masks_np = self.as_buffer.sample_arrays(self.config.batch_size)

        obs = self._to_device(obs_np)
        ah = self._to_device(ah_np)
        ah_len = self._to_device(ah_len_np)
        actions = self._to_device(actions_np)
        masks = self._to_device(masks_np)

        # Suit permutation augmentation
        obs = apply_card_augmentation(obs)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            logits = self.as_net.forward_logits(obs, ah, ah_len, masks)
            loss = F.cross_entropy(logits, actions)

        # Skip optimizer step on NaN/inf loss to avoid poisoning weights
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            self.as_updates += 1
            print(f"  [WARNING] AS loss is {loss_val} at update {self.as_updates}, skipping optimizer step")
            return loss_val

        self.as_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.as_scaler.scale(loss).backward()
            self.as_scaler.unscale_(self.as_optimizer)
            self._last_as_grad_norm = torch.nn.utils.clip_grad_norm_(self.as_net.parameters(), 10.0).item()
            self.as_scaler.step(self.as_optimizer)
            self.as_scaler.update()
        else:
            loss.backward()
            self._last_as_grad_norm = torch.nn.utils.clip_grad_norm_(self.as_net.parameters(), 10.0).item()
            self.as_optimizer.step()

        self.as_updates += 1
        return loss_val

    def update_target_network(self):
        """Polyak soft update: target = tau * online + (1 - tau) * target."""
        tau = self.config.tau
        for p_target, p_online in zip(
            self._unwrap(self.br_target).parameters(),
            self._unwrap(self.br_net).parameters(),
        ):
            p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    def train(self):
        """Main training loop."""
        print(f"Starting NFSP training for {self.config.total_episodes} episodes")
        print(f"BR buffer: {self.config.br_buffer_size}, AS buffer: {self.config.as_buffer_size}")

        start_time = time.time()
        episode_count = self.total_episodes
        episodes_at_start = episode_count
        log_every = 10000
        next_log = ((episode_count // log_every) + 1) * log_every
        next_eval = ((episode_count // self.config.eval_every) + 1) * self.config.eval_every
        if self.config.checkpoint_every > 0:
            next_checkpoint = (
                ((episode_count // self.config.checkpoint_every) + 1) * self.config.checkpoint_every
            )
        else:
            next_checkpoint = self.config.total_episodes + 1
        train_rounds = episode_count // self.config.num_envs

        while episode_count < self.config.total_episodes:
            epsilon = self.get_epsilon()
            eta = self.get_eta()

            # Run self-play episodes
            steps = self.worker.run_episodes(epsilon, eta)
            self.total_steps += steps
            episode_count += self.config.num_envs
            self.total_episodes = episode_count

            # Update learning rates
            self._update_lr()

            # Train BR (DQN) — multiple gradient steps per self-play batch
            br_loss = 0.0
            for _ in range(self.config.br_train_steps):
                br_loss = self.train_br_step()
            if self.br_updates % 50 == 0:
                self.writer.add_scalar("loss/br", br_loss, self.total_steps)

            # Train AS (supervised) — multiple gradient steps
            as_loss = 0.0
            if not self.is_as_frozen():
                for _ in range(self.config.as_train_steps):
                    as_loss = self.train_as_step()
                if self.as_updates % 50 == 0:
                    self.writer.add_scalar("loss/as", as_loss, self.total_steps)

            # Soft-update target network
            train_rounds += 1
            if self.config.target_update_every > 0 and train_rounds % self.config.target_update_every == 0:
                self.update_target_network()

            # Logging
            if episode_count >= next_log:
                elapsed = time.time() - start_time
                eps_per_sec = (episode_count - episodes_at_start) / max(elapsed, 1)
                cur_lr_f = self._get_lr_factor()
                print(
                    f"Episodes: {episode_count:,} | Steps: {self.total_steps:,} | "
                    f"BR buf: {len(self.br_buffer):,} | AS buf: {len(self.as_buffer):,} | "
                    f"eps: {epsilon:.4f} | eta: {eta:.4f} | lr: {cur_lr_f:.4f} | {eps_per_sec:.0f} ep/s"
                )
                cur_br_lr = self.br_optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("meta/epsilon", epsilon, self.total_steps)
                self.writer.add_scalar("meta/eta", eta, self.total_steps)
                self.writer.add_scalar("meta/episodes", episode_count, self.total_steps)
                self.writer.add_scalar("meta/br_lr", cur_br_lr, self.total_steps)
                self.writer.add_scalar("meta/lr_factor", self._get_lr_factor(), self.total_steps)
                self.writer.add_scalar("buffer/br_size", len(self.br_buffer), self.total_steps)
                self.writer.add_scalar("buffer/as_size", len(self.as_buffer), self.total_steps)
                # Weight and gradient norms
                br_wnorm = sum(p.data.norm().item()**2 for p in self._unwrap(self.br_net).parameters())**0.5
                as_wnorm = sum(p.data.norm().item()**2 for p in self._unwrap(self.as_net).parameters())**0.5
                self.writer.add_scalar("norms/br_weight", br_wnorm, self.total_steps)
                self.writer.add_scalar("norms/as_weight", as_wnorm, self.total_steps)
                self.writer.add_scalar("norms/br_grad", self._last_br_grad_norm, self.total_steps)
                self.writer.add_scalar("norms/as_grad", self._last_as_grad_norm, self.total_steps)
                while episode_count >= next_log:
                    next_log += log_every

            # Evaluation
            while episode_count >= next_eval:
                self.evaluate(next_eval)
                next_eval += self.config.eval_every

            # Checkpointing
            while episode_count >= next_checkpoint:
                self.save_checkpoint(next_checkpoint)
                next_checkpoint += self.config.checkpoint_every

        print(f"Training complete! Total episodes: {episode_count:,}")
        self.save_checkpoint(episode_count)

    def evaluate(self, episode: int):
        """Evaluate AS network vs baselines.

        Uses a CPU copy of the model to avoid GPU contention with self-play.
        Dispatches to heads-up eval for 2 players, multiway eval for 6+.
        """
        # Create CPU copy for eval (avoids GPU contention in async mode)
        eval_model = copy.deepcopy(self._unwrap(self.as_net)).to("cpu")
        eval_model.eval()

        # Fingerprint: deterministic forward pass to detect silent model changes
        with torch.no_grad():
            rng = torch.Generator()
            rng.manual_seed(42)
            probe_obs = torch.randn(1, STATIC_FEATURE_SIZE, dtype=torch.float32, generator=rng) * 0.01
            probe_ah = torch.zeros(1, self.config.max_history_len, 11, dtype=torch.float32)
            probe_len = torch.zeros(1, dtype=torch.long)
            probe_mask = torch.ones(1, 9, dtype=torch.bool)
            probe_logits = eval_model.forward_logits(probe_obs, probe_ah, probe_len, probe_mask)
            if probe_logits.isnan().any():
                h = eval_model.history_encoder(probe_ah, probe_len)
                x = torch.cat([probe_obs, h], dim=-1)
                trunk_out = eval_model.net.trunk(x)
                print(f"  [eval probe] NaN debug: history={h.isnan().any().item()}, "
                      f"trunk={trunk_out.isnan().any().item()}, "
                      f"trunk_range=[{trunk_out.min().item():.2f}, {trunk_out.max().item():.2f}]")
            else:
                top3 = probe_logits[0].topk(3)
                print(f"  [eval probe] logits top3: {list(zip(top3.indices.tolist(), [f'{v:.2f}' for v in top3.values.tolist()]))}")

        if self.config.num_players == 2:
            self._evaluate_heads_up(eval_model, episode)
        else:
            self._evaluate_multiway(eval_model, episode)
        del eval_model

    def _evaluate_heads_up(self, eval_model, episode: int):
        """Heads-up evaluation: vs Random, Caller, TAG, and exploitability."""
        vs_random = self._eval_vs(eval_model, "random", num_hands=self.config.eval_hands)
        vs_caller = self._eval_vs(eval_model, "caller", num_hands=self.config.eval_hands)
        vs_tag = self._eval_vs(eval_model, "tag", num_hands=self.config.eval_hands)
        vs_potbet = self._eval_vs(eval_model, "potbet", num_hands=self.config.eval_hands)

        # Exploitability probe: BR vs AS on CPU
        br_cpu = copy.deepcopy(self._unwrap(self.br_net)).to("cpu")
        br_cpu.eval()
        exploit = self._eval_br_vs_as(
            num_hands=self.config.eval_hands,
            br_model=br_cpu, as_model=eval_model, eval_device=torch.device("cpu")
        )
        del br_cpu

        print(
            f"  Eval @ {episode:,}: "
            f"vs Random: {vs_random.bb100:+.2f} +/- {vs_random.ci95:.2f} bb/100 "
            f"(95% CI, n={vs_random.num_hands}) | "
            f"vs Caller: {vs_caller.bb100:+.2f} +/- {vs_caller.ci95:.2f} bb/100 "
            f"(95% CI, n={vs_caller.num_hands}) | "
            f"vs TAG: {vs_tag.bb100:+.2f} +/- {vs_tag.ci95:.2f} bb/100 "
            f"(95% CI, n={vs_tag.num_hands}) | "
            f"vs PotBet: {vs_potbet.bb100:+.2f} +/- {vs_potbet.ci95:.2f} bb/100 "
            f"(95% CI, n={vs_potbet.num_hands}) | "
            f"BR exploit: {exploit.bb100:+.2f} +/- {exploit.ci95:.2f} bb/100"
        )
        self.writer.add_scalar("winrate/vs_random_bb100", vs_random.bb100, episode)
        self.writer.add_scalar("winrate/vs_random_ci95", vs_random.ci95, episode)
        self.writer.add_scalar("winrate/vs_random_seat0_bb100", vs_random.seat0_bb100, episode)
        self.writer.add_scalar("winrate/vs_random_seat1_bb100", vs_random.seat1_bb100, episode)
        self.writer.add_scalar("winrate/vs_caller_bb100", vs_caller.bb100, episode)
        self.writer.add_scalar("winrate/vs_caller_ci95", vs_caller.ci95, episode)
        self.writer.add_scalar("winrate/vs_caller_seat0_bb100", vs_caller.seat0_bb100, episode)
        self.writer.add_scalar("winrate/vs_caller_seat1_bb100", vs_caller.seat1_bb100, episode)
        self.writer.add_scalar("winrate/vs_tag_bb100", vs_tag.bb100, episode)
        self.writer.add_scalar("winrate/vs_tag_ci95", vs_tag.ci95, episode)
        self.writer.add_scalar("winrate/vs_tag_seat0_bb100", vs_tag.seat0_bb100, episode)
        self.writer.add_scalar("winrate/vs_tag_seat1_bb100", vs_tag.seat1_bb100, episode)
        self.writer.add_scalar("winrate/vs_potbet_bb100", vs_potbet.bb100, episode)
        self.writer.add_scalar("winrate/vs_potbet_ci95", vs_potbet.ci95, episode)
        self.writer.add_scalar("winrate/vs_potbet_seat0_bb100", vs_potbet.seat0_bb100, episode)
        self.writer.add_scalar("winrate/vs_potbet_seat1_bb100", vs_potbet.seat1_bb100, episode)

        # Bluff stats (from vs TAG as most meaningful)
        self.writer.add_scalar("eval/bluff_pct", vs_tag.bluff_pct, episode)
        self.writer.add_scalar("eval/thin_value_pct", vs_tag.thin_value_pct, episode)
        self.writer.add_scalar("eval/value_bet_pct", vs_tag.value_bet_pct, episode)

        # Exploitability proxy (BR advantage over AS — lower is better)
        self.writer.add_scalar("eval/br_exploit_bb100", exploit.bb100, episode)
        self.writer.add_scalar("eval/br_exploit_ci95", exploit.ci95, episode)

        # HUD stats (vs TAG — most representative opponent)
        self.writer.add_scalar("hud/vpip", vs_tag.vpip, episode)
        self.writer.add_scalar("hud/pfr", vs_tag.pfr, episode)
        self.writer.add_scalar("hud/aggression", vs_tag.aggression, episode)
        self.writer.add_scalar("hud/wtsd", vs_tag.wtsd, episode)
        self.writer.add_scalar("hud/cbet", vs_tag.cbet, episode)
        self.writer.add_scalar("hud/showdown_pct", vs_tag.showdown_pct, episode)
        self.writer.add_scalar("hud/avg_bet_size", vs_tag.avg_bet_size, episode)

        # Per-street action distributions (vs TAG)
        self.writer.add_scalar("street/flop_fold", vs_tag.flop_fold_pct, episode)
        self.writer.add_scalar("street/flop_call", vs_tag.flop_call_pct, episode)
        self.writer.add_scalar("street/flop_raise", vs_tag.flop_raise_pct, episode)
        self.writer.add_scalar("street/turn_fold", vs_tag.turn_fold_pct, episode)
        self.writer.add_scalar("street/turn_call", vs_tag.turn_call_pct, episode)
        self.writer.add_scalar("street/turn_raise", vs_tag.turn_raise_pct, episode)
        self.writer.add_scalar("street/river_fold", vs_tag.river_fold_pct, episode)
        self.writer.add_scalar("street/river_call", vs_tag.river_call_pct, episode)
        self.writer.add_scalar("street/river_raise", vs_tag.river_raise_pct, episode)

        # Per-street bluff rates (vs TAG)
        self.writer.add_scalar("bluff/flop", vs_tag.flop_bluff_pct, episode)
        self.writer.add_scalar("bluff/turn", vs_tag.turn_bluff_pct, episode)
        self.writer.add_scalar("bluff/river", vs_tag.river_bluff_pct, episode)

        # Fold-to-bet per street (vs TAG)
        self.writer.add_scalar("fold_to_bet/preflop", vs_tag.fold_to_preflop_bet, episode)
        self.writer.add_scalar("fold_to_bet/flop", vs_tag.fold_to_flop_bet, episode)
        self.writer.add_scalar("fold_to_bet/turn", vs_tag.fold_to_turn_bet, episode)
        self.writer.add_scalar("fold_to_bet/river", vs_tag.fold_to_river_bet, episode)

        # Fold-to-bet per street (vs PotBet — most meaningful for postflop)
        self.writer.add_scalar("fold_to_bet_potbet/preflop", vs_potbet.fold_to_preflop_bet, episode)
        self.writer.add_scalar("fold_to_bet_potbet/flop", vs_potbet.fold_to_flop_bet, episode)
        self.writer.add_scalar("fold_to_bet_potbet/turn", vs_potbet.fold_to_turn_bet, episode)
        self.writer.add_scalar("fold_to_bet_potbet/river", vs_potbet.fold_to_river_bet, episode)

    def _evaluate_multiway(self, eval_model, episode: int):
        """Multiway evaluation: vs TAG table with positional and HUD stats."""
        num_players = self.config.num_players
        # Use more hands for multiway (higher variance)
        num_hands = max(self.config.eval_hands, 3000)
        pos_names = _get_position_names(num_players)

        vs_tag = self._eval_multiway_vs(eval_model, "tag", num_hands=num_hands)
        vs_random = self._eval_multiway_vs(eval_model, "random", num_hands=num_hands)
        vs_potbet = self._eval_multiway_vs(eval_model, "potbet", num_hands=num_hands)

        # Print summary
        pos_str = " | ".join(f"{n}:{vs_tag.position_bb100[n]:+.0f}" for n in pos_names)
        print(
            f"  Eval @ {episode:,} ({num_players}-max, n={num_hands}):\n"
            f"    vs TAG: {vs_tag.bb100:+.2f} +/- {vs_tag.ci95:.2f} bb/100\n"
            f"    vs Random: {vs_random.bb100:+.2f} +/- {vs_random.ci95:.2f} bb/100\n"
            f"    vs PotBet: {vs_potbet.bb100:+.2f} +/- {vs_potbet.ci95:.2f} bb/100\n"
            f"    Position bb/100: {pos_str}\n"
            f"    HUD: VPIP={vs_tag.vpip:.1f}% PFR={vs_tag.pfr:.1f}% "
            f"3Bet={vs_tag.three_bet:.1f}% Steal={vs_tag.steal_attempt:.1f}% "
            f"FoldToSteal={vs_tag.fold_to_steal:.1f}%"
        )

        # TensorBoard: overall
        self.writer.add_scalar("winrate/vs_tag_bb100", vs_tag.bb100, episode)
        self.writer.add_scalar("winrate/vs_tag_ci95", vs_tag.ci95, episode)
        self.writer.add_scalar("winrate/vs_random_bb100", vs_random.bb100, episode)
        self.writer.add_scalar("winrate/vs_random_ci95", vs_random.ci95, episode)
        self.writer.add_scalar("winrate/vs_potbet_bb100", vs_potbet.bb100, episode)
        self.writer.add_scalar("winrate/vs_potbet_ci95", vs_potbet.ci95, episode)

        # TensorBoard: positional
        for name in pos_names:
            safe_name = name.replace("+", "p")  # UTG+1 → UTGp1
            self.writer.add_scalar(f"winrate/pos_{safe_name}_bb100", vs_tag.position_bb100[name], episode)

        # TensorBoard: HUD stats
        self.writer.add_scalar("hud/vpip", vs_tag.vpip, episode)
        self.writer.add_scalar("hud/pfr", vs_tag.pfr, episode)
        self.writer.add_scalar("hud/3bet", vs_tag.three_bet, episode)
        self.writer.add_scalar("hud/steal_attempt", vs_tag.steal_attempt, episode)
        self.writer.add_scalar("hud/fold_to_steal", vs_tag.fold_to_steal, episode)

    def _eval_vs_random(self, num_hands: int = 1000) -> EvalStats:
        """Evaluate AS network vs random player (heads-up)."""
        return self._eval_heads_up(opponent="random", num_hands=num_hands)

    def _eval_vs_caller(self, num_hands: int = 1000) -> EvalStats:
        """Evaluate AS network vs calling station."""
        return self._eval_heads_up(opponent="caller", num_hands=num_hands)

    def _eval_vs_tag(self, num_hands: int = 1000) -> EvalStats:
        """Evaluate AS network vs tight-aggressive scripted baseline."""
        return self._eval_heads_up(opponent="tag", num_hands=num_hands)

    def _eval_vs(self, eval_model, opponent: str, num_hands: int) -> EvalStats:
        """Evaluate a model vs baseline on CPU."""
        return self._eval_heads_up(opponent=opponent, num_hands=num_hands,
                                   eval_model=eval_model, eval_device=torch.device("cpu"))

    def eval_exploitability_proxy(self, num_hands: int = 10000) -> EvalStats:
        """Approximate exploitability via BR-vs-AS heads-up.

        Returns BR advantage in bb/100 against the current AS policy.
        Lower is better; values near 0 suggest lower exploitability in this abstraction.
        """
        return self._eval_br_vs_as(num_hands=num_hands)

    def _select_baseline_action(self, opponent: str, obs: np.ndarray, mask: np.ndarray) -> int:
        """Select action for baseline opponent policy."""
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return 1

        if opponent == "random":
            return int(np.random.choice(legal))
        if opponent == "caller":
            return 1 if mask[1] else int(legal[0])
        if opponent == "potbet":
            # Always pot-size raise (action 6), fallback to biggest available raise, then all-in, then call
            for action in (6, 7, 5, 4, 3, 2, 8, 1):
                if mask[action]:
                    return action
            return int(legal[0])
        if opponent != "tag":
            raise ValueError(f"Unknown opponent policy: {opponent}")

        # Tight-aggressive baseline:
        # - folds weak hands to pressure,
        # - calls medium hands,
        # - raises strong hands with larger sizes when available.
        phase_oh = obs[364:370]
        phase = int(np.argmax(phase_oh))  # 0=preflop
        # Game state: 364+6(phase)+1(stack)+1(pot_bb)+1(spr)+1(pos)+1(opp)+1(can_act) = 376
        to_call_ratio = float(obs[376])  # to_call/pot normalized to [0, 1] from [0, 5]
        hand_rank = float(obs[HAND_STRENGTH_START])      # postflop normalized hand rank
        preflop_strength = float(obs[HAND_STRENGTH_START + 1])

        if phase == 0:
            strength = preflop_strength
        else:
            strength = max(hand_rank, 0.6 * preflop_strength)

        if strength >= 0.72:
            for action in (6, 5, 4, 3, 2, 7, 1):
                if mask[action]:
                    return action
        elif strength >= 0.58:
            if to_call_ratio <= 0.20:
                for action in (4, 3, 2, 1):
                    if mask[action]:
                        return action
            if mask[1]:
                return 1
            if mask[0]:
                return 0
        elif strength >= 0.48:
            if to_call_ratio <= 0.08 and mask[1]:
                return 1
            if mask[0]:
                return 0
            if mask[1]:
                return 1
        else:
            if to_call_ratio <= 0.02 and mask[1]:
                return 1
            if mask[0]:
                return 0
            if mask[1]:
                return 1

        return int(legal[0])

    def _eval_heads_up(self, opponent: str, num_hands: int,
                       eval_model=None, eval_device=None) -> EvalStats:
        """Run heads-up evaluation."""
        env = PokerEnv(
            num_players=2,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
        )

        if eval_model is None:
            eval_model = self._unwrap(self.as_net)
            eval_model.eval()
        if eval_device is None:
            eval_device = self.device

        max_hist = self.config.max_history_len
        hand_returns_bb100 = np.zeros(num_hands, dtype=np.float64)
        seat_returns_bb100: list[list[float]] = [[], []]
        action_counts = np.zeros(9, dtype=np.int64)

        # Bluff tracking: categorize raises by hand strength
        bluff_count = 0          # raise with strength < 0.3
        thin_value_count = 0     # raise with 0.3 <= strength < 0.6
        value_bet_count = 0      # raise with strength >= 0.6
        total_raises = 0
        # Per-street bluff tracking (postflop only)
        street_bluffs: dict[str, list[int]] = {"flop": [0, 0], "turn": [0, 0], "river": [0, 0]}  # [bluffs, total_raises]

        # HUD stats counters
        hands_vpip = 0       # hands where hero voluntarily put $ in
        hands_pfr = 0        # hands where hero raised preflop
        total_hero_hands = 0
        raise_actions = 0    # total raises across all streets
        passive_actions = 0  # calls + checks
        # Per-street action tracking
        street_actions: dict[str, np.ndarray] = {
            "flop": np.zeros(3, dtype=np.int64),    # [fold, call, raise]
            "turn": np.zeros(3, dtype=np.int64),
            "river": np.zeros(3, dtype=np.int64),
        }
        # Bet sizing
        bet_size_sum = 0.0
        bet_size_count = 0
        _BET_SIZES = {2: 0.25, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0, 7: 1.5}
        # Showdown tracking
        showdown_count = 0
        # Fold-to-bet per street
        fold_to_bet: dict[str, list[int]] = {
            "preflop": [0, 0], "flop": [0, 0], "turn": [0, 0], "river": [0, 0]  # [folds, opportunities]
        }
        # C-bet tracking
        cbet_opportunities = 0
        cbet_taken = 0

        for hand_idx in range(num_hands):
            hero_seat = hand_idx % 2
            player, obs, mask = env.reset()
            action_history: list[list[np.ndarray]] = [[], []]
            done = False
            total_hero_hands += 1

            # Per-hand state
            hero_vpip = False
            hero_pfr = False
            is_preflop = True
            hero_raised_preflop = False
            hero_acted_on_flop = False
            last_action_was_bet = False  # opponent bet/raised, hero faces it
            current_phase = 0
            hand_showdown = False

            while not done:
                phase = int(np.argmax(obs[364:370]))
                # Detect phase transition
                if phase > current_phase:
                    is_preflop = (phase == 0)
                    current_phase = phase
                    last_action_was_bet = False
                    hero_acted_on_flop = (phase > 1)  # past flop

                if player == hero_seat:
                    static_obs = extract_static_features_batch(obs.reshape(1, -1))[0]
                    ah_padded, ah_len = pad_action_history(action_history[player], max_hist)
                    obs_t = torch.tensor(static_obs, device=eval_device).unsqueeze(0)
                    ah_t = torch.tensor(ah_padded, device=eval_device).unsqueeze(0)
                    ah_len_t = torch.tensor([ah_len], device=eval_device)
                    mask_t = torch.tensor(mask, device=eval_device).unsqueeze(0)

                    with torch.no_grad():
                        action = eval_model.select_action(obs_t, ah_t, ah_len_t, mask_t).item()
                    action_counts[action] += 1

                    strength = float(obs[HAND_STRENGTH_START]) if phase > 0 else float(obs[HAND_STRENGTH_START + 1])

                    # HUD: VPIP/PFR (preflop only)
                    if phase == 0 and action >= 1:  # call or raise preflop
                        hero_vpip = True
                    if phase == 0 and action >= 2:
                        hero_pfr = True
                        hero_raised_preflop = True

                    # Action categorization
                    if action == 0:
                        cat = 0  # fold
                    elif action == 1:
                        cat = 1  # call/check
                        passive_actions += 1
                    else:
                        cat = 2  # raise
                        raise_actions += 1
                        passive_actions += 0  # don't count raise as passive

                    # Per-street action tracking
                    street_name = {1: "flop", 2: "turn", 3: "river"}.get(phase)
                    ftb_street = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}.get(phase)
                    if street_name:
                        street_actions[street_name][cat] += 1

                    # Bluff tracking: classify raises by hand strength
                    if action >= 2:  # any raise
                        total_raises += 1
                        if strength < 0.3:
                            bluff_count += 1
                        elif strength < 0.6:
                            thin_value_count += 1
                        else:
                            value_bet_count += 1
                        if street_name:
                            street_bluffs[street_name][1] += 1
                            if strength < 0.3:
                                street_bluffs[street_name][0] += 1
                        # Bet sizing
                        if action in _BET_SIZES:
                            bet_size_sum += _BET_SIZES[action]
                            bet_size_count += 1

                    # Fold-to-bet: did hero fold when facing a bet?
                    if last_action_was_bet and ftb_street:
                        fold_to_bet[ftb_street][1] += 1  # opportunity
                        if action == 0:
                            fold_to_bet[ftb_street][0] += 1  # folded

                    # C-bet: hero raised preflop and now acts on flop
                    if phase == 1 and hero_raised_preflop and not hero_acted_on_flop:
                        hero_acted_on_flop = True
                        cbet_opportunities += 1
                        if action >= 2:
                            cbet_taken += 1

                    last_action_was_bet = False
                else:
                    action = self._select_baseline_action(opponent, obs, mask)
                    # Track if opponent bet/raised (hero will face it next)
                    if action >= 2:
                        last_action_was_bet = True
                    else:
                        last_action_was_bet = False

                action_record = make_action_record(player, action, 2)
                for p in range(2):
                    action_history[p].append(action_record.copy())

                player, obs, mask, rewards, done = env.step(action)

                if done:
                    # rewards are already in big blinds (normalized in Rust engine)
                    hand_bb100 = float(rewards[hero_seat]) * 100.0
                    hand_returns_bb100[hand_idx] = hand_bb100
                    seat_returns_bb100[hero_seat].append(hand_bb100)
                    # Showdown detection: neither player folded (both have non-zero rewards or tie)
                    if action != 0 and (player != hero_seat or action != 0):
                        # Approximate: if last action wasn't fold, it's a showdown
                        showdown_count += 1

            if hero_vpip:
                hands_vpip += 1
            if hero_pfr:
                hands_pfr += 1

        # Log action distribution for debugging
        total_actions = action_counts.sum()
        if total_actions > 0:
            pcts = action_counts / total_actions * 100
            action_names = ["fold", "call", "0.25x", "0.4x", "0.6x", "0.8x", "1x", "1.5x", "allin"]
            dist_str = " ".join(f"{action_names[i]}:{pcts[i]:.1f}%" for i in range(9))
            print(f"    [{opponent}] actions: {dist_str}")

        # Log bluff stats
        if total_raises > 0:
            bluff_pct = bluff_count / total_raises * 100
            thin_pct = thin_value_count / total_raises * 100
            value_pct = value_bet_count / total_raises * 100
            bv_ratio = f"{bluff_count / value_bet_count:.2f}" if value_bet_count > 0 else "inf"
            street_str = " | ".join(
                f"{s}:{sb[0]}/{sb[1]}({sb[0]/sb[1]*100:.0f}%)" if sb[1] > 0 else f"{s}:0/0"
                for s, sb in street_bluffs.items()
            )
            print(f"    [{opponent}] raises: bluff={bluff_pct:.1f}% thin={thin_pct:.1f}% "
                  f"value={value_pct:.1f}% (B:V={bv_ratio}) | {street_str}")

        mean_bb100 = float(hand_returns_bb100.mean()) if num_hands > 0 else 0.0
        std_bb100 = float(hand_returns_bb100.std(ddof=1)) if num_hands > 1 else 0.0
        stderr = std_bb100 / math.sqrt(num_hands) if num_hands > 1 else 0.0
        ci95 = 1.96 * stderr

        seat0 = float(np.mean(seat_returns_bb100[0])) if seat_returns_bb100[0] else 0.0
        seat1 = float(np.mean(seat_returns_bb100[1])) if seat_returns_bb100[1] else 0.0

        b_pct = (bluff_count / total_raises * 100) if total_raises > 0 else 0.0
        t_pct = (thin_value_count / total_raises * 100) if total_raises > 0 else 0.0
        v_pct = (value_bet_count / total_raises * 100) if total_raises > 0 else 0.0

        # Compute per-street action percentages
        def _street_pcts(arr):
            s = arr.sum()
            if s == 0:
                return 0.0, 0.0, 0.0
            return float(arr[0] / s * 100), float(arr[1] / s * 100), float(arr[2] / s * 100)

        ff, fc, fr = _street_pcts(street_actions["flop"])
        tf, tc, tr = _street_pcts(street_actions["turn"])
        rf, rc, rr = _street_pcts(street_actions["river"])

        # Fold-to-bet percentages
        def _ftb_pct(arr):
            return (arr[0] / arr[1] * 100) if arr[1] > 0 else 0.0

        agg = raise_actions / max(raise_actions + passive_actions, 1) * 100

        return EvalStats(
            bb100=mean_bb100,
            ci95=ci95,
            std_bb100=std_bb100,
            num_hands=num_hands,
            seat0_bb100=seat0,
            seat1_bb100=seat1,
            bluff_pct=b_pct,
            thin_value_pct=t_pct,
            value_bet_pct=v_pct,
            vpip=hands_vpip / max(total_hero_hands, 1) * 100,
            pfr=hands_pfr / max(total_hero_hands, 1) * 100,
            aggression=agg,
            wtsd=showdown_count / max(total_hero_hands, 1) * 100,
            cbet=cbet_taken / max(cbet_opportunities, 1) * 100,
            flop_fold_pct=ff, flop_call_pct=fc, flop_raise_pct=fr,
            turn_fold_pct=tf, turn_call_pct=tc, turn_raise_pct=tr,
            river_fold_pct=rf, river_call_pct=rc, river_raise_pct=rr,
            avg_bet_size=bet_size_sum / max(bet_size_count, 1),
            flop_bluff_pct=(street_bluffs["flop"][0] / max(street_bluffs["flop"][1], 1) * 100),
            turn_bluff_pct=(street_bluffs["turn"][0] / max(street_bluffs["turn"][1], 1) * 100),
            river_bluff_pct=(street_bluffs["river"][0] / max(street_bluffs["river"][1], 1) * 100),
            showdown_pct=showdown_count / max(total_hero_hands, 1) * 100,
            fold_to_preflop_bet=_ftb_pct(fold_to_bet["preflop"]),
            fold_to_flop_bet=_ftb_pct(fold_to_bet["flop"]),
            fold_to_turn_bet=_ftb_pct(fold_to_bet["turn"]),
            fold_to_river_bet=_ftb_pct(fold_to_bet["river"]),
        )

    def _eval_multiway(self, opponent: str, num_hands: int,
                       eval_model=None, eval_device=None) -> MultiwayEvalStats:
        """Run multiway evaluation (6-max / 9-ring).

        Hero rotates through all seats. All other seats play the baseline opponent.
        Tracks positional winrate and HUD-style stats.
        """
        num_players = self.config.num_players
        env = PokerEnv(
            num_players=num_players,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
        )

        if eval_model is None:
            eval_model = self._unwrap(self.as_net)
            eval_model.eval()
        if eval_device is None:
            eval_device = self.device

        max_hist = self.config.max_history_len
        pos_names = _get_position_names(num_players)
        hand_returns_bb100 = np.zeros(num_hands, dtype=np.float64)
        pos_returns: dict[str, list[float]] = {name: [] for name in pos_names}
        action_counts = np.zeros(9, dtype=np.int64)

        # HUD stat counters
        preflop_opportunities = 0   # hands where hero acts preflop
        vpip_count = 0              # voluntarily put $ in pot
        pfr_count = 0               # preflop raise
        three_bet_opportunities = 0
        three_bet_count = 0
        steal_opportunities = 0     # hero in CO/BTN, folded to hero preflop
        steal_count = 0
        fold_to_steal_opportunities = 0  # hero in SB/BB facing steal
        fold_to_steal_count = 0

        # Steal positions: CO and BTN
        steal_pos_names = {"CO", "BTN"}
        blind_pos_names = {"SB", "BB"}

        for hand_idx in range(num_hands):
            hero_seat = hand_idx % num_players
            # Dealer advances each reset; starts at 0, first advance → 1
            dealer = (hand_idx + 1) % num_players
            hero_pos_name = _seat_to_position_name(hero_seat, dealer, num_players)

            player, obs, mask = env.reset()
            action_history: list[list[np.ndarray]] = [[] for _ in range(num_players)]
            done = False

            # Track preflop state for HUD stats
            hero_acted_preflop = False
            hero_vpip_this_hand = False
            hero_pfr_this_hand = False
            preflop_raise_count = 0  # number of raises before hero acts
            folded_to_hero_preflop = True  # track if it's folded to hero
            is_preflop = True

            while not done:
                phase_oh = obs[364:370]
                current_phase = int(np.argmax(phase_oh))
                if current_phase > 0:
                    is_preflop = False

                if player == hero_seat:
                    static_obs = extract_static_features_batch(obs.reshape(1, -1))[0]
                    ah_padded, ah_len = pad_action_history(action_history[player], max_hist)
                    obs_t = torch.tensor(static_obs, device=eval_device).unsqueeze(0)
                    ah_t = torch.tensor(ah_padded, device=eval_device).unsqueeze(0)
                    ah_len_t = torch.tensor([ah_len], device=eval_device)
                    mask_t = torch.tensor(mask, device=eval_device).unsqueeze(0)

                    with torch.no_grad():
                        action = eval_model.select_action(obs_t, ah_t, ah_len_t, mask_t).item()
                    action_counts[action] += 1

                    # HUD tracking for hero's preflop action
                    if is_preflop and not hero_acted_preflop:
                        hero_acted_preflop = True
                        preflop_opportunities += 1
                        # VPIP: any action other than fold or check (action 1 when no bet to call)
                        if action >= 2:  # any raise
                            hero_vpip_this_hand = True
                            hero_pfr_this_hand = True
                        elif action == 1 and obs[376] > 0.001:  # call with money to put in
                            hero_vpip_this_hand = True

                        # 3-bet: hero raises after a raise
                        if preflop_raise_count >= 1:
                            three_bet_opportunities += 1
                            if action >= 2:
                                three_bet_count += 1

                        # Steal: hero in CO/BTN, folded to hero
                        if hero_pos_name in steal_pos_names and folded_to_hero_preflop:
                            steal_opportunities += 1
                            if action >= 2:
                                steal_count += 1

                        # Fold to steal: hero in blinds, facing open from steal position
                        if hero_pos_name in blind_pos_names and preflop_raise_count == 1:
                            fold_to_steal_opportunities += 1
                            if action == 0:
                                fold_to_steal_count += 1
                else:
                    action = self._select_baseline_action(opponent, obs, mask)
                    # Track opponent preflop raises for 3-bet/steal detection
                    if is_preflop:
                        if action >= 2:  # opponent raised
                            preflop_raise_count += 1
                        # If opponent enters pot before hero acts, not folded to hero
                        if action >= 1 and not hero_acted_preflop:
                            folded_to_hero_preflop = False

                action_record = make_action_record(player, action, num_players)
                for p in range(num_players):
                    action_history[p].append(action_record.copy())

                player, obs, mask, rewards, done = env.step(action)

                if done:
                    hand_bb100 = float(rewards[hero_seat]) * 100.0
                    hand_returns_bb100[hand_idx] = hand_bb100
                    pos_returns[hero_pos_name].append(hand_bb100)

            if hero_vpip_this_hand:
                vpip_count += 1
            if hero_pfr_this_hand:
                pfr_count += 1

        # Compute stats
        mean_bb100 = float(hand_returns_bb100.mean()) if num_hands > 0 else 0.0
        std_bb100 = float(hand_returns_bb100.std(ddof=1)) if num_hands > 1 else 0.0
        stderr = std_bb100 / math.sqrt(num_hands) if num_hands > 1 else 0.0
        ci95 = 1.96 * stderr

        position_bb100 = {}
        for name in pos_names:
            vals = pos_returns[name]
            position_bb100[name] = float(np.mean(vals)) if vals else 0.0

        vpip_pct = (vpip_count / preflop_opportunities * 100) if preflop_opportunities > 0 else 0.0
        pfr_pct = (pfr_count / preflop_opportunities * 100) if preflop_opportunities > 0 else 0.0
        three_bet_pct = (three_bet_count / three_bet_opportunities * 100) if three_bet_opportunities > 0 else 0.0
        steal_pct = (steal_count / steal_opportunities * 100) if steal_opportunities > 0 else 0.0
        fold_to_steal_pct = (fold_to_steal_count / fold_to_steal_opportunities * 100) if fold_to_steal_opportunities > 0 else 0.0

        total_actions = action_counts.sum()
        action_names = ["fold", "call", "0.25x", "0.4x", "0.6x", "0.8x", "1x", "1.5x", "allin"]
        action_pcts = {}
        if total_actions > 0:
            for i, name in enumerate(action_names):
                action_pcts[name] = float(action_counts[i] / total_actions * 100)
            dist_str = " ".join(f"{action_names[i]}:{action_pcts[action_names[i]]:.1f}%" for i in range(9))
            print(f"    [{opponent}] actions: {dist_str}")

        return MultiwayEvalStats(
            bb100=mean_bb100,
            ci95=ci95,
            std_bb100=std_bb100,
            num_hands=num_hands,
            position_bb100=position_bb100,
            vpip=vpip_pct,
            pfr=pfr_pct,
            three_bet=three_bet_pct,
            steal_attempt=steal_pct,
            fold_to_steal=fold_to_steal_pct,
            action_pcts=action_pcts,
        )

    def _eval_multiway_vs(self, eval_model, opponent: str, num_hands: int) -> MultiwayEvalStats:
        """Evaluate a model vs baseline in multiway on CPU."""
        return self._eval_multiway(opponent=opponent, num_hands=num_hands,
                                   eval_model=eval_model, eval_device=torch.device("cpu"))

    def _eval_br_vs_as(self, num_hands: int,
                      br_model=None, as_model=None, eval_device=None) -> EvalStats:
        """Evaluate greedy BR policy against AS policy in heads-up."""
        env = PokerEnv(
            num_players=2,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
        )

        if br_model is None:
            br_model = self._unwrap(self.br_net)
        if as_model is None:
            as_model = self._unwrap(self.as_net)
        if eval_device is None:
            eval_device = self.device

        max_hist = self.config.max_history_len
        hand_returns_bb100 = np.zeros(num_hands, dtype=np.float64)
        seat_returns_bb100: list[list[float]] = [[], []]

        for hand_idx in range(num_hands):
            br_seat = hand_idx % 2
            player, obs, mask = env.reset()
            action_history: list[list[np.ndarray]] = [[], []]
            done = False

            while not done:
                static_obs = extract_static_features_batch(obs.reshape(1, -1))[0]
                ah_padded, ah_len = pad_action_history(action_history[player], max_hist)
                obs_t = torch.tensor(static_obs, device=eval_device).unsqueeze(0)
                ah_t = torch.tensor(ah_padded, device=eval_device).unsqueeze(0)
                ah_len_t = torch.tensor([ah_len], device=eval_device)
                mask_t = torch.tensor(mask, device=eval_device).unsqueeze(0)

                with torch.no_grad():
                    if player == br_seat:
                        action = br_model.select_action(
                            obs_t, ah_t, ah_len_t, mask_t, epsilon=0.0
                        ).item()
                    else:
                        action = as_model.select_action(
                            obs_t, ah_t, ah_len_t, mask_t
                        ).item()

                action_record = make_action_record(player, action, 2)
                for p in range(2):
                    action_history[p].append(action_record.copy())

                player, obs, mask, rewards, done = env.step(action)

                if done:
                    # rewards are already in big blinds (normalized in Rust engine)
                    hand_bb100 = float(rewards[br_seat]) * 100.0
                    hand_returns_bb100[hand_idx] = hand_bb100
                    seat_returns_bb100[br_seat].append(hand_bb100)

        mean_bb100 = float(hand_returns_bb100.mean()) if num_hands > 0 else 0.0
        std_bb100 = float(hand_returns_bb100.std(ddof=1)) if num_hands > 1 else 0.0
        stderr = std_bb100 / math.sqrt(num_hands) if num_hands > 1 else 0.0
        ci95 = 1.96 * stderr

        seat0 = float(np.mean(seat_returns_bb100[0])) if seat_returns_bb100[0] else 0.0
        seat1 = float(np.mean(seat_returns_bb100[1])) if seat_returns_bb100[1] else 0.0
        return EvalStats(
            bb100=mean_bb100,
            ci95=ci95,
            std_bb100=std_bb100,
            num_hands=num_hands,
            seat0_bb100=seat0,
            seat1_bb100=seat1,
        )

    def _unwrap(self, model: nn.Module) -> nn.Module:
        """Get underlying module from torch.compile wrapper."""
        return getattr(model, "_orig_mod", model)

    def save_checkpoint(self, episode: int):
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "total_steps": self.total_steps,
            "br_net": self._unwrap(self.br_net).state_dict(),
            "br_target": self._unwrap(self.br_target).state_dict(),
            "as_net": self._unwrap(self.as_net).state_dict(),
            "br_optimizer": self.br_optimizer.state_dict(),
            "as_optimizer": self.as_optimizer.state_dict(),
            "br_updates": self.br_updates,
            "as_updates": self.as_updates,
            "schedule_step_offset": self._schedule_step_offset,
        }
        if self.use_amp:
            checkpoint["br_scaler"] = self.br_scaler.state_dict()
            checkpoint["as_scaler"] = self.as_scaler.state_dict()
        torch.save(checkpoint, path / f"checkpoint_{episode}.pt")
        torch.save(checkpoint, path / "checkpoint_latest.pt")
        print(f"  Saved checkpoint at episode {episode:,}")

        # Snapshot AS weights into the pool for diverse historical opponents
        self.checkpoint_pool.snapshot(self._unwrap(self.as_net))
        if len(self.checkpoint_pool) > 1:
            print(f"  Checkpoint pool: {len(self.checkpoint_pool)} historical opponents")

        if self.config.save_buffers:
            import time as _time
            t0 = _time.time()
            br_path = path / "br_buffer_latest.npz"
            as_path = path / "as_buffer_latest.npz"
            self.br_buffer.save(str(br_path))
            self.as_buffer.save(str(as_path))
            elapsed = _time.time() - t0
            br_mb = br_path.stat().st_size / 1e6 if br_path.exists() else 0
            as_mb = as_path.stat().st_size / 1e6 if as_path.exists() else 0
            print(f"  Saved buffers: BR={br_mb:.0f}MB, AS={as_mb:.0f}MB ({elapsed:.1f}s)")

    def _migrate_dueling_state_dict(self, old_sd: dict) -> dict:
        """Migrate pre-dueling checkpoint (value_head) to dueling (value_stream + advantage_stream).

        Maps old value_head weights into the advantage_stream (same shape: residual_dim → 9).
        The value_stream (residual_dim → 1) gets fresh random init since it didn't exist before.
        """
        new_sd = {}
        migrated = False
        for key, val in old_sd.items():
            if ".value_head." in key:
                # Map value_head → advantage_stream (both output num_actions)
                new_key = key.replace(".value_head.", ".advantage_stream.")
                new_sd[new_key] = val
                migrated = True
            else:
                new_sd[key] = val

        if migrated:
            print("  Migrated old value_head → advantage_stream (value_stream freshly initialized)")

        return new_sd

    def bootstrap_as_buffer(self, target_size: int | None = None):
        """Pre-fill AS buffer by running the AS network in self-play.

        This generates (state, action) pairs that are consistent with the
        current AS strategy, preventing AS training from overwriting the
        historical average with only recent BR-policy data after resume.
        """
        if target_size is None:
            target_size = self.config.as_buffer_size

        if len(self.as_buffer) >= target_size:
            print(f"  AS buffer already has {len(self.as_buffer):,} samples, skipping bootstrap")
            return

        needed = target_size - len(self.as_buffer)
        print(f"  Bootstrapping AS buffer: generating {needed:,} samples from AS self-play...")
        t0 = time.time()

        num_envs = self.config.num_envs
        env = BatchPokerEnv(
            num_envs=num_envs,
            num_players=self.config.num_players,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
        )
        max_hist = self.config.max_history_len
        hist_dim = self.config.history_input_dim

        # Use AS network for inference — prefer compiled version (avoids ROCm SDPA issues)
        as_model = self.as_net
        if hasattr(as_model, '_orig_mod'):
            # torch.compile'd — use it directly (already handles ROCm quirks)
            pass
        else:
            as_model.eval()

        # Init envs
        results = env.reset_all()
        prev_obs = np.zeros((num_envs, self.config.input_dim), dtype=np.float32)
        prev_mask = np.zeros((num_envs, self.config.num_actions), dtype=bool)
        prev_player = np.zeros(num_envs, dtype=np.intp)
        ah_arrays = np.zeros((num_envs, self.config.num_players, max_hist, hist_dim), dtype=np.float32)
        ah_lens = np.zeros((num_envs, self.config.num_players), dtype=np.int64)
        ah_pos = np.zeros((num_envs, self.config.num_players), dtype=np.int64)

        for i in range(num_envs):
            player, obs, mask = results[i]
            prev_obs[i] = obs
            prev_mask[i] = mask
            prev_player[i] = player

        env_idx = np.arange(num_envs, dtype=np.intp)
        offsets = np.arange(max_hist, dtype=np.int64)
        static_obs = np.empty((num_envs, len(_STATIC_COLS)), dtype=np.float32)
        ah_batch = np.empty((num_envs, max_hist, hist_dim), dtype=np.float32)
        ah_lens_batch = np.empty(num_envs, dtype=np.int64)
        actions_np = np.empty(num_envs, dtype=np.int64)
        samples_added = 0

        while samples_added < needed:
            players = prev_player
            np.take(prev_obs, _STATIC_COLS, axis=1, out=static_obs)

            # Gather action histories
            ah_batch.fill(0.0)
            np.copyto(ah_lens_batch, ah_lens[env_idx, players], casting="unsafe")
            starts = (ah_pos[env_idx, players] - ah_lens_batch) % max_hist
            gather_idx = (starts[:, None] + offsets[None, :]) % max_hist
            ring = ah_arrays[env_idx, players]
            gathered = ring[np.arange(num_envs)[:, None], gather_idx]
            valid = offsets[None, :] < ah_lens_batch[:, None]
            ah_batch[valid] = gathered[valid]

            obs_t = torch.from_numpy(static_obs).to(self.device)
            ah_t = torch.from_numpy(ah_batch).to(self.device)
            ah_len_t = torch.from_numpy(ah_lens_batch).to(self.device)
            mask_t = torch.from_numpy(prev_mask).to(self.device)

            with torch.inference_mode():
                actions_gpu = as_model.select_action(obs_t, ah_t, ah_len_t, mask_t)
                actions_np[:] = actions_gpu.cpu().numpy()

            # Push ALL actions to AS buffer
            self.as_buffer.push_batch(
                obs=static_obs,
                action_history=ah_batch,
                history_length=ah_lens_batch,
                actions=actions_np,
                legal_mask=prev_mask,
            )
            samples_added += num_envs

            # Build action records and append to all players' histories
            action_records = np.zeros((num_envs, hist_dim), dtype=np.float32)
            for i in range(9):
                mask_i = actions_np == i
                if mask_i.any():
                    action_records[mask_i, 1 + i] = 1.0
                    action_records[mask_i, 10] = min(i / 8.0, 1.0)
            for p in range(self.config.num_players):
                action_records[:, 0] = players / max(1, self.config.num_players - 1)
            for p in range(self.config.num_players):
                pos = ah_pos[env_idx, p]
                ah_arrays[env_idx, p, pos] = action_records
                ah_lens[env_idx, p] = np.minimum(ah_lens[env_idx, p] + 1, max_hist)
                ah_pos[env_idx, p] = (pos + 1) % max_hist

            # Step envs
            next_players, next_obs, next_masks, _, dones = env.step_batch_dense(actions_np)

            # Handle resets
            done_idx = np.where(dones)[0]
            for d in done_idx:
                ah_arrays[d] = 0.0
                ah_lens[d] = 0
                ah_pos[d] = 0

            prev_obs[:] = next_obs
            prev_mask[:] = next_masks
            prev_player[:] = next_players

            if samples_added % (num_envs * 100) == 0:
                elapsed = time.time() - t0
                rate = samples_added / max(elapsed, 1)
                print(f"    bootstrap: {samples_added:,}/{needed:,} samples ({rate:.0f}/s)")

        elapsed = time.time() - t0
        print(f"  AS buffer bootstrapped: {len(self.as_buffer):,} samples ({elapsed:.1f}s)")

    def load_checkpoint(self, path: str, load_buffers: bool = True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Detect old checkpoint format (pre-dueling: has value_head, no value_stream)
        br_sd = checkpoint["br_net"]
        is_old_format = any(".value_head." in k for k in br_sd)

        if is_old_format:
            print("  Detected pre-dueling checkpoint, migrating BR networks...")
            br_sd = self._migrate_dueling_state_dict(br_sd)
            bt_sd = self._migrate_dueling_state_dict(checkpoint["br_target"])
            # strict=False: value_stream weights stay at their random init
            self._unwrap(self.br_net).load_state_dict(br_sd, strict=False)
            self._unwrap(self.br_target).load_state_dict(bt_sd, strict=False)
            # Reset BR optimizer since architecture changed
            self.br_optimizer = torch.optim.Adam(
                self.br_net.parameters(), lr=self.config.br_lr
            )
            print("  BR optimizer reset (new architecture parameters)")
        else:
            self._unwrap(self.br_net).load_state_dict(br_sd)
            self._unwrap(self.br_target).load_state_dict(checkpoint["br_target"])
            self.br_optimizer.load_state_dict(checkpoint["br_optimizer"])

        as_sd = checkpoint["as_net"]
        if is_old_format:
            print("  Migrating AS network...")
            as_sd = self._migrate_dueling_state_dict(as_sd)
            self._unwrap(self.as_net).load_state_dict(as_sd, strict=False)
            # Reset AS optimizer too — parameter set changed
            self.as_optimizer = torch.optim.Adam(
                self.as_net.parameters(), lr=self.config.as_lr
            )
            print("  AS optimizer reset (new architecture parameters)")
        else:
            self._unwrap(self.as_net).load_state_dict(as_sd)
            self.as_optimizer.load_state_dict(checkpoint["as_optimizer"])
        if self.use_amp:
            if "br_scaler" in checkpoint:
                self.br_scaler.load_state_dict(checkpoint["br_scaler"])
            if "as_scaler" in checkpoint:
                self.as_scaler.load_state_dict(checkpoint["as_scaler"])
        self.total_steps = checkpoint["total_steps"]
        self.total_episodes = checkpoint["episode"]
        self.br_updates = checkpoint.get("br_updates", 0)
        self.as_updates = checkpoint.get("as_updates", 0)
        self._schedule_step_offset = checkpoint.get("schedule_step_offset", 0)
        print(f"Loaded checkpoint from episode {checkpoint['episode']:,}")

        # Try to load saved buffers from the same checkpoint directory
        if load_buffers:
            ckpt_dir = Path(path).parent
            br_buf_path = ckpt_dir / "br_buffer_latest.npz"
            as_buf_path = ckpt_dir / "as_buffer_latest.npz"
            if br_buf_path.exists() and as_buf_path.exists():
                import time as _time
                t0 = _time.time()
                self.br_buffer.load(str(br_buf_path))
                self.as_buffer.load(str(as_buf_path))
                elapsed = _time.time() - t0
                print(f"  Loaded buffers: BR={len(self.br_buffer):,}, AS={len(self.as_buffer):,} ({elapsed:.1f}s)")
        else:
            print("  Skipping buffer load (--no-load-buffers)")

        # Set AS freeze duration if configured
        if self.config.as_freeze_duration > 0:
            self._as_unfreeze_episode = self.total_episodes + self.config.as_freeze_duration
            self._as_optimizer_reset_pending = True
            print(f"  AS frozen until episode {self._as_unfreeze_episode:,} "
                  f"({self.config.as_freeze_duration:,} episodes from now)")
