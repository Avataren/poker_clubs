"""NFSP (Neural Fictitious Self-Play) training loop."""

import copy
import math
import os
import time
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
from poker_ai.training.self_play import SelfPlayWorker, extract_static_features_batch, pad_action_history, make_action_record
from poker_ai.env.poker_env import PokerEnv
from poker_ai.model.state_encoder import HAND_STRENGTH_START


@dataclass
class EvalStats:
    """Summary statistics for heads-up evaluation."""
    bb100: float
    ci95: float
    std_bb100: float
    num_hands: int
    seat0_bb100: float
    seat1_bb100: float


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
        scaler_device = "cuda" if self.use_amp else "cpu"
        self.br_scaler = torch.amp.GradScaler(scaler_device, enabled=self.use_amp)
        self.as_scaler = torch.amp.GradScaler(scaler_device, enabled=self.use_amp)

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
        self.worker = SelfPlayWorker(
            config, self.br_net, self.as_net, self.br_buffer, self.as_buffer, self.device
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

    def get_epsilon(self) -> float:
        """Get current epsilon for exploration."""
        progress = min(self.total_steps / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def get_eta(self) -> float:
        """Get current eta (AS/BR mix) with linear ramp."""
        progress = min(self.total_steps / max(self.config.eta_ramp_steps, 1), 1.0)
        return self.config.eta_start + progress * (
            self.config.eta_end - self.config.eta_start
        )

    def _get_lr_factor(self) -> float:
        """Cosine decay with linear warmup. Returns multiplier in [min_factor, 1.0]."""
        steps = self.total_steps
        warmup = self._lr_warmup_steps
        if steps < warmup:
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
        for pg in self.as_optimizer.param_groups:
            pg["lr"] = self.config.as_lr * factor

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        """Move numpy array to training device with optional pinned-memory staging."""
        tensor = torch.from_numpy(array)
        if self.use_cuda_transfer:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.use_cuda_transfer)

    def train_br_step(self) -> float:
        """One DQN training step on Best Response network."""
        if len(self.br_buffer) < self.config.batch_size:
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

            loss = F.smooth_l1_loss(q_taken, target)

        self.br_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.br_scaler.scale(loss).backward()
            self.br_scaler.unscale_(self.br_optimizer)
            torch.nn.utils.clip_grad_norm_(self.br_net.parameters(), 10.0)
            self.br_scaler.step(self.br_optimizer)
            self.br_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.br_net.parameters(), 10.0)
            self.br_optimizer.step()

        self.br_updates += 1
        return loss.item()

    def train_as_step(self) -> float:
        """One supervised learning step on Average Strategy network."""
        if len(self.as_buffer) < self.config.batch_size:
            return 0.0

        # Sample directly as numpy arrays
        obs_np, ah_np, ah_len_np, actions_np, masks_np = self.as_buffer.sample_arrays(self.config.batch_size)

        obs = self._to_device(obs_np)
        ah = self._to_device(ah_np)
        ah_len = self._to_device(ah_len_np)
        actions = self._to_device(actions_np)
        masks = self._to_device(masks_np)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            logits = self.as_net.forward_logits(obs, ah, ah_len, masks)
            loss = F.cross_entropy(logits, actions)

        self.as_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.as_scaler.scale(loss).backward()
            self.as_scaler.unscale_(self.as_optimizer)
            torch.nn.utils.clip_grad_norm_(self.as_net.parameters(), 10.0)
            self.as_scaler.step(self.as_optimizer)
            self.as_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.as_net.parameters(), 10.0)
            self.as_optimizer.step()

        self.as_updates += 1
        return loss.item()

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
            for _ in range(self.config.br_train_steps):
                br_loss = self.train_br_step()
            if br_loss > 0 and self.br_updates % 50 == 0:
                self.writer.add_scalar("loss/br", br_loss, self.total_steps)

            # Train AS (supervised) — multiple gradient steps
            for _ in range(self.config.as_train_steps):
                as_loss = self.train_as_step()
            if as_loss > 0 and self.as_updates % 50 == 0:
                self.writer.add_scalar("loss/as", as_loss, self.total_steps)

            # Soft-update target network every round
            train_rounds += 1
            self.update_target_network()

            # Logging
            if episode_count >= next_log:
                elapsed = time.time() - start_time
                eps_per_sec = episode_count / max(elapsed, 1)
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
        """Evaluate AS network vs baselines."""
        vs_random = self._eval_vs_random(num_hands=self.config.eval_hands)
        vs_caller = self._eval_vs_caller(num_hands=self.config.eval_hands)
        vs_tag = self._eval_vs_tag(num_hands=self.config.eval_hands)

        print(
            f"  Eval @ {episode:,}: "
            f"vs Random: {vs_random.bb100:+.2f} +/- {vs_random.ci95:.2f} bb/100 "
            f"(95% CI, n={vs_random.num_hands}) | "
            f"vs Caller: {vs_caller.bb100:+.2f} +/- {vs_caller.ci95:.2f} bb/100 "
            f"(95% CI, n={vs_caller.num_hands}) | "
            f"vs TAG: {vs_tag.bb100:+.2f} +/- {vs_tag.ci95:.2f} bb/100 "
            f"(95% CI, n={vs_tag.num_hands})"
        )
        self.writer.add_scalar("eval/vs_random_bb100", vs_random.bb100, episode)
        self.writer.add_scalar("eval/vs_random_ci95", vs_random.ci95, episode)
        self.writer.add_scalar("eval/vs_random_seat0_bb100", vs_random.seat0_bb100, episode)
        self.writer.add_scalar("eval/vs_random_seat1_bb100", vs_random.seat1_bb100, episode)
        self.writer.add_scalar("eval/vs_caller_bb100", vs_caller.bb100, episode)
        self.writer.add_scalar("eval/vs_caller_ci95", vs_caller.ci95, episode)
        self.writer.add_scalar("eval/vs_caller_seat0_bb100", vs_caller.seat0_bb100, episode)
        self.writer.add_scalar("eval/vs_caller_seat1_bb100", vs_caller.seat1_bb100, episode)
        self.writer.add_scalar("eval/vs_tag_bb100", vs_tag.bb100, episode)
        self.writer.add_scalar("eval/vs_tag_ci95", vs_tag.ci95, episode)
        self.writer.add_scalar("eval/vs_tag_seat0_bb100", vs_tag.seat0_bb100, episode)
        self.writer.add_scalar("eval/vs_tag_seat1_bb100", vs_tag.seat1_bb100, episode)

    def _eval_vs_random(self, num_hands: int = 1000) -> EvalStats:
        """Evaluate AS network vs random player (heads-up)."""
        return self._eval_heads_up(opponent="random", num_hands=num_hands)

    def _eval_vs_caller(self, num_hands: int = 1000) -> EvalStats:
        """Evaluate AS network vs calling station."""
        return self._eval_heads_up(opponent="caller", num_hands=num_hands)

    def _eval_vs_tag(self, num_hands: int = 1000) -> EvalStats:
        """Evaluate AS network vs tight-aggressive scripted baseline."""
        return self._eval_heads_up(opponent="tag", num_hands=num_hands)

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

    def _eval_heads_up(self, opponent: str, num_hands: int) -> EvalStats:
        """Run heads-up evaluation."""
        env = PokerEnv(
            num_players=2,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
        )

        max_hist = self.config.max_history_len
        hand_returns_bb100 = np.zeros(num_hands, dtype=np.float64)
        seat_returns_bb100: list[list[float]] = [[], []]

        for hand_idx in range(num_hands):
            hero_seat = hand_idx % 2  # balance positional bias across buttons/blinds
            player, obs, mask = env.reset()
            action_history: list[list[np.ndarray]] = [[], []]
            done = False

            while not done:
                if player == hero_seat:
                    static_obs = extract_static_features_batch(obs.reshape(1, -1))[0]
                    ah_padded, ah_len = pad_action_history(action_history[player], max_hist)
                    obs_t = torch.tensor(static_obs, device=self.device).unsqueeze(0)
                    ah_t = torch.tensor(ah_padded, device=self.device).unsqueeze(0)
                    ah_len_t = torch.tensor([ah_len], device=self.device)
                    mask_t = torch.tensor(mask, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        action = self.as_net.select_action(obs_t, ah_t, ah_len_t, mask_t).item()
                else:
                    action = self._select_baseline_action(opponent, obs, mask)

                action_record = make_action_record(player, action, 2)
                for p in range(2):
                    action_history[p].append(action_record.copy())

                player, obs, mask, rewards, done = env.step(action)

                if done:
                    # rewards are already in big blinds (normalized in Rust engine)
                    hand_bb100 = float(rewards[hero_seat]) * 100.0
                    hand_returns_bb100[hand_idx] = hand_bb100
                    seat_returns_bb100[hero_seat].append(hand_bb100)

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

    def _eval_br_vs_as(self, num_hands: int) -> EvalStats:
        """Evaluate greedy BR policy against AS policy in heads-up."""
        env = PokerEnv(
            num_players=2,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
        )

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
                obs_t = torch.tensor(static_obs, device=self.device).unsqueeze(0)
                ah_t = torch.tensor(ah_padded, device=self.device).unsqueeze(0)
                ah_len_t = torch.tensor([ah_len], device=self.device)
                mask_t = torch.tensor(mask, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    if player == br_seat:
                        action = self.br_net.select_action(
                            obs_t, ah_t, ah_len_t, mask_t, epsilon=0.0
                        ).item()
                    else:
                        action = self.as_net.select_action(
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
        }
        if self.use_amp:
            checkpoint["br_scaler"] = self.br_scaler.state_dict()
            checkpoint["as_scaler"] = self.as_scaler.state_dict()
        torch.save(checkpoint, path / f"checkpoint_{episode}.pt")
        torch.save(checkpoint, path / "checkpoint_latest.pt")
        print(f"  Saved checkpoint at episode {episode:,}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self._unwrap(self.br_net).load_state_dict(checkpoint["br_net"])
        self._unwrap(self.br_target).load_state_dict(checkpoint["br_target"])
        self._unwrap(self.as_net).load_state_dict(checkpoint["as_net"])
        self.br_optimizer.load_state_dict(checkpoint["br_optimizer"])
        self.as_optimizer.load_state_dict(checkpoint["as_optimizer"])
        if self.use_amp:
            if "br_scaler" in checkpoint:
                self.br_scaler.load_state_dict(checkpoint["br_scaler"])
            if "as_scaler" in checkpoint:
                self.as_scaler.load_state_dict(checkpoint["as_scaler"])
        self.total_steps = checkpoint["total_steps"]
        self.total_episodes = checkpoint["episode"]
        print(f"Loaded checkpoint from episode {checkpoint['episode']:,}")
