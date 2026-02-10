"""NFSP (Neural Fictitious Self-Play) training loop."""

import copy
import os
import time
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


class NFSPTrainer:
    """NFSP training manager."""

    def __init__(self, config: NFSPConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.use_cuda_transfer = self.device.type == "cuda"
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        )
        torch.set_float32_matmul_precision("high")
        print(f"Using device: {self.device}")
        if self.use_amp:
            print(f"AMP enabled: dtype={self.amp_dtype}")
        print(f"Model: hidden={config.hidden_dim}, residual={config.residual_dim}, "
              f"lstm_hidden={config.lstm_hidden_dim}, batch={config.batch_size}")

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
        self.br_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.as_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Replay buffers
        self.br_buffer = CircularBuffer(config.br_buffer_size, max_seq_len=config.max_history_len)
        self.as_buffer = ReservoirBuffer(config.as_buffer_size, max_seq_len=config.max_history_len)

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
        """Copy BR network weights to target network."""
        self._unwrap(self.br_target).load_state_dict(
            self._unwrap(self.br_net).state_dict()
        )

    def train(self):
        """Main training loop."""
        print(f"Starting NFSP training for {self.config.total_episodes} episodes")
        print(f"BR buffer: {self.config.br_buffer_size}, AS buffer: {self.config.as_buffer_size}")

        start_time = time.time()
        episode_count = self.total_episodes
        log_every = 10000
        next_log = ((episode_count // log_every) + 1) * log_every
        next_eval = ((episode_count // self.config.eval_every) + 1) * self.config.eval_every
        next_checkpoint = (
            ((episode_count // self.config.checkpoint_every) + 1) * self.config.checkpoint_every
        )
        train_rounds = episode_count // self.config.num_envs

        while episode_count < self.config.total_episodes:
            epsilon = self.get_epsilon()

            # Run self-play episodes
            steps = self.worker.run_episodes(epsilon)
            self.total_steps += steps
            episode_count += self.config.num_envs
            self.total_episodes = episode_count

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

            # Update target network
            train_rounds += 1
            if train_rounds % self.config.target_update_every == 0:
                self.update_target_network()

            # Logging
            if episode_count >= next_log:
                elapsed = time.time() - start_time
                eps_per_sec = episode_count / max(elapsed, 1)
                print(
                    f"Episodes: {episode_count:,} | Steps: {self.total_steps:,} | "
                    f"BR buf: {len(self.br_buffer):,} | AS buf: {len(self.as_buffer):,} | "
                    f"eps: {epsilon:.4f} | {eps_per_sec:.0f} ep/s"
                )
                self.writer.add_scalar("meta/epsilon", epsilon, self.total_steps)
                self.writer.add_scalar("meta/episodes", episode_count, self.total_steps)
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
        wins_vs_random = self._eval_vs_random(num_hands=1000)
        wins_vs_caller = self._eval_vs_caller(num_hands=1000)

        print(
            f"  Eval @ {episode:,}: "
            f"vs Random: {wins_vs_random:+.2f} bb/100 | "
            f"vs Caller: {wins_vs_caller:+.2f} bb/100"
        )
        self.writer.add_scalar("eval/vs_random_bb100", wins_vs_random, episode)
        self.writer.add_scalar("eval/vs_caller_bb100", wins_vs_caller, episode)

    def _eval_vs_random(self, num_hands: int = 1000) -> float:
        """Evaluate AS network vs random player (heads-up). Returns bb/100."""
        return self._eval_heads_up(opponent="random", num_hands=num_hands)

    def _eval_vs_caller(self, num_hands: int = 1000) -> float:
        """Evaluate AS network vs calling station. Returns bb/100."""
        return self._eval_heads_up(opponent="caller", num_hands=num_hands)

    def _eval_heads_up(self, opponent: str, num_hands: int) -> float:
        """Run heads-up evaluation."""
        env = PokerEnv(
            num_players=2,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
        )

        max_hist = self.config.max_history_len
        total_reward = 0.0
        hero_seat = 0

        for _ in range(num_hands):
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
                    if opponent == "random":
                        legal = np.where(mask)[0]
                        action = np.random.choice(legal) if len(legal) > 0 else 1
                    else:
                        action = 1  # caller

                action_record = make_action_record(player, action, 2)
                for p in range(2):
                    action_history[p].append(action_record.copy())

                player, obs, mask, rewards, done = env.step(action)

                if done:
                    total_reward += rewards[hero_seat]

        return (total_reward / num_hands) * 100

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
