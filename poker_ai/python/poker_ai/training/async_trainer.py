"""Async NFSP trainer with concurrent self-play and training.

Uses a double-buffered GPU approach: inference network copies for self-play
run on a background thread while the main thread trains on the training networks.
Weights are periodically synced from training → inference copies.
"""

import copy
import math
import threading
import time

import numpy as np
import torch
import torch.nn as nn

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.training.nfsp import NFSPTrainer
from poker_ai.training.self_play import SelfPlayWorker
from poker_ai.model.network import BestResponseNet, AverageStrategyNet


class AsyncNFSPTrainer(NFSPTrainer):
    """NFSP trainer with concurrent self-play and training threads.

    Inherits all training step logic, evaluation, checkpointing, and LR scheduling
    from NFSPTrainer. Overrides train() to run self-play on a background thread
    while the main thread runs gradient updates continuously.
    """

    def __init__(self, config: NFSPConfig):
        super().__init__(config)

        # Create inference copies of networks (read-only, used by self-play thread)
        self.br_inference = copy.deepcopy(self._unwrap(self.br_net)).to(self.device)
        self.as_inference = copy.deepcopy(self._unwrap(self.as_net)).to(self.device)
        self.br_inference.eval()
        self.as_inference.eval()

        # torch.compile the inference copies too
        if self.device.type == "cuda":
            try:
                self.br_inference = torch.compile(self.br_inference)
                self.as_inference = torch.compile(self.as_inference)
                print("torch.compile enabled for inference copies")
            except Exception as e:
                print(f"torch.compile not available for inference copies: {e}")

        # Re-create worker with inference copies
        self.worker = SelfPlayWorker(
            config, self.br_net, self.as_net,
            self.br_buffer, self.as_buffer, self.device,
            br_inference=self.br_inference,
            as_inference=self.as_inference,
        )

        # Threading primitives
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = paused
        self._step_lock = threading.Lock()
        self._self_play_thread = None
        # Track self-play rounds to pace training
        self._self_play_rounds = 0

        inf_params = sum(p.numel() for p in self._unwrap(self.br_inference).parameters())
        inf_params += sum(p.numel() for p in self._unwrap(self.as_inference).parameters())
        print(f"Async mode: inference copy VRAM ~{inf_params * 4 / 1e6:.1f} MB")

    def _sync_inference_weights(self):
        """Copy training network weights to inference copies."""
        with torch.no_grad():
            for p_inf, p_train in zip(
                self._unwrap(self.br_inference).parameters(),
                self._unwrap(self.br_net).parameters(),
            ):
                p_inf.data.copy_(p_train.data)
            for p_inf, p_train in zip(
                self._unwrap(self.as_inference).parameters(),
                self._unwrap(self.as_net).parameters(),
            ):
                p_inf.data.copy_(p_train.data)

    def _self_play_loop(self):
        """Background thread: run self-play episodes continuously."""
        torch.set_float32_matmul_precision("high")
        while not self._stop_event.is_set():
            # Check if paused (e.g. during evaluation)
            if self._pause_event.is_set():
                time.sleep(0.01)
                continue

            epsilon = self.get_epsilon()
            eta = self.get_eta()
            steps = self.worker.run_episodes(epsilon, eta)

            with self._step_lock:
                self.total_steps += steps
                self.total_episodes += self.config.num_envs
                self._self_play_rounds += 1

    def _do_logging_eval_checkpoint(
        self, episode_count, total_steps, start_time, episodes_at_start,
        next_log, next_eval, next_checkpoint,
    ):
        """Handle logging, evaluation, and checkpointing. Returns updated next_* values."""
        # Logging
        if episode_count >= next_log:
            elapsed = time.time() - start_time
            eps_per_sec = (episode_count - episodes_at_start) / max(elapsed, 1)
            epsilon = self.get_epsilon()
            eta = self.get_eta()
            cur_lr_f = self._get_lr_factor()
            print(
                f"Episodes: {episode_count:,} | Steps: {total_steps:,} | "
                f"BR buf: {len(self.br_buffer):,} | AS buf: {len(self.as_buffer):,} | "
                f"eps: {epsilon:.4f} | eta: {eta:.4f} | lr: {cur_lr_f:.4f} | "
                f"{eps_per_sec:.0f} ep/s [async]"
            )
            cur_br_lr = self.br_optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("meta/epsilon", epsilon, total_steps)
            self.writer.add_scalar("meta/eta", eta, total_steps)
            self.writer.add_scalar("meta/episodes", episode_count, total_steps)
            self.writer.add_scalar("meta/br_lr", cur_br_lr, total_steps)
            self.writer.add_scalar("meta/lr_factor", cur_lr_f, total_steps)
            self.writer.add_scalar("buffer/br_size", len(self.br_buffer), total_steps)
            self.writer.add_scalar("buffer/as_size", len(self.as_buffer), total_steps)
            while episode_count >= next_log:
                next_log += 10000

        # Evaluation (pause self-play to avoid GPU contention)
        while episode_count >= next_eval:
            self._pause_event.set()
            time.sleep(0.05)
            self.evaluate(next_eval)
            self._pause_event.clear()
            next_eval += self.config.eval_every

        # Checkpointing
        while episode_count >= next_checkpoint:
            self.save_checkpoint(next_checkpoint)
            next_checkpoint += self.config.checkpoint_every

        return next_log, next_eval, next_checkpoint

    def train(self):
        """Main training loop with concurrent self-play."""
        print(f"Starting ASYNC NFSP training for {self.config.total_episodes} episodes")
        print(f"BR buffer: {self.config.br_buffer_size}, AS buffer: {self.config.as_buffer_size}")
        print(f"Sync every: {self.config.sync_every} training rounds, "
              f"train ahead: {self.config.train_ahead} rounds")

        start_time = time.time()
        episodes_at_start = self.total_episodes
        next_log = ((self.total_episodes // 10000) + 1) * 10000
        next_eval = ((self.total_episodes // self.config.eval_every) + 1) * self.config.eval_every
        if self.config.checkpoint_every > 0:
            next_checkpoint = (
                ((self.total_episodes // self.config.checkpoint_every) + 1) * self.config.checkpoint_every
            )
        else:
            next_checkpoint = self.config.total_episodes + 1
        train_rounds = self.total_episodes // self.config.num_envs
        # Align self-play round counter with checkpoint so pacing works on resume
        self._self_play_rounds = train_rounds

        # Initial sync so inference copies start with current weights
        self._sync_inference_weights()

        # Start self-play thread
        self._self_play_thread = threading.Thread(
            target=self._self_play_loop, daemon=True, name="self-play"
        )
        self._self_play_thread.start()
        print("Self-play thread started")

        # Minimum buffer fill before training begins
        min_br_samples = max(self.config.batch_size, int(0.05 * self.config.br_buffer_size))
        min_as_samples = max(self.config.batch_size, int(0.05 * self.config.as_buffer_size))
        warmup_done = False

        try:
            while True:
                # Read episode count and self-play progress
                with self._step_lock:
                    episode_count = self.total_episodes
                    total_steps = self.total_steps
                    sp_rounds = self._self_play_rounds

                if episode_count >= self.config.total_episodes:
                    break

                # Always handle logging/eval/checkpoint regardless of pacing
                next_log, next_eval, next_checkpoint = self._do_logging_eval_checkpoint(
                    episode_count, total_steps, start_time, episodes_at_start,
                    next_log, next_eval, next_checkpoint,
                )

                # Wait for buffer warmup
                if not warmup_done:
                    if len(self.br_buffer) >= min_br_samples and len(self.as_buffer) >= min_as_samples:
                        warmup_done = True
                        # Snap train_rounds to current sp_rounds so training starts
                        # in sync and only gradually builds its train_ahead budget.
                        # Without this, the rounds accumulated during warmup would
                        # let training burst through 100+ rounds on a tiny buffer.
                        with self._step_lock:
                            sp_rounds = self._self_play_rounds
                        train_rounds = sp_rounds
                        print(f"Buffer warmup complete (BR: {len(self.br_buffer):,}, "
                              f"AS: {len(self.as_buffer):,}), training started")
                    else:
                        time.sleep(0.1)
                        continue

                # Pace training: allow training to run ahead by train_ahead rounds.
                # This lets training keep the GPU busy while self-play generates data,
                # without runaway oversampling that destroys the network.
                if train_rounds >= sp_rounds + self.config.train_ahead:
                    time.sleep(0.001)
                    continue

                # Update learning rates
                self._update_lr()

                # Train BR (DQN)
                br_loss = 0.0
                for _ in range(self.config.br_train_steps):
                    br_loss = self.train_br_step()
                if br_loss > 0 and self.br_updates % 50 == 0:
                    self.writer.add_scalar("loss/br", br_loss, total_steps)

                # Train AS (supervised)
                as_loss = 0.0
                for _ in range(self.config.as_train_steps):
                    as_loss = self.train_as_step()
                if as_loss > 0 and self.as_updates % 50 == 0:
                    self.writer.add_scalar("loss/as", as_loss, total_steps)

                # Soft-update target network
                train_rounds += 1
                if self.config.target_update_every > 0 and train_rounds % self.config.target_update_every == 0:
                    self.update_target_network()

                # Periodic sync to inference copies
                if train_rounds % self.config.sync_every == 0:
                    self._sync_inference_weights()

        finally:
            # Graceful shutdown
            self._stop_event.set()
            if self._self_play_thread is not None:
                self._self_play_thread.join(timeout=5.0)
            print(f"Training complete! Total episodes: {self.total_episodes:,}")

        self.save_checkpoint(self.total_episodes)
