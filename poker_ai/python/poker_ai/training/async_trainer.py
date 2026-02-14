"""Async NFSP trainer with concurrent self-play and training.

Uses a double-buffered GPU approach: inference network copies for self-play
run on a background thread while the main thread trains on the training networks.
Weights are periodically synced from training → inference copies.

IMPORTANT: Weight sync must pause self-play to avoid a CUDA data race.
Different threads use different CUDA streams, so an in-place weight copy
on one stream can race with a forward pass on another stream reading the
same parameter tensors. We use a pause/acknowledge handshake to ensure
self-play has finished its current run_episodes() before syncing.
"""

import copy
import math
import threading
import traceback
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

        # Re-create worker with inference copies and pause callback
        self.worker = SelfPlayWorker(
            config, self.br_net, self.as_net,
            self.br_buffer, self.as_buffer, self.device,
            br_inference=self.br_inference,
            as_inference=self.as_inference,
            pause_check=self._handle_pause_in_worker,
        )

        # Threading primitives
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # main thread sets to request pause
        self._paused_ack = threading.Event()   # self-play sets to acknowledge pause
        self._step_lock = threading.Lock()
        self._self_play_thread = None
        self._self_play_error = None  # store exception from self-play thread
        # Track self-play rounds to pace training
        self._self_play_rounds = 0
        self._as_unfroze_logged = False

        inf_params = sum(p.numel() for p in self._unwrap(self.br_inference).parameters())
        inf_params += sum(p.numel() for p in self._unwrap(self.as_inference).parameters())
        print(f"Async mode: inference copy VRAM ~{inf_params * 4 / 1e6:.1f} MB")

    def _handle_pause_in_worker(self):
        """Called from inside run_episodes() to check for pause requests."""
        if self._pause_event.is_set():
            self._paused_ack.set()
            while self._pause_event.is_set() and not self._stop_event.is_set():
                time.sleep(0.001)

    def _pause_self_play(self):
        """Request self-play pause and wait for acknowledgment."""
        # If self-play thread has crashed, don't wait
        if self._self_play_error is not None:
            raise RuntimeError(
                f"Self-play thread crashed: {self._self_play_error}"
            ) from self._self_play_error
        self._pause_event.set()
        if not self._paused_ack.wait(timeout=30.0):
            # Check if thread crashed while we were waiting
            if self._self_play_error is not None:
                raise RuntimeError(
                    f"Self-play thread crashed: {self._self_play_error}"
                ) from self._self_play_error
            print("WARNING: self-play pause timeout — waiting indefinitely")
            self._paused_ack.wait()

    def _resume_self_play(self):
        """Resume self-play after pause."""
        self._paused_ack.clear()
        self._pause_event.clear()

    def _sync_inference_weights(self):
        """Copy training network weights to inference copies.

        Pauses self-play to avoid CUDA data race between the weight copy
        (main thread's stream) and forward passes (self-play thread's stream)
        on the same parameter tensors.
        """
        self._pause_self_play()
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
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._resume_self_play()

    def _self_play_loop(self):
        """Background thread: run self-play episodes continuously."""
        try:
            torch.set_float32_matmul_precision("high")
            while not self._stop_event.is_set():
                # Check if paused (e.g. during weight sync or evaluation)
                if self._pause_event.is_set():
                    self._paused_ack.set()  # acknowledge we're paused
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.001)
                    continue

                epsilon = self.get_epsilon()
                eta = self.get_eta()
                steps = self.worker.run_episodes(epsilon, eta)

                with self._step_lock:
                    self.total_steps += steps
                    self.total_episodes += self.config.num_envs
                    self._self_play_rounds += 1
        except Exception as e:
            self._self_play_error = e
            print(f"\nFATAL: Self-play thread crashed:\n{traceback.format_exc()}")
            # Unblock any pending pause
            self._paused_ack.set()

    def _check_self_play_alive(self):
        """Raise if self-play thread has crashed."""
        if self._self_play_error is not None:
            raise RuntimeError(
                f"Self-play thread crashed: {self._self_play_error}"
            ) from self._self_play_error

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
            actual_lr = self.br_optimizer.param_groups[0]["lr"]
            print(
                f"Episodes: {episode_count:,} | Steps: {total_steps:,} | "
                f"BR buf: {len(self.br_buffer):,} | AS buf: {len(self.as_buffer):,} | "
                f"eps: {epsilon:.4f} | eta: {eta:.4f} | lr: {cur_lr_f:.4f} (actual: {actual_lr:.2e}) | "
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
            self._pause_self_play()
            self.evaluate(next_eval)
            self._resume_self_play()
            next_eval += self.config.eval_every

        # Checkpointing (pause self-play if saving buffers to avoid lock contention)
        while episode_count >= next_checkpoint:
            if self.config.save_buffers:
                self._pause_self_play()
            self.save_checkpoint(next_checkpoint)
            if self.config.save_buffers:
                self._resume_self_play()
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
        # (no need to pause — self-play thread hasn't started yet)
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

        # Start self-play thread
        self._self_play_thread = threading.Thread(
            target=self._self_play_loop, daemon=True, name="self-play"
        )
        self._self_play_thread.start()
        print("Self-play thread started")

        # Minimum buffer fill before training begins.
        # Use at least 50% of buffer capacity to ensure diversity, and at least
        # 3x per-round samples so each sample is drawn <0.33 times per round.
        br_per_round = self.config.br_train_steps * self.config.batch_size
        as_per_round = self.config.as_train_steps * self.config.batch_size
        min_br_samples = max(self.config.br_buffer_size // 2, 3 * br_per_round)
        min_as_samples = max(self.config.as_buffer_size // 4, 3 * as_per_round)
        warmup_done = False

        try:
            while True:
                self._check_self_play_alive()

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

                # Train AS (supervised) — skip if frozen (resume without historical buffer)
                as_loss = 0.0
                if not self.is_as_frozen():
                    if not self._as_unfroze_logged and self._as_unfreeze_episode > 0:
                        # Log AS weight norm before any training
                        as_wnorm = sum(p.data.norm().item()**2 for p in self._unwrap(self.as_net).parameters())**0.5
                        as_lr = self.as_optimizer.param_groups[0]["lr"]
                        print(f"AS network UNFROZEN at episode {episode_count:,} — "
                              f"buffer has {len(self.as_buffer):,} samples from diverse BR policies")
                        print(f"  [AS debug] weight_norm={as_wnorm:.4f}, lr={as_lr:.2e}, "
                              f"as_updates={self.as_updates}")
                        self._as_unfroze_logged = True
                    for _ in range(self.config.as_train_steps):
                        as_loss = self.train_as_step()
                    if as_loss > 0 and self.as_updates % 50 == 0:
                        self.writer.add_scalar("loss/as", as_loss, total_steps)
                    # Debug: log AS LR, loss, and weight norm after unfreeze
                    if self.as_updates > 0 and self.as_updates <= 20:
                        as_lr = self.as_optimizer.param_groups[0]["lr"]
                        as_wnorm = sum(p.data.norm().item()**2 for p in self._unwrap(self.as_net).parameters())**0.5
                        print(f"  [AS debug] update #{self.as_updates}: loss={as_loss:.4f}, "
                              f"lr={as_lr:.2e}, weight_norm={as_wnorm:.4f}")
                    elif self.as_updates > 0 and self.as_updates % 500 == 0:
                        as_lr = self.as_optimizer.param_groups[0]["lr"]
                        as_wnorm = sum(p.data.norm().item()**2 for p in self._unwrap(self.as_net).parameters())**0.5
                        print(f"  [AS debug] update #{self.as_updates}: loss={as_loss:.4f}, "
                              f"lr={as_lr:.2e}, weight_norm={as_wnorm:.4f}")

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
