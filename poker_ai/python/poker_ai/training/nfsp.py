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
from poker_ai.training.circular_buffer import CircularBuffer, Transition
from poker_ai.training.reservoir import ReservoirBuffer, SLTransition
from poker_ai.training.self_play import SelfPlayWorker, extract_static_features, pad_action_history
from poker_ai.env.poker_env import PokerEnv


def collate_transitions(batch: list[Transition], device: torch.device):
    """Collate a batch of RL transitions into tensors."""
    obs = torch.tensor(np.stack([t.obs for t in batch]), device=device)
    ah = torch.tensor(np.stack([t.action_history for t in batch]), device=device)
    ah_len = torch.tensor([t.history_length for t in batch], device=device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), device=device)
    next_ah = torch.tensor(np.stack([t.next_action_history for t in batch]), device=device)
    next_ah_len = torch.tensor([t.next_history_length for t in batch], device=device)
    next_mask = torch.tensor(np.stack([t.next_legal_mask for t in batch]), device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
    masks = torch.tensor(np.stack([t.legal_mask for t in batch]), device=device)
    return obs, ah, ah_len, actions, rewards, next_obs, next_ah, next_ah_len, next_mask, dones, masks


def collate_sl_transitions(batch: list[SLTransition], device: torch.device):
    """Collate a batch of SL transitions into tensors."""
    obs = torch.tensor(np.stack([t.obs for t in batch]), device=device)
    ah = torch.tensor(np.stack([t.action_history for t in batch]), device=device)
    ah_len = torch.tensor([t.history_length for t in batch], device=device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    masks = torch.tensor(np.stack([t.legal_mask for t in batch]), device=device)
    return obs, ah, ah_len, actions, masks


class NFSPTrainer:
    """NFSP training manager."""

    def __init__(self, config: NFSPConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.br_net = BestResponseNet(config).to(self.device)
        self.br_target = copy.deepcopy(self.br_net).to(self.device)
        self.br_target.eval()
        self.as_net = AverageStrategyNet(config).to(self.device)

        # Optimizers
        self.br_optimizer = torch.optim.Adam(self.br_net.parameters(), lr=config.br_lr)
        self.as_optimizer = torch.optim.Adam(self.as_net.parameters(), lr=config.as_lr)

        # Replay buffers
        self.br_buffer = CircularBuffer(config.br_buffer_size)
        self.as_buffer = ReservoirBuffer(config.as_buffer_size)

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

    def train_br_step(self) -> float:
        """One DQN training step on Best Response network."""
        if len(self.br_buffer) < self.config.batch_size:
            return 0.0

        batch = self.br_buffer.sample(self.config.batch_size)
        (obs, ah, ah_len, actions, rewards, next_obs, next_ah,
         next_ah_len, next_mask, dones, masks) = collate_transitions(batch, self.device)

        # Current Q-values
        q_values = self.br_net(obs, ah, ah_len, masks)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_q_online = self.br_net(next_obs, next_ah, next_ah_len, next_mask)
            next_actions = next_q_online.argmax(dim=-1)
            next_q_target = self.br_target(next_obs, next_ah, next_ah_len, next_mask)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.config.gamma * next_q * (1 - dones)

        # Huber loss
        loss = F.smooth_l1_loss(q_taken, target)

        self.br_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.br_net.parameters(), 10.0)
        self.br_optimizer.step()

        self.br_updates += 1
        return loss.item()

    def train_as_step(self) -> float:
        """One supervised learning step on Average Strategy network."""
        if len(self.as_buffer) < self.config.batch_size:
            return 0.0

        batch = self.as_buffer.sample(self.config.batch_size)
        obs, ah, ah_len, actions, masks = collate_sl_transitions(batch, self.device)

        # Get raw logits (numerically stable for cross-entropy)
        logits = self.as_net.forward_logits(obs, ah, ah_len, masks)

        # Cross-entropy loss on logits
        loss = F.cross_entropy(logits, actions)

        self.as_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.as_net.parameters(), 10.0)
        self.as_optimizer.step()

        self.as_updates += 1
        return loss.item()

    def update_target_network(self):
        """Copy BR network weights to target network."""
        self.br_target.load_state_dict(self.br_net.state_dict())

    def train(self):
        """Main training loop."""
        print(f"Starting NFSP training for {self.config.total_episodes} episodes")
        print(f"BR buffer: {self.config.br_buffer_size}, AS buffer: {self.config.as_buffer_size}")

        start_time = time.time()
        episode_count = 0

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
            train_rounds = episode_count // self.config.num_envs
            if train_rounds % self.config.target_update_every == 0:
                self.update_target_network()

            # Logging
            if episode_count % 10000 == 0:
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

            # Evaluation
            if episode_count % self.config.eval_every == 0:
                self.evaluate(episode_count)

            # Checkpointing
            if episode_count % self.config.checkpoint_every == 0:
                self.save_checkpoint(episode_count)

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

        total_reward = 0.0
        hero_seat = 0
        action_history: list[list[np.ndarray]] = [[], []]

        for _ in range(num_hands):
            player, obs, mask = env.reset()
            action_history = [[], []]
            done = False

            while not done:
                if player == hero_seat:
                    # Use AS network
                    static_obs = extract_static_features(obs)
                    ah_padded, ah_len = pad_action_history(action_history[player])
                    obs_t = torch.tensor(static_obs, device=self.device).unsqueeze(0)
                    ah_t = torch.tensor(ah_padded, device=self.device).unsqueeze(0)
                    ah_len_t = torch.tensor([ah_len], device=self.device)
                    mask_t = torch.tensor(mask, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        action = self.as_net.select_action(obs_t, ah_t, ah_len_t, mask_t).item()
                else:
                    # Opponent strategy
                    if opponent == "random":
                        legal = np.where(mask)[0]
                        action = np.random.choice(legal) if len(legal) > 0 else 1
                    elif opponent == "caller":
                        action = 1  # always check/call
                    else:
                        action = 1

                # Update action history
                action_record = np.zeros(7, dtype=np.float32)
                action_record[0] = player
                if action == 0:
                    action_record[1] = 1.0
                elif action == 1:
                    action_record[2] = 1.0
                elif action <= 3:
                    action_record[3] = 1.0
                elif action <= 5:
                    action_record[4] = 1.0
                else:
                    action_record[5] = 1.0
                action_record[6] = min(action / 7.0, 1.0)

                for p in range(2):
                    action_history[p].append(action_record.copy())

                player, obs, mask, rewards, done = env.step(action)

                if done:
                    total_reward += rewards[hero_seat]

        # Convert to bb/100
        bb_per_100 = (total_reward / num_hands) * 100
        return bb_per_100

    def save_checkpoint(self, episode: int):
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "total_steps": self.total_steps,
            "br_net": self.br_net.state_dict(),
            "br_target": self.br_target.state_dict(),
            "as_net": self.as_net.state_dict(),
            "br_optimizer": self.br_optimizer.state_dict(),
            "as_optimizer": self.as_optimizer.state_dict(),
        }
        torch.save(checkpoint, path / f"checkpoint_{episode}.pt")
        torch.save(checkpoint, path / "checkpoint_latest.pt")
        print(f"  Saved checkpoint at episode {episode:,}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.br_net.load_state_dict(checkpoint["br_net"])
        self.br_target.load_state_dict(checkpoint["br_target"])
        self.as_net.load_state_dict(checkpoint["as_net"])
        self.br_optimizer.load_state_dict(checkpoint["br_optimizer"])
        self.as_optimizer.load_state_dict(checkpoint["as_optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.total_episodes = checkpoint["episode"]
        print(f"Loaded checkpoint from episode {checkpoint['episode']:,}")
