"""NFSP hyperparameters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NFSPConfig:
    # Environment
    num_players: int = 2
    starting_stack: int = 10000
    small_blind: int = 50
    big_blind: int = 100
    num_envs: int = 1024

    # Network architecture
    input_dim: int = 590
    num_actions: int = 9
    hidden_dim: int = 1024
    residual_dim: int = 512
    history_input_dim: int = 11
    history_hidden_dim: int = 256
    max_history_len: int = 30  # max action history steps (heads-up ~6, 9-player ~20)
    history_embed_dim: int = 64    # transformer embedding dimension
    history_num_heads: int = 4     # transformer attention heads
    history_num_layers: int = 2    # transformer encoder layers
    history_ffn_dim: int = 128     # transformer feedforward dimension

    # NFSP parameters (eta is now scheduled, see eta_start/eta_end below)

    # Training
    total_episodes: int = 10_000_000
    batch_size: int = 4096
    br_lr: float = 1e-4      # best response learning rate
    as_lr: float = 1e-4      # average strategy learning rate (low to stabilize averaging)
    gamma: float = 1.0       # episodic, no discounting
    huber_delta: float = 10.0  # Huber loss beta — squared error for <10 BB, linear above

    # Replay buffers
    br_buffer_size: int = 1_000_000   # circular buffer for RL (~125k hands of recent experience)
    as_buffer_size: int = 4_000_000   # reservoir for SL (large to preserve long-run average)

    # Update frequencies — steps per training round
    br_train_steps: int = 8     # BR gradient steps per self-play batch
    as_train_steps: int = 4     # AS gradient steps per self-play batch
    target_update_every: int = 1    # update DQN target network (in training rounds); 1 = every round (Polyak)

    # Evaluation
    eval_every: int = 50_000
    eval_hands: int = 1_000
    checkpoint_every: int = 100_000

    # Learning rate schedule (cosine decay with warmup)
    # Note: all *_steps params below are in env steps (~8 steps/episode)
    lr_warmup_steps: int = 4_000_000      # linear warmup (~500k episodes)
    lr_min_factor: float = 0.01           # decay to lr * this factor (e.g. 1e-4 -> 1e-6)

    # Target network (Polyak soft update)
    tau: float = 0.005                    # soft update coefficient (1.0 = hard copy)

    # Epsilon-greedy for BR exploration
    epsilon_start: float = 0.10
    epsilon_end: float = 0.003
    epsilon_decay_steps: int = 200_000_000  # ~25M episodes — explore first quarter, exploit rest

    # Eta scheduling (anticipatory parameter): P(use AS policy) during self-play.
    # eta=0.1 means 10% AS / 90% BR. Higher eta → more average strategy play.
    # BR transitions are collected only from BR-policy actions (~is_as).
    eta_start: float = 0.1
    eta_end: float = 0.4
    eta_ramp_steps: int = 200_000_000       # ~25M episodes — gradual shift to AS

    # Hardware
    device: str = "cuda"  # ROCm via HIP exposes as cuda
    use_amp: Optional[bool] = None  # None=auto (on for CUDA), False disables AMP

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
