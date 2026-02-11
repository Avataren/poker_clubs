"""NFSP hyperparameters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NFSPConfig:
    # Environment
    num_players: int = 6
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

    # NFSP parameters (eta is now scheduled, see eta_start/eta_end below)

    # Training
    total_episodes: int = 10_000_000
    batch_size: int = 4096
    br_lr: float = 1e-4      # best response learning rate
    as_lr: float = 1e-4      # average strategy learning rate (low to stabilize averaging)
    gamma: float = 1.0       # episodic, no discounting

    # Replay buffers
    br_buffer_size: int = 1_000_000   # circular buffer for RL
    as_buffer_size: int = 5_000_000   # reservoir for SL (large to preserve long-run average)

    # Update frequencies — steps per training round
    br_train_steps: int = 8     # BR gradient steps per self-play batch
    as_train_steps: int = 4     # AS gradient steps per self-play batch
    target_update_every: int = 300  # update DQN target network (in training rounds)

    # Evaluation
    eval_every: int = 50_000
    eval_hands: int = 1_000
    checkpoint_every: int = 100_000

    # Learning rate schedule (cosine decay with warmup)
    lr_warmup_steps: int = 500_000        # linear warmup over this many env steps
    lr_min_factor: float = 0.01           # decay to lr * this factor (e.g. 1e-4 -> 1e-6)

    # Target network (Polyak soft update)
    tau: float = 0.005                    # soft update coefficient (1.0 = hard copy)

    # Epsilon-greedy for BR exploration
    epsilon_start: float = 0.12
    epsilon_end: float = 0.003
    epsilon_decay_steps: int = 40_000_000

    # Eta scheduling (AS/BR mix linear ramp)
    eta_start: float = 0.1
    eta_end: float = 0.4
    eta_ramp_steps: int = 30_000_000

    # Hardware
    device: str = "cuda"  # ROCm via HIP exposes as cuda
    use_amp: Optional[bool] = None  # None=auto (on for CUDA), False disables AMP

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
