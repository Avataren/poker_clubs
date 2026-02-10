"""NFSP hyperparameters."""

from dataclasses import dataclass, field


@dataclass
class NFSPConfig:
    # Environment
    num_players: int = 6
    starting_stack: int = 10000
    small_blind: int = 50
    big_blind: int = 100
    num_envs: int = 1024

    # Network architecture
    input_dim: int = 569
    num_actions: int = 8
    hidden_dim: int = 1024
    residual_dim: int = 512
    lstm_input_dim: int = 7
    lstm_hidden_dim: int = 256
    lstm_layers: int = 2
    lstm_embed_dim: int = 128
    max_history_len: int = 30  # max action history steps (heads-up ~6, 9-player ~20)

    # NFSP parameters
    eta: float = 0.1  # anticipatory parameter (prob of using AS vs BR)

    # Training
    total_episodes: int = 10_000_000
    batch_size: int = 2048
    br_lr: float = 1e-4      # best response learning rate
    as_lr: float = 5e-4      # average strategy learning rate
    gamma: float = 1.0       # episodic, no discounting

    # Replay buffers
    br_buffer_size: int = 2_000_000   # circular buffer for RL
    as_buffer_size: int = 2_000_000   # reservoir for SL

    # Update frequencies — steps per training round
    br_train_steps: int = 8     # BR gradient steps per self-play batch
    as_train_steps: int = 4     # AS gradient steps per self-play batch
    target_update_every: int = 300  # update DQN target network (in training rounds)

    # Evaluation
    eval_every: int = 50_000
    checkpoint_every: int = 100_000

    # Epsilon-greedy for BR exploration
    epsilon_start: float = 0.06
    epsilon_end: float = 0.001
    epsilon_decay_steps: int = 2_000_000

    # Hardware
    device: str = "cuda"  # ROCm via HIP exposes as cuda

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
