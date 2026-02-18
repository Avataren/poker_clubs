"""Neural network architectures for NFSP poker AI."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.model.state_encoder import STATIC_FEATURE_SIZE


class ActionHistoryTransformer(nn.Module):
    """Transformer encoder for action history sequences.

    Replaces the flat MLP with attention over the sequence of action records,
    capturing temporal patterns (e.g. raise-then-check indicating weakness).
    """

    def __init__(self, config: NFSPConfig):
        super().__init__()
        embed_dim = config.history_embed_dim  # 64
        max_len = config.max_history_len  # 30
        self.hidden_dim = config.history_hidden_dim  # 256

        # Project each 11-dim action record to embed space
        self.input_proj = nn.Linear(config.history_input_dim, embed_dim)

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.history_num_heads,
            dim_feedforward=config.history_ffn_dim,
            batch_first=True,
            dropout=0.0,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.history_num_layers,
            enable_nested_tensor=False,
        )

        # Output projection to history_hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        # Initialize output projection with small weights so initial output
        # is near-zero, preserving pretrained trunk behavior when resuming.
        with torch.no_grad():
            self.output_proj[0].weight.mul_(0.01)
            self.output_proj[0].bias.zero_()

    def forward(
        self, action_seq: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode action history with transformer attention.

        Args:
            action_seq: (batch, max_seq_len, 11) action features
            lengths: (batch,) number of valid entries per sample

        Returns:
            (batch, hidden_dim) encoded history
        """
        batch_size, seq_len, _ = action_seq.shape
        dtype = action_seq.dtype

        # All-float masking to avoid Triton codegen bug with bool types on ROCm.
        if lengths is not None:
            # Clamp to min 1 so softmax always has at least one unmasked position
            # (lengths==0 rows are zeroed out after pooling via has_history below)
            safe_lengths = lengths.clamp(min=1)
            positions = torch.arange(seq_len, device=action_seq.device).unsqueeze(0)
            # valid_mask: 1.0 for valid positions, 0.0 for padding (no bools)
            valid_mask_f = (positions < safe_lengths.unsqueeze(1)).to(dtype)  # (batch, seq_len)
            # Additive attention mask: 0.0 for valid, large negative for padding
            pad_mask = (1.0 - valid_mask_f) * (-1e4)
        else:
            valid_mask_f = None
            pad_mask = None

        # Project input and add positional embeddings
        x = self.input_proj(action_seq)  # (batch, seq_len, embed_dim)
        pos_ids = torch.arange(seq_len, device=action_seq.device)
        x = x + self.pos_embed(pos_ids).unsqueeze(0)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (batch, seq_len, embed_dim)

        # Mean pool over valid positions (float32 to prevent accumulation overflow)
        if valid_mask_f is not None:
            valid_counts = valid_mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
            x = (x.float() * valid_mask_f.unsqueeze(-1)).sum(dim=1) / valid_counts  # (batch, embed_dim)
            # Zero out output for fully empty histories (lengths==0)
            has_history = (lengths > 0).unsqueeze(1).to(dtype)  # (batch, 1) float
            x = (x * has_history).to(dtype)
        else:
            x = x.float().mean(dim=1).to(dtype)

        return self.output_proj(x)  # (batch, hidden_dim)


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PokerNet(nn.Module):
    """Shared trunk network with policy and value heads.

    Used for both Best Response (BR) and Average Strategy (AS) networks.
    BR uses the value head (Q-values per action).
    AS uses the policy head (action probabilities).
    """

    def __init__(self, config: NFSPConfig):
        super().__init__()
        self.config = config

        # Static features: observation without history placeholder
        # 364 (cards) + 166 (game state) + 52 (hand strength) = 582
        static_input = config.static_feature_size or STATIC_FEATURE_SIZE

        # Input layer combines static features + history encoding
        total_input = static_input + config.history_hidden_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(total_input, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            ResidualBlock(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.residual_dim),
            nn.LayerNorm(config.residual_dim),
            nn.ReLU(),
            ResidualBlock(config.residual_dim),
        )

        # Head hidden dim matches residual dim for more capacity
        head_dim = config.residual_dim  # 512

        # Policy head (for AS: outputs action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(config.residual_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, config.num_actions),
        )

        # Dueling DQN value head (for BR: outputs Q-value per action)
        # State value stream: V(s) — scalar
        self.value_stream = nn.Sequential(
            nn.Linear(config.residual_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        # Advantage stream: A(s,a) — per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.residual_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, config.num_actions),
        )

    def forward(
        self,
        obs: torch.Tensor,
        history_hidden: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (batch, 582) static features (cards + game state + hand strength)
            history_hidden: (batch, 256) history encoding
            legal_mask: (batch, 9) boolean mask of legal actions

        Returns:
            (policy_logits, q_values) both (batch, num_actions)
        """
        x = torch.cat([obs, history_hidden], dim=-1)
        trunk_out = self.trunk(x)

        # Run output heads in float32 to prevent float16 overflow.
        # The trunk output is bounded by LayerNorm+ReLU, but the heads
        # produce unbounded logits/Q-values that can exceed float16 range.
        # Disable autocast so Linear layers stay in float32.
        trunk_f32 = trunk_out.float()
        with torch.amp.autocast(trunk_out.device.type, enabled=False):
            policy_logits = self.policy_head(trunk_f32)

            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A)
            v = self.value_stream(trunk_f32)       # (batch, 1)
            a = self.advantage_stream(trunk_f32)   # (batch, num_actions)
            q_values = v + a - a.mean(dim=-1, keepdim=True)

            # Mask illegal actions (float math only — no bools for ROCm Triton compat)
            if legal_mask is not None:
                neg_inf = torch.tensor(-1e9, dtype=policy_logits.dtype, device=policy_logits.device)
                illegal_f = 1.0 - legal_mask.to(policy_logits.dtype)  # 1.0=illegal, 0.0=legal
                policy_logits = policy_logits + illegal_f * neg_inf
                q_values = q_values + illegal_f * neg_inf

        return policy_logits, q_values

    def policy(
        self,
        obs: torch.Tensor,
        history_hidden: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get action probabilities (softmax of policy logits)."""
        logits, _ = self.forward(obs, history_hidden, legal_mask)
        # Already masked by forward(); softmax over masked logits
        return F.softmax(logits, dim=-1)

    def q_values(
        self,
        obs: torch.Tensor,
        history_hidden: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-values for all actions."""
        _, q = self.forward(obs, history_hidden, legal_mask)
        return q


class BestResponseNet(nn.Module):
    """Best Response network (DQN) - wraps PokerNet's value head."""

    def __init__(self, config: NFSPConfig):
        super().__init__()
        self.net = PokerNet(config)
        self.history_encoder = ActionHistoryTransformer(config)
        self.zero_history = False  # set True to ablate history encoder

    def _encode_history(self, action_history, history_lengths):
        if self.zero_history:
            return torch.zeros(action_history.size(0), self.history_encoder.hidden_dim,
                               device=action_history.device, dtype=action_history.dtype)
        return self.history_encoder(action_history, history_lengths)

    def forward(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        history_hidden = self._encode_history(action_history, history_lengths)
        return self.net.q_values(obs, history_hidden, legal_mask)

    def select_action(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Epsilon-greedy action selection (batched)."""
        q = self.forward(obs, action_history, history_lengths, legal_mask)
        greedy = q.argmax(dim=-1)

        if epsilon > 0:
            batch_size = obs.size(0)
            # Per-sample random exploration
            explore_mask = torch.rand(batch_size, device=obs.device) < epsilon
            probs = legal_mask.float()
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            random_actions = torch.multinomial(probs, 1).squeeze(-1)
            return torch.where(explore_mask, random_actions, greedy)

        return greedy

    def select_action_with_greedy(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Epsilon-greedy action selection that also returns the greedy action.

        Returns (eps_greedy_action, greedy_action) in a single forward pass.
        """
        q = self.forward(obs, action_history, history_lengths, legal_mask)
        greedy = q.argmax(dim=-1)

        batch_size = obs.size(0)
        explore_mask = torch.rand(batch_size, device=obs.device) < epsilon
        probs = legal_mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        random_actions = torch.multinomial(probs, 1).squeeze(-1)
        eps_greedy = torch.where(explore_mask, random_actions, greedy)

        return eps_greedy, greedy


class AverageStrategyNet(nn.Module):
    """Average Strategy network (supervised learning)."""

    def __init__(self, config: NFSPConfig):
        super().__init__()
        self.net = PokerNet(config)
        self.history_encoder = ActionHistoryTransformer(config)
        self.zero_history = False  # set True to ablate history encoder

    def _encode_history(self, action_history, history_lengths):
        if self.zero_history:
            return torch.zeros(action_history.size(0), self.history_encoder.hidden_dim,
                               device=action_history.device, dtype=action_history.dtype)
        return self.history_encoder(action_history, history_lengths)

    def forward(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns action probabilities."""
        history_hidden = self._encode_history(action_history, history_lengths)
        return self.net.policy(obs, history_hidden, legal_mask)

    def forward_logits(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns raw masked logits (for numerically stable cross-entropy)."""
        history_hidden = self._encode_history(action_history, history_lengths)
        logits, _ = self.net.forward(obs, history_hidden, legal_mask)
        return logits

    def select_action(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample action from policy (only over legal actions)."""
        probs = self.forward(obs, action_history, history_lengths, legal_mask)
        # Replace any NaN/inf rows with uniform over legal actions
        bad = probs.isnan().any(dim=-1) | probs.isinf().any(dim=-1)
        if bad.any():
            uniform = legal_mask.float()
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            probs[bad] = uniform[bad]
        # Clamp only legal actions to avoid zero-probability; keep illegal at zero
        probs = torch.where(legal_mask, probs.clamp(min=1e-8), torch.zeros_like(probs))
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.multinomial(probs, 1).squeeze(-1)
