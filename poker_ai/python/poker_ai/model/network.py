"""Neural network architectures for NFSP poker AI."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from poker_ai.config.hyperparams import NFSPConfig


class ActionHistoryMLP(nn.Module):
    """MLP for encoding flattened action history."""

    def __init__(self, config: NFSPConfig):
        super().__init__()
        flat_input = config.max_history_len * config.history_input_dim  # 30 * 11 = 330
        hidden = config.history_hidden_dim  # output dim (256)
        self.net = nn.Sequential(
            nn.Linear(flat_input, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.hidden_dim = hidden

    def forward(
        self, action_seq: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode action history.

        Args:
            action_seq: (batch, max_seq_len, 11) action features
            lengths: ignored (kept for API compatibility)

        Returns:
            (batch, hidden_dim) encoded history
        """
        return self.net(action_seq.reshape(action_seq.size(0), -1))


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
        # 364 (cards) + 46 (game state) + 52 (hand strength) = 462
        static_input = 462

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

        # Value head (for BR: outputs Q-value per action)
        self.value_head = nn.Sequential(
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
            obs: (batch, 462) static features (cards + game state + hand strength)
            history_hidden: (batch, 256) history encoding
            legal_mask: (batch, 9) boolean mask of legal actions

        Returns:
            (policy_logits, q_values) both (batch, num_actions)
        """
        x = torch.cat([obs, history_hidden], dim=-1)
        trunk_out = self.trunk(x)

        policy_logits = self.policy_head(trunk_out)
        q_values = self.value_head(trunk_out)

        # Mask illegal actions
        if legal_mask is not None:
            illegal_mask = ~legal_mask
            policy_logits = policy_logits.masked_fill(illegal_mask, float("-inf"))
            q_values = q_values.masked_fill(illegal_mask, float("-inf"))

        return policy_logits, q_values

    def policy(
        self,
        obs: torch.Tensor,
        history_hidden: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get action probabilities (softmax of policy logits)."""
        logits, _ = self.forward(obs, history_hidden, legal_mask)
        # Use torch.where for proper masking instead of clamp
        safe_logits = torch.where(
            legal_mask, logits, torch.tensor(-1e9, device=logits.device)
        )
        return F.softmax(safe_logits, dim=-1)

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
        self.history_encoder = ActionHistoryMLP(config)

    def forward(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        history_hidden = self.history_encoder(action_history, history_lengths)
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


class AverageStrategyNet(nn.Module):
    """Average Strategy network (supervised learning)."""

    def __init__(self, config: NFSPConfig):
        super().__init__()
        self.net = PokerNet(config)
        self.history_encoder = ActionHistoryMLP(config)

    def forward(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns action probabilities."""
        history_hidden = self.history_encoder(action_history, history_lengths)
        return self.net.policy(obs, history_hidden, legal_mask)

    def forward_logits(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns raw masked logits (for numerically stable cross-entropy)."""
        history_hidden = self.history_encoder(action_history, history_lengths)
        logits, _ = self.net.forward(obs, history_hidden, legal_mask)
        return logits

    def select_action(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor | None,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample action from policy."""
        probs = self.forward(obs, action_history, history_lengths, legal_mask)
        # Replace any NaN/inf rows with uniform over legal actions
        bad = probs.isnan().any(dim=-1) | probs.isinf().any(dim=-1)
        if bad.any():
            uniform = legal_mask.float()
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            probs[bad] = uniform[bad]
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, 1).squeeze(-1)
