"""Tests for neural network architecture."""

import torch
import pytest


def test_poker_net_forward():
    """Test forward pass with random input."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import PokerNet

    config = NFSPConfig()
    net = PokerNet(config)

    batch_size = 4
    obs = torch.randn(batch_size, 441)
    lstm_hidden = torch.randn(batch_size, config.lstm_hidden_dim)
    legal_mask = torch.ones(batch_size, config.num_actions, dtype=torch.bool)

    policy_logits, q_values = net(obs, lstm_hidden, legal_mask)

    assert policy_logits.shape == (batch_size, config.num_actions)
    assert q_values.shape == (batch_size, config.num_actions)


def test_poker_net_masking():
    """Test that illegal actions get -inf logits."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import PokerNet

    config = NFSPConfig()
    net = PokerNet(config)

    obs = torch.randn(1, 441)
    lstm_hidden = torch.randn(1, config.lstm_hidden_dim)
    legal_mask = torch.tensor([[True, True, False, False, False, False, False, True]])

    policy_logits, q_values = net(obs, lstm_hidden, legal_mask)

    # Illegal actions should have -inf
    assert policy_logits[0, 2] == float("-inf")
    assert policy_logits[0, 3] == float("-inf")
    assert q_values[0, 4] == float("-inf")


def test_br_net_action_selection():
    """Test BR network action selection."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import BestResponseNet

    config = NFSPConfig()
    net = BestResponseNet(config)

    obs = torch.randn(1, 441)
    action_history = torch.randn(1, config.max_history_len, 7)
    history_lengths = torch.tensor([10])
    legal_mask = torch.tensor([[False, True, False, False, True, False, False, True]])

    action = net.select_action(obs, action_history, history_lengths, legal_mask)
    assert action.item() in [1, 4, 7], f"Action {action.item()} should be legal"


def test_as_net_action_probs():
    """Test AS network produces valid probabilities."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import AverageStrategyNet

    config = NFSPConfig()
    net = AverageStrategyNet(config)

    obs = torch.randn(1, 441)
    action_history = torch.randn(1, config.max_history_len, 7)
    history_lengths = torch.tensor([5])
    legal_mask = torch.ones(1, config.num_actions, dtype=torch.bool)

    probs = net(obs, action_history, history_lengths, legal_mask)

    assert probs.shape == (1, config.num_actions)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)
    assert (probs >= 0).all()


def test_history_encoder_zero_input():
    """Test history MLP handles zero-padded input."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import ActionHistoryMLP

    config = NFSPConfig()
    mlp = ActionHistoryMLP(config)

    zero_seq = torch.zeros(2, config.max_history_len, 7)
    output = mlp(zero_seq)

    assert output.shape == (2, config.lstm_hidden_dim)


def test_parameter_count():
    """Verify approximate parameter count (~1.2M per network)."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import BestResponseNet, AverageStrategyNet

    config = NFSPConfig()
    br = BestResponseNet(config)
    as_net = AverageStrategyNet(config)

    br_params = sum(p.numel() for p in br.parameters())
    as_params = sum(p.numel() for p in as_net.parameters())

    print(f"BR params: {br_params:,}")
    print(f"AS params: {as_params:,}")

    # Should be roughly 1-2M each
    assert 500_000 < br_params < 5_000_000, f"BR params {br_params:,} out of expected range"
    assert 500_000 < as_params < 5_000_000, f"AS params {as_params:,} out of expected range"
