"""Tests for neural network architecture."""

import torch
import pytest

from poker_ai.model.state_encoder import STATIC_FEATURE_SIZE


def test_poker_net_forward():
    """Test forward pass with random input."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import PokerNet

    config = NFSPConfig()
    net = PokerNet(config)

    batch_size = 4
    obs = torch.randn(batch_size, STATIC_FEATURE_SIZE)
    history_hidden = torch.randn(batch_size, config.history_hidden_dim)
    legal_mask = torch.ones(batch_size, config.num_actions, dtype=torch.bool)

    policy_logits, q_values = net(obs, history_hidden, legal_mask)

    assert policy_logits.shape == (batch_size, config.num_actions)
    assert q_values.shape == (batch_size, config.num_actions)


def test_poker_net_masking():
    """Test that illegal actions get -inf logits."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import PokerNet

    config = NFSPConfig()
    net = PokerNet(config)

    obs = torch.randn(1, STATIC_FEATURE_SIZE)
    history_hidden = torch.randn(1, config.history_hidden_dim)
    legal_mask = torch.tensor([[True, True, False, False, False, False, False, True, True]])

    policy_logits, q_values = net(obs, history_hidden, legal_mask)

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

    obs = torch.randn(1, STATIC_FEATURE_SIZE)
    action_history = torch.randn(1, config.max_history_len, config.history_input_dim)
    history_lengths = torch.tensor([10])
    legal_mask = torch.tensor([[False, True, False, False, True, False, False, True, True]])

    action = net.select_action(obs, action_history, history_lengths, legal_mask)
    assert action.item() in [1, 4, 7, 8], f"Action {action.item()} should be legal"


def test_as_net_action_probs():
    """Test AS network produces valid probabilities."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import AverageStrategyNet

    config = NFSPConfig()
    net = AverageStrategyNet(config)

    obs = torch.randn(1, STATIC_FEATURE_SIZE)
    action_history = torch.randn(1, config.max_history_len, config.history_input_dim)
    history_lengths = torch.tensor([5])
    legal_mask = torch.ones(1, config.num_actions, dtype=torch.bool)

    probs = net(obs, action_history, history_lengths, legal_mask)

    assert probs.shape == (1, config.num_actions)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)
    assert (probs >= 0).all()


def test_as_net_no_illegal_mass():
    """Test AS select_action gives zero probability to illegal actions."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import AverageStrategyNet

    config = NFSPConfig()
    net = AverageStrategyNet(config)

    obs = torch.randn(100, STATIC_FEATURE_SIZE)
    action_history = torch.randn(100, config.max_history_len, config.history_input_dim)
    history_lengths = torch.full((100,), 5)
    # Only actions 0, 1, 8 are legal
    legal_mask = torch.zeros(100, config.num_actions, dtype=torch.bool)
    legal_mask[:, 0] = True
    legal_mask[:, 1] = True
    legal_mask[:, 8] = True

    with torch.no_grad():
        actions = net.select_action(obs, action_history, history_lengths, legal_mask)

    # All selected actions should be legal
    for a in actions:
        assert a.item() in [0, 1, 8], f"Action {a.item()} is illegal"


def test_history_transformer_shapes():
    """Test transformer history encoder produces correct shapes."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import ActionHistoryTransformer

    config = NFSPConfig()
    encoder = ActionHistoryTransformer(config)

    # Non-empty history
    seq = torch.randn(4, config.max_history_len, config.history_input_dim)
    lengths = torch.tensor([5, 10, 30, 1])
    output = encoder(seq, lengths)
    assert output.shape == (4, config.history_hidden_dim)

    # Empty history should produce zeros
    zero_seq = torch.zeros(2, config.max_history_len, config.history_input_dim)
    zero_lengths = torch.tensor([0, 0])
    zero_output = encoder(zero_seq, zero_lengths)
    assert zero_output.shape == (2, config.history_hidden_dim)
    assert torch.allclose(zero_output, torch.zeros_like(zero_output))

    # No lengths provided
    output_no_len = encoder(seq)
    assert output_no_len.shape == (4, config.history_hidden_dim)


def test_parameter_count():
    """Verify approximate parameter count."""
    from poker_ai.config.hyperparams import NFSPConfig
    from poker_ai.model.network import BestResponseNet, AverageStrategyNet

    config = NFSPConfig()
    br = BestResponseNet(config)
    as_net = AverageStrategyNet(config)

    br_params = sum(p.numel() for p in br.parameters())
    as_params = sum(p.numel() for p in as_net.parameters())

    print(f"BR params: {br_params:,}")
    print(f"AS params: {as_params:,}")

    # With transformer history encoder, expect ~2-6M each
    assert 500_000 < br_params < 10_000_000, f"BR params {br_params:,} out of expected range"
    assert 500_000 < as_params < 10_000_000, f"AS params {as_params:,} out of expected range"
