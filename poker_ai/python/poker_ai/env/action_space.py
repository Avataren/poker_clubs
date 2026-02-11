"""Discrete action space for poker AI."""

ACTION_NAMES = [
    "Fold",
    "Check/Call",
    "Raise 0.25x Pot",
    "Raise 0.4x Pot",
    "Raise 0.6x Pot",
    "Raise 0.8x Pot",
    "Raise 1x Pot",
    "Raise 1.5x Pot",
    "All-In",
]

NUM_ACTIONS = 9


def action_name(action_idx: int) -> str:
    """Get human-readable action name."""
    if 0 <= action_idx < NUM_ACTIONS:
        return ACTION_NAMES[action_idx]
    return f"Unknown({action_idx})"
