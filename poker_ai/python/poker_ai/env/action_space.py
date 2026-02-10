"""Discrete action space for poker AI."""

ACTION_NAMES = [
    "Fold",
    "Check/Call",
    "Raise 0.5x Pot",
    "Raise 0.75x Pot",
    "Raise 1x Pot",
    "Raise 1.5x Pot",
    "Raise 2x Pot",
    "All-In",
]

NUM_ACTIONS = 8


def action_name(action_idx: int) -> str:
    """Get human-readable action name."""
    if 0 <= action_idx < NUM_ACTIONS:
        return ACTION_NAMES[action_idx]
    return f"Unknown({action_idx})"
