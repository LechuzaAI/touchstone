from dataclasses import dataclass
from typing import Any


@dataclass
class Experience:
    state: Any
    action: Any
    reward: Any
    done: Any
    new_state: Any
    action_log_prob: Any
    value: Any
