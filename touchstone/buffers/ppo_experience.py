from dataclasses import dataclass
from typing import Any

from touchstone.buffers import Experience


@dataclass
class PPOExperience(Experience):
    action_log_probs: Any
    value: Any