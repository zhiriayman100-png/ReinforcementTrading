from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    @abstractmethod
    def select_action(self, model_output: Any, env_state: dict) -> int:
        """Return an environment action index."""
        raise NotImplementedError
