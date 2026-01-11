from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    def train(self, env, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, obs, deterministic: bool = True) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str, env=None):
        raise NotImplementedError
