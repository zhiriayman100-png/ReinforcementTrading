from typing import Any, Optional
from stable_baselines3 import DQN
from .base import BaseAgent


class DQNAgent(BaseAgent):
    def __init__(self, model: DQN):
        self.model = model

    @classmethod
    def from_scratch(cls, env, **model_kwargs):
        model = DQN(env=env, **model_kwargs)
        return cls(model)

    @classmethod
    def load(cls, path: str, env=None):
        model = DQN.load(path, env=env)
        return cls(model)

    def train(self, env=None, total_timesteps: int = 100_000, callback=None, **kwargs):
        if env is not None:
            self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, callback=callback, **kwargs)

    def predict(self, obs, deterministic: bool = True) -> Any:
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str) -> None:
        self.model.save(path)
