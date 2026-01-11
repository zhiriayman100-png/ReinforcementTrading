# Agents — README

This folder contains agent wrappers that adapt Stable Baselines 3 models to a small project-friendly interface.

Key files

- `base.py` — `BaseAgent` abstract base class. Any custom agent should implement `train`, `predict`, `save` and `load`.
- `ppo_agent.py` — a thin wrapper around Stable Baselines 3's `PPO` model (provides `from_scratch`, `load`, `train`, `predict`, `save`).
- `dqn_agent.py` — wrapper around `DQN` with similar interface.

How agents are used

`train_agent.py` selects an agent by name ("PPO" or "DQN"), creates an instance with `.from_scratch(... )` and calls `.train(...)`. After training the code evaluates checkpoints and uses `.load(...)` to restore saved models.

Adding a custom agent

1. Create a new file in `agents/`, e.g. `my_agent.py`.
2. Implement a class that inherits from `BaseAgent` and implements the required methods:

```python
from .base import BaseAgent

class MyAgent(BaseAgent):
    @classmethod
    def from_scratch(cls, env, **kwargs):
        # return a wrapper instance with an untrained model
        ...

    @classmethod
    def load(cls, path: str, env=None):
        # load a trained model from disk and return wrapper instance
        ...

    def train(self, env=None, total_timesteps: int = 100_000, callback=None, **kwargs):
        # train the underlying model
        ...

    def predict(self, obs, deterministic: bool = True):
        # return action(s)
        ...

    def save(self, path: str):
        # persist model
        ...
```

3. Update `train_agent.py` to recognize the new `model_type` string and instantiate your `MyAgent` in the agent selection block (or implement a simple registry to avoid editing `train_agent.py` regularly).

Notes & tips

- If your agent uses Stable Baselines 3, follow the same `.from_scratch`/`.load` pattern used by `ppo_agent.py` and `dqn_agent.py`.
- Ensure `predict` returns `(action, info)` if you mimic SB3 APIs, or adapt `train_agent.evaluate_model` to your agent's predict signature.
- Write unit tests exercising `train` and `load`/`save` behavior, and keep a small example training run to validate checkpointing semantics.

If you'd like, I can add a small example custom agent or a registry to `train_agent.py` to avoid manual edits when adding new agents.