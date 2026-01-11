from .base import Strategy


class RuleBasedStrategy(Strategy):
    def select_action(self, model_output, env_state):
        # Simple rule: if large drawdown and in position -> CLOSE
        equity = env_state.get("equity_usd", 0)
        if equity < 9000 and env_state.get("position", 0) != 0:
            return 1  # CLOSE
        return int(model_output)
