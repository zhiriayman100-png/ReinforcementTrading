from .base import Strategy


class ModelPolicyStrategy(Strategy):
    def select_action(self, model_output, env_state):
        # model_output is expected to be an integer action already
        return int(model_output)
