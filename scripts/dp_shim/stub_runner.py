# DP's workspace calls env_runner.run at epoch 0 whatever rollout_every says; we evaluate with
# collect.py instead, so this runner reports nothing.
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class StubRunner(BaseImageRunner):
    def run(self, policy):
        return dict()
