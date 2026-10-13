import math
from typing import Dict, List, Tuple


class DPOLossCalculator:
    """Direct Preference Optimization (DPO) loss and implicit reward calculation for LLM alignment."""

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Numerically stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def compute_loss(
        self,
        policy_chosen_logps: float,
        policy_rejected_logps: float,
        reference_chosen_logps: float,
        reference_rejected_logps: float,
    ) -> Dict[str, float]:
        """Calculates DPO loss, implicit rewards, and preference margins.

        Args:
            policy_chosen_logps: log pi_theta(y_w | x)
            policy_rejected_logps: log pi_theta(y_l | x)
            reference_chosen_logps: log pi_ref(y_w | x)
            reference_rejected_logps: log pi_ref(y_l | x)
        """
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        logits = self.beta * (chosen_logratios - rejected_logratios)
        prob = self._sigmoid(logits)
        # Binary cross entropy loss: -log(sigmoid(logits))
        loss = -math.log(max(1e-12, prob))

        # Implicit rewards
        reward_chosen = self.beta * chosen_logratios
        reward_rejected = self.beta * rejected_logratios
        margin = reward_chosen - reward_rejected

        return {
            "loss": round(loss, 6),
            "chosen_reward": round(reward_chosen, 4),
            "rejected_reward": round(reward_rejected, 4),
            "reward_margin": round(margin, 4),
        }
