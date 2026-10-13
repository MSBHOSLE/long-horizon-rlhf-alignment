import unittest
from alignment.dpo_loss import DPOLossCalculator


class TestDPOAlignment(unittest.TestCase):
    def test_dpo_loss_preference_margin(self):
        calc = DPOLossCalculator(beta=0.1)

        # Policy assigns higher probability to chosen vs reference, and lower to rejected
        res = calc.compute_loss(
            policy_chosen_logps=-1.2,
            policy_rejected_logps=-3.5,
            reference_chosen_logps=-2.0,
            reference_rejected_logps=-2.0,
        )

        self.assertGreater(res["reward_margin"], 0.0)
        self.assertGreater(res["chosen_reward"], res["rejected_reward"])
        self.assertGreater(res["loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
