import datetime as dt
import tempfile
import unittest

import numpy as np

from bandit.policy import FEATURE_NAMES, LinUCBPolicy, build_feature_vector, load_policy_state, save_policy_state


class BanditPolicyTests(unittest.TestCase):
    def test_feature_vector_shape(self):
        vec, fmap = build_feature_vector(
            city="ny",
            trade_date=dt.date(2026, 2, 1),
            spread_f=2.5,
            provider_count=6,
            condition_token="rain",
            sky_label="cloudy",
            mean_cloud_cover=87.0,
            vote_entropy=0.2,
        )
        self.assertEqual(vec.shape[0], len(FEATURE_NAMES))
        self.assertEqual(len(fmap), len(FEATURE_NAMES))

    def test_select_and_update(self):
        p = LinUCBPolicy(alpha=0.0, reg_lambda=1.0, epsilon=0.0)
        x = np.ones(len(FEATURE_NAMES), dtype=np.float64)
        action, meta = p.select_action(x, available_actions=["forecast", "blend"])
        self.assertIn(action, ("forecast", "blend"))
        self.assertEqual(meta.get("selected_via"), "linucb")

        for _ in range(10):
            p.update("blend", x, 1.0)
        action2, _ = p.select_action(x, available_actions=["forecast", "blend"])
        self.assertEqual(action2, "blend")

    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/bandit_state.json"
            p, state = load_policy_state(path, alpha=0.7, reg_lambda=1.0, epsilon=0.1)
            save_policy_state(path, p, state)
            p2, state2 = load_policy_state(path, alpha=0.7, reg_lambda=1.0, epsilon=0.1)
            self.assertEqual(p2.feature_dim, p.feature_dim)
            self.assertIn("policy", state2)


if __name__ == "__main__":
    unittest.main()
