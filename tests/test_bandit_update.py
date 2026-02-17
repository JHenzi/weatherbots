import csv
import datetime as dt
import tempfile
import unittest

import bandit_update as bu


class BanditUpdateTests(unittest.TestCase):
    def test_reward_from_error_bounds(self):
        self.assertAlmostEqual(bu._reward_from_error(0.0, 10.0), 1.0)
        self.assertAlmostEqual(bu._reward_from_error(5.0, 10.0), 0.5)
        self.assertAlmostEqual(bu._reward_from_error(20.0, 10.0), 0.0)

    def test_load_latest_decisions_prefers_trade_role(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
            fieldnames = ["run_ts", "city", "trade_date", "decision_role", "selected_action"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(
                {
                    "run_ts": "2026-02-10T10:00:00+00:00",
                    "city": "ny",
                    "trade_date": "2026-02-10",
                    "decision_role": "monitoring",
                    "selected_action": "blend",
                }
            )
            w.writerow(
                {
                    "run_ts": "2026-02-10T09:00:00+00:00",
                    "city": "ny",
                    "trade_date": "2026-02-10",
                    "decision_role": "trade",
                    "selected_action": "forecast",
                }
            )
            path = f.name

        loaded = bu._load_latest_decisions(path, "2026-02-10")
        self.assertIn("ny", loaded)
        self.assertEqual((loaded["ny"].get("decision_role") or "").strip(), "trade")

    def test_parse_feature_vector_from_row(self):
        row = {
            "city": "ny",
            "spread_f": "2.1",
            "provider_count": "6",
            "condition_token": "partly_cloudy",
            "sky_label": "mixed",
            "mean_cloud_cover": "48.0",
            "vote_entropy": "0.12",
        }
        vec = bu._parse_feature_vector_from_row(row, dt.date(2026, 2, 10))
        self.assertTrue(len(vec) > 0)


if __name__ == "__main__":
    unittest.main()
