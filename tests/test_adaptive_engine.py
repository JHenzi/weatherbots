import datetime as dt
import unittest

import adaptive_ensemble as ae
import decision_policy as dp
import feedback_loop as fl


class NormalizeWithCapTests(unittest.TestCase):
    def test_caps_runaway_single_source(self):
        # A provider with MAE -> 0 would otherwise take ~100% of the ensemble.
        raw = {"a": 10000.0, "b": 1.0, "c": 1.0, "d": 1.0}
        w = ae._normalize_with_cap(raw, max_weight=0.40)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)
        self.assertLessEqual(w["a"], 0.40 + 1e-9)
        self.assertGreater(w["b"], 0.0)

    def test_falls_back_to_uniform_when_cap_infeasible(self):
        # 2 providers cannot both sit under a 0.4 cap.
        w = ae._normalize_with_cap({"a": 9.0, "b": 1.0}, max_weight=0.40)
        self.assertAlmostEqual(w["a"], 0.5, places=9)
        self.assertAlmostEqual(w["b"], 0.5, places=9)

    def test_preserves_ordering_when_uncapped(self):
        w = ae._normalize_with_cap({"a": 4.0, "b": 2.0, "c": 1.0}, max_weight=0.9)
        self.assertGreater(w["a"], w["b"])
        self.assertGreater(w["b"], w["c"])


class EwmaStatsTests(unittest.TestCase):
    def _obs(self, city="ny", source="x", err=2.0, days=(0, 1, 2)):
        base = dt.date(2026, 8, 31)
        return [(base - dt.timedelta(days=d), city, source, 70.0 + err, 70.0) for d in days]

    def test_recent_errors_dominate(self):
        base = dt.date(2026, 8, 31)
        obs = [
            (base, "ny", "x", 71.0, 70.0),                        # today, 1F off
            (base - dt.timedelta(days=20), "ny", "x", 78.0, 70.0),  # 20d ago, 8F off
        ]
        stats = ae.accumulate_stats(obs, as_of=base, half_life_days=7.0, window_days=30)
        mae = stats["ny"]["x"].ewma_mae
        # Flat mean would be 4.5; decay must pull it well below that.
        self.assertLess(mae, 3.0)
        self.assertGreater(mae, 1.0)

    def test_winsorization_caps_outlier(self):
        base = dt.date(2026, 8, 31)
        obs = [(base, "ny", "x", 150.0, 70.0)]  # 80F error, as seen in the real log
        stats = ae.accumulate_stats(obs, as_of=base, winsor_cap_f=8.0, window_days=30)
        self.assertLessEqual(stats["ny"]["x"].ewma_mae, 8.0 + 1e-9)

    def test_excludes_consensus_and_lstm(self):
        base = dt.date(2026, 8, 31)
        obs = [
            (base, "ny", "consensus", 71.0, 70.0),
            (base, "ny", "lstm", 71.0, 70.0),
            (base, "ny", "tomorrow", 71.0, 70.0),
        ]
        stats = ae.accumulate_stats(obs, as_of=base, window_days=30)
        self.assertEqual(set(stats["ny"]), {"tomorrow"})

    def test_shrinkage_pulls_small_samples_to_prior(self):
        st = ae.ProviderStats(source="x", city="ny", ewma_mae=0.05, weight_mass=0.5)
        shrunk = st.shrunk_mae(prior_mae=3.0, prior_strength=3.0)
        self.assertGreater(shrunk, 2.0)  # one lucky day should not read as 0.05F skill


class WeightingTests(unittest.TestCase):
    def test_low_availability_provider_excluded(self):
        st = ae.ProviderStats(source="flaky", city="ny", ewma_mae=1.0,
                              weight_mass=2.0, n_obs=2, n_days_in_window=30)
        good = ae.ProviderStats(source="solid", city="ny", ewma_mae=2.0,
                                weight_mass=25.0, n_obs=30, n_days_in_window=30)
        ws = ae.compute_weights({"flaky": st, "solid": good}, city="ny",
                                min_availability=0.25)
        self.assertIn("flaky", ws.excluded)
        self.assertIn("solid", ws.weights)

    def test_effective_sample_size(self):
        self.assertAlmostEqual(ae.effective_sample_size({"a": 0.5, "b": 0.5}), 2.0, places=6)
        self.assertAlmostEqual(ae.effective_sample_size({"a": 1.0, "b": 0.0}), 1.0, places=6)

    def test_weighted_consensus_applies_bias(self):
        v = ae.weighted_consensus({"a": 70.0, "b": 80.0}, {"a": 0.5, "b": 0.5},
                                  bias_correction_f=1.0)
        self.assertAlmostEqual(v, 76.0, places=6)


class DecisionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.p = dp.PolicyParams()

    def test_required_confidence_rises_with_spread_and_sigma(self):
        low = dp.required_confidence(yes_spread_cents=1.0, sigma_f=1.0,
                                     hours_to_settlement=12.0, params=self.p)
        high = dp.required_confidence(yes_spread_cents=15.0, sigma_f=5.0,
                                      hours_to_settlement=12.0, params=self.p)
        self.assertGreater(high, low)

    def test_required_confidence_relaxes_near_settlement(self):
        early = dp.required_confidence(yes_spread_cents=8.0, sigma_f=3.0,
                                       hours_to_settlement=24.0, params=self.p)
        late = dp.required_confidence(yes_spread_cents=8.0, sigma_f=3.0,
                                      hours_to_settlement=0.5, params=self.p)
        self.assertLess(late, early)

    def test_big_edge_at_moderate_confidence_is_not_vetoed(self):
        # The behaviour the old fixed 0.75 gate could not express.
        d = dp.evaluate(
            model_prob_yes=0.80, market_prob_yes=0.30, yes_ask_cents=30.0,
            yes_bid_cents=28.0, effective_confidence=0.60, sigma_f=1.0,
            hours_to_settlement=8.0, bankroll_dollars=50.0, params=self.p,
        )
        self.assertEqual(d.action, "trade")
        self.assertGreater(d.contracts, 0)

    def test_tiny_edge_at_high_confidence_is_skipped(self):
        d = dp.evaluate(
            model_prob_yes=0.52, market_prob_yes=0.50, yes_ask_cents=50.0,
            yes_bid_cents=49.0, effective_confidence=0.95, sigma_f=0.5,
            hours_to_settlement=8.0, bankroll_dollars=50.0, params=self.p,
        )
        self.assertEqual(d.action, "skip")
        self.assertIn("edge_too_small", d.reason)

    def test_undiversified_ensemble_blocks_trade(self):
        d = dp.evaluate(
            model_prob_yes=0.90, market_prob_yes=0.20, yes_ask_cents=20.0,
            yes_bid_cents=19.0, effective_confidence=0.9, sigma_f=1.0,
            hours_to_settlement=8.0, bankroll_dollars=50.0, params=self.p,
            effective_sources=1.1,
        )
        self.assertEqual(d.action, "skip")
        self.assertIn("undiversified", d.reason)

    def test_calibrator_is_applied_before_ev(self):
        # Raw 0.80 looks great; a calibrator that maps it to 0.31 must kill the trade.
        d = dp.evaluate(
            model_prob_yes=0.80, market_prob_yes=0.30, yes_ask_cents=30.0,
            yes_bid_cents=29.0, effective_confidence=0.7, sigma_f=1.0,
            hours_to_settlement=8.0, bankroll_dollars=50.0, params=self.p,
            calibrator=lambda p: 0.31,
        )
        self.assertEqual(d.action, "skip")
        self.assertAlmostEqual(d.calibrated_prob, 0.31, places=6)

    def test_depth_limits_size(self):
        d = dp.evaluate(
            model_prob_yes=0.90, market_prob_yes=0.20, yes_ask_cents=20.0,
            yes_bid_cents=19.0, effective_confidence=0.9, sigma_f=0.5,
            hours_to_settlement=8.0, bankroll_dollars=500.0, params=self.p,
            ask_depth=10,
        )
        self.assertLessEqual(d.contracts, 5)  # max_fraction_of_depth = 0.5

    def test_kelly_declines_negative_edge(self):
        c, dollars = dp.kelly_contracts(calibrated_prob=0.10, yes_ask_cents=50.0,
                                        bankroll_dollars=100.0, params=self.p)
        self.assertEqual(c, 0)
        self.assertEqual(dollars, 0.0)

    def test_bucket_probability_uses_sigma_floor(self):
        # sigma=0 must not yield a degenerate 0/1 probability.
        p = dp.bucket_probability(mu_f=70.0, sigma_f=0.0, bucket_lo=69.0, bucket_hi=71.0)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)


class CalibrationTests(unittest.TestCase):
    def test_pava_is_monotone(self):
        rows = []
        # Deliberately non-monotone raw probabilities, like the real logs.
        for _ in range(40):
            rows.append({"model_prob_yes": "0.05", "bucket_hit": "true"})
        for _ in range(40):
            rows.append({"model_prob_yes": "0.45", "bucket_hit": "false"})
        for _ in range(40):
            rows.append({"model_prob_yes": "0.80", "bucket_hit": "true"})
        cal = fl.fit_calibration(rows)
        ys = [cal.predict(x / 20.0) for x in range(21)]
        for a, b in zip(ys, ys[1:]):
            self.assertLessEqual(a, b + 1e-9)

    def test_identity_when_too_few_rows(self):
        cal = fl.fit_calibration([{"model_prob_yes": "0.5", "bucket_hit": "true"}])
        self.assertAlmostEqual(cal.predict(0.42), 0.42, places=6)

    def test_smoothing_keeps_probabilities_off_the_bounds(self):
        rows = [{"model_prob_yes": "0.9", "bucket_hit": "true"} for _ in range(80)]
        cal = fl.fit_calibration(rows, smoothing=0.02)
        self.assertLess(cal.predict(0.9), 1.0)

    def test_significance_gate_blocks_tuning_on_noise(self):
        # Coin-flip outcomes at a fair price -> no real edge.
        rows = []
        for i in range(200):
            rows.append({
                "model_prob_yes": "0.5", "yes_ask": "50", "yes_bid": "49",
                "count": "10", "confidence_score": "0.8", "sigma_f": "1.0",
                "market_prob_yes": "0.5",
                "bucket_hit": "true" if i % 2 == 0 else "false",
            })
        _, report = fl.tune_params(rows, dp.PolicyParams())
        self.assertEqual(report["status"], "insignificant_edge")

    def test_pnl_significance_arithmetic(self):
        rows = [{"count": "10", "yes_ask": "50", "bucket_hit": "true"},
                {"count": "10", "yes_ask": "50", "bucket_hit": "false"}]
        sig = fl.pnl_significance(rows)
        self.assertEqual(sig["n"], 2)
        self.assertAlmostEqual(sig["total"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
