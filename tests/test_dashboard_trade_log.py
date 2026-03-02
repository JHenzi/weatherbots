import csv
import tempfile
import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import trade_log_summary as tls


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class RecentTradeLogTests(unittest.TestCase):
    def test_recent_trade_log_prefers_trade_reason_over_later_skip(self):
        decision_fields = [
            "run_ts",
            "env",
            "trade_date",
            "city",
            "series_ticker",
            "event_ticker",
            "pred_tmax_f",
            "spread_f",
            "confidence_score",
            "decision",
            "reason",
        ]
        trade_fields = [
            "run_ts",
            "env",
            "trade_date",
            "city",
            "series_ticker",
            "event_ticker",
            "market_ticker",
            "market_subtitle",
            "pred_tmax_f",
            "side",
            "count",
            "yes_price",
            "no_price",
            "send_orders",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            decisions_path = Path(tmp) / "decisions.csv"
            trades_path = Path(tmp) / "trades.csv"

            _write_csv(
                decisions_path,
                decision_fields,
                [
                    {
                        "run_ts": "2026-02-28T13:00:00-05:00",
                        "env": "prod",
                        "trade_date": "2026-02-28",
                        "city": "il",
                        "series_ticker": "KXHIGHCHI",
                        "event_ticker": "KXHIGHCHI-26FEB28",
                        "pred_tmax_f": "45.2",
                        "spread_f": "0.3",
                        "confidence_score": "0.81",
                        "decision": "trade",
                        "reason": "intraday_soft;mode=forecast_bucket;ask=11",
                    },
                    {
                        "run_ts": "2026-03-01T13:00:00-05:00",
                        "env": "prod",
                        "trade_date": "2026-03-01",
                        "city": "ny",
                        "series_ticker": "KXHIGHNY",
                        "event_ticker": "KXHIGHNY-26MAR01",
                        "pred_tmax_f": "39.8",
                        "spread_f": "1.0",
                        "confidence_score": "0.70",
                        "decision": "skip",
                        "reason": "count_zero_after_caps;yes_ask=89;budget_cap=0",
                    },
                    {
                        "run_ts": "2026-03-02T14:00:00-05:00",
                        "env": "prod",
                        "trade_date": "2026-03-02",
                        "city": "tx",
                        "series_ticker": "KXHIGHAUS",
                        "event_ticker": "KXHIGHAUS-26MAR02",
                        "pred_tmax_f": "82.7",
                        "spread_f": "0.0",
                        "confidence_score": "0.76",
                        "decision": "trade",
                        "reason": "intraday_ok;mode=forecast_bucket;p=0.44;ask=12",
                    },
                    {
                        "run_ts": "2026-03-02T14:05:00-05:00",
                        "env": "prod",
                        "trade_date": "2026-03-02",
                        "city": "tx",
                        "series_ticker": "KXHIGHAUS",
                        "event_ticker": "KXHIGHAUS-26MAR02",
                        "pred_tmax_f": "",
                        "spread_f": "",
                        "confidence_score": "",
                        "decision": "skip",
                        "reason": "already_traded_live",
                    },
                ],
            )

            _write_csv(
                trades_path,
                trade_fields,
                [
                    {
                        "run_ts": "2026-02-28T13:00:01-05:00",
                        "env": "prod",
                        "trade_date": "2026-02-28",
                        "city": "il",
                        "series_ticker": "KXHIGHCHI",
                        "event_ticker": "KXHIGHCHI-26FEB28",
                        "market_ticker": "KXHIGHCHI-26FEB28-B45.5",
                        "market_subtitle": "45° to 46°",
                        "pred_tmax_f": "45.2",
                        "side": "yes",
                        "count": "3",
                        "yes_price": "11",
                        "no_price": "89",
                        "send_orders": "False",
                    },
                    {
                        "run_ts": "2026-03-02T14:00:02-05:00",
                        "env": "prod",
                        "trade_date": "2026-03-02",
                        "city": "tx",
                        "series_ticker": "KXHIGHAUS",
                        "event_ticker": "KXHIGHAUS-26MAR02",
                        "market_ticker": "KXHIGHAUS-26MAR02-B82.5",
                        "market_subtitle": "82° to 83°",
                        "pred_tmax_f": "82.7",
                        "side": "yes",
                        "count": "4",
                        "yes_price": "12",
                        "no_price": "88",
                        "send_orders": "True",
                    },
                ],
            )

            payload = tls.build_recent_trade_log(
                decisions_path=decisions_path,
                trades_path=trades_path,
                city_order=["ny", "il", "tx", "fl"],
                days=3,
                env="prod",
            )

        self.assertEqual(
            [group["trade_date"] for group in payload["groups"]],
            ["2026-03-02", "2026-03-01", "2026-02-28"],
        )

        latest = payload["groups"][0]["items"][0]
        self.assertEqual(latest["city"], "tx")
        self.assertEqual(latest["status"], "executed")
        self.assertEqual(latest["decision"], "trade")
        self.assertEqual(latest["reason"], "intraday_ok;mode=forecast_bucket;p=0.44;ask=12")
        self.assertEqual(latest["attempt_count"], 2)
        self.assertEqual(latest["trade"]["count"], 4)

        skipped = payload["groups"][1]["items"][0]
        self.assertEqual(skipped["status"], "skipped")
        self.assertIn("count_zero_after_caps", skipped["reason"])

        planned = payload["groups"][2]["items"][0]
        self.assertEqual(planned["status"], "planned")
        self.assertFalse(planned["trade"]["send_orders"])


if __name__ == "__main__":
    unittest.main()
