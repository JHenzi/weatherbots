import csv
import tempfile
import unittest

import calibrate_sources as cs


class CalibrateSourcesTests(unittest.TestCase):
    def test_append_performance_rows_skips_existing_key(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
            path = f.name
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "date",
                    "city",
                    "source_name",
                    "predicted_tmax",
                    "actual_tmax",
                    "absolute_error",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "date": "2026-02-16",
                    "city": "il",
                    "source_name": "consensus",
                    "predicted_tmax": "65.0",
                    "actual_tmax": "65.0",
                    "absolute_error": "0.0",
                }
            )

        rows = [
            {
                "date": "2026-02-16",
                "city": "il",
                "source_name": "consensus",
                "predicted_tmax": "65.2",
                "actual_tmax": "65.0",
                "absolute_error": "0.2",
            },
            {
                "date": "2026-02-16",
                "city": "il",
                "source_name": "open-meteo",
                "predicted_tmax": "64.8",
                "actual_tmax": "65.0",
                "absolute_error": "0.2",
            },
        ]
        written, skipped = cs._append_performance_rows(path, rows)

        self.assertEqual(written, 1)
        self.assertEqual(skipped, 1)

        with open(path, "r", newline="") as f:
            out = list(csv.DictReader(f))
        self.assertEqual(len(out), 2)
        keys = {(r["date"], r["city"], r["source_name"]) for r in out}
        self.assertIn(("2026-02-16", "il", "consensus"), keys)
        self.assertIn(("2026-02-16", "il", "open-meteo"), keys)


if __name__ == "__main__":
    unittest.main()
