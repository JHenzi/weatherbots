import datetime as dt
import unittest

import truth_engine as te


class TruthEngineParseTests(unittest.TestCase):
    def test_parse_yesterday_max_plain_integer(self):
        text = """
SUMMARY FOR FEB 15 2026

TEMPERATURE (F)
 YESTERDAY
  MAXIMUM         60   3:53 PM
  MINIMUM         31   5:51 AM
"""
        self.assertEqual(te._parse_target_date(text), dt.date(2026, 2, 15))
        self.assertEqual(te._parse_yesterday_max(text), 60)

    def test_parse_yesterday_max_with_trailing_flag(self):
        text = """
SUMMARY FOR FEB 16 2026

TEMPERATURE (F)
 YESTERDAY
  MAXIMUM         65R  4:31 PM
  MINIMUM         35   1:52 AM
"""
        self.assertEqual(te._parse_target_date(text), dt.date(2026, 2, 16))
        self.assertEqual(te._parse_yesterday_max(text), 65)

    def test_parse_today_block_not_treated_as_yesterday_truth(self):
        text = """
SUMMARY FOR FEB 16 2026

TEMPERATURE (F)
 TODAY
  MAXIMUM         65R  3:59 PM
"""
        self.assertIsNone(te._parse_yesterday_max(text))


if __name__ == "__main__":
    unittest.main()
