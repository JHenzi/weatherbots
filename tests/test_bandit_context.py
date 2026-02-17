import unittest

from bandit.context import normalize_condition_text, sky_label_from_cloud_cover, vote_provider_conditions


class BanditContextTests(unittest.TestCase):
    def test_normalize_condition_text(self):
        self.assertEqual(normalize_condition_text("Clear"), "clear")
        self.assertEqual(normalize_condition_text("Partly cloudy"), "partly_cloudy")
        self.assertEqual(normalize_condition_text("Rain, Overcast"), "rain")
        self.assertEqual(normalize_condition_text("Snow showers"), "snow")
        self.assertEqual(normalize_condition_text("Dense fog"), "fog")

    def test_sky_label(self):
        self.assertEqual(sky_label_from_cloud_cover(10.0), "sunny")
        self.assertEqual(sky_label_from_cloud_cover(50.0), "mixed")
        self.assertEqual(sky_label_from_cloud_cover(90.0), "cloudy")
        self.assertEqual(sky_label_from_cloud_cover(None, "clear"), "sunny")

    def test_vote_provider_conditions_weighted(self):
        payloads = {
            "visual-crossing": {
                "condition_text": "Overcast",
                "condition_icon": "cloudy",
                "cloud_cover": 92,
            },
            "openweathermap": {
                "condition_text": "Clear sky",
                "condition_icon": "01d",
                "cloud_cover": 5,
            },
        }
        weights = {"visual-crossing": 0.8, "openweathermap": 0.2}
        out = vote_provider_conditions(payloads, provider_weights=weights)
        self.assertEqual(out["condition_token"], "cloudy_overcast")
        self.assertEqual(out["sky_label"], "cloudy")
        self.assertGreater(out["mean_cloud_cover"], 60)


if __name__ == "__main__":
    unittest.main()
