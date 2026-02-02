# Forecast API Limits and Schedule

We fetch forecasts for **2 dates** (today + tomorrow) × **4 cities** × **8 sources** per intraday pulse run.

- **Calls per run per source**: 2 × 4 = **8** (NWS has no key; others are paid/free-tier).
- **Runs per day**: Determined by the tightest free-tier limit so we stay under all limits.

## Free-tier limits (per source)

| Source            | Limit              | At 41 runs/day (41×8 calls) | Safe? |
|-------------------|--------------------|-----------------------------|-------|
| Google Weather    | 10,000/month      | 41×8×30 = **9,840/month**   | Yes (binding) |
| Tomorrow.io       | 25/hour, 500/day  | 328/day; peak ~2–3 runs/hr | Yes   |
| Open-Meteo        | 10,000/day        | 328/day                     | Yes   |
| Visual Crossing   | 1,000 records/day | 328/day                     | Yes   |
| WeatherAPI        | 1,000,000/month   | ~9,840/month                | Yes   |
| **OpenWeatherMap**| **1,000/day**     | **328/day**                 | **Yes** |
| Pirate Weather    | 10,000/month      | 9,840/month        | Yes   |
| NWS (weather.gov) | No key; fair use  | N/A                         | Yes   |

**Binding constraint**: **Google at 41 runs/day** (10,000 ÷ 30 ÷ 8 ≈ 41). Tomorrow.io and OpenWeatherMap are well under (500/day and 1,000/day).

### OpenWeatherMap (1,000/day)

- Free tier: **1,000 API calls per day**.
- At 41 runs/day: 41 × 8 = **328 calls/day** — under 1,000. Safe.

### Pirate Weather (10,000/month)

- Free tier: **10,000 calls/month** (same as Google). At 41 runs/day: 41 × 8 × 30 = **9,840/month** — under limit.

## Schedule

- **Previous**: Intraday pulse every **2 hours** → 12 runs/day (+ 1 from run_trade) → ~13 runs/day.
- **Current**: **41 runs/day** — run at :00 every hour (24/day) and at :30 for hours 0–16 (17/day) = **41 runs/day**.

**Verified at 41 runs/day** (no exhaustion):

- Google: 9,840/month (under 10,000).
- Tomorrow.io: 328/day (under 500); hourly peak under 25 with cache.
- OpenWeatherMap: 328/day (under 1,000).
- Visual Crossing, Open-Meteo, WeatherAPI: well under.

Pirate Weather: 9,840/month (under 10,000).

Caches (1h) reduce actual calls when multiple runs fall in the same hour; the schedule assumes worst case (every run uncached) for limits above.
