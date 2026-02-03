# Environment variables

Copy `.env.example` to `.env` and fill in the values you need. Minimum for trading: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_ENV`. For full forecasts and calibration: add `VISUAL_CROSSING_API_KEY` and `NWS_USER_AGENT`; other provider keys are optional.

> **Important:** `.env` and private-key files are gitignored. Never commit secrets. API keys can also end up in SQLite cache files (`Data/*.sqlite`, `.cache.sqlite`); these are in `.gitignore`. If keys were ever committed, rotate/revoke and see [SECURITY.md](../SECURITY.md) for purging history and deleting cache files.

| Variable | Required | Description |
|----------|----------|-------------|
| **Kalshi** | | |
| `KALSHI_ENV` | Yes (for trading) | `demo` or `prod`. |
| `KALSHI_API_KEY_ID` | Yes (for trading) | Your Kalshi API key ID (not the private key). Also accepted: `KALSHI_API_KEY`, `KALSHI_KEY_ID`, `API_KEY_ID`, `API_KEY`. |
| `KALSHI_PRIVATE_KEY_PATH` | Yes (for trading) | Path to your RSA private key PEM file. Inside Docker use a path in the container (e.g. `/run/secrets/kalshi_key.pem`). Also accepted: `KALSHI_PRIVATE_KEY_FILE`, `PRIVATE_KEY_PATH`, `PRIVATE_KEY_FILE`. |
| `KALSHI_BASE_URL` | No | Override API base URL (rarely needed). |
| `KALSHI_SERIES_NY`, `KALSHI_SERIES_IL`, `KALSHI_SERIES_TX`, `KALSHI_SERIES_FL` | No | Override high-temp series tickers (defaults: `KXHIGHNY`, `KXHIGHCHI`, `KXHIGHAUS`, `KXHIGHMIA`). |
| **Weather / forecasts** | | |
| `NWS_USER_AGENT` | Recommended | User-Agent for api.weather.gov; include contact info (e.g. `weather-trader (contact: you@example.com)`). |
| `VISUAL_CROSSING_API_KEY` | Recommended | Visual Crossing API key; used for historical and forecast data. |
| `TOMORROW` | No | Tomorrow.io API key. |
| `WEATHERAPI` | No | WeatherAPI.com key. |
| `GOOGLE` or `GOOGLE_WEATHER_API_KEY` | No | Google Weather API key. |
| `OPENWEATHERMAP_API_KEY` | No | OpenWeatherMap key. Set `DISABLE_OPENWEATHERMAP=1` to skip. |
| `PIRATE_WEATHER_API_KEY` | No | Pirate Weather key (also `PIRATE_WEATER_API_KEY`). |
| **Postgres** | | |
| `PGDATABASE_URL` or `DATABASE_URL` | No | Postgres connection URL (e.g. `postgresql://user:pass@host:5432/dbname`). |
| `ENABLE_PG_WRITE` | No | Set to `1`/`true`/`yes`/`y` to mirror writes to Postgres. Default: off. |
| `ENABLE_PG_READ` | No | Set to `true`/`1`/`yes`/`y` to read from Postgres (with CSV fallback). Default: off. |
| **Other** | | |
| `TZ` | No | Timezone for cron and dashboard (e.g. `America/New_York`). Default: `America/New_York`. |
| `WT_ENV` | No | Used by run_trade.sh; `demo` or `prod`. Default in script: `demo`; override in docker-compose. |
| `WT_SEND_ORDERS` | No | Set to `true` to place orders; default is dry-run. |
| `WT_DAILY_BUDGET` | No | Daily spend cap in dollars (e.g. `50`). |

**Note:** A Visual Crossing API key may appear in examples — it is not our key and may be invalid. Set your own key in `.env` and never commit secrets.
