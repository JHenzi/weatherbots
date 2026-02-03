# Kalshi market mapping + resolution

This project targets “daily high temperature” markets whose settlement source is the **National Weather Service (NWS) climatological report** for a specific station.

## Series tickers (demo + production)

We default to the `KX*` series because they exist in the Kalshi demo environment. Override via env: `KALSHI_SERIES_NY`, `KALSHI_SERIES_IL`, `KALSHI_SERIES_TX`, `KALSHI_SERIES_FL`.

| City | Code | Default series ticker | NWS station | NWS CLI link |
|------|------|----------------------|-------------|--------------|
| New York City | `ny` | `KXHIGHNY` | NYC (Central Park) | [OKX NYC CLI](https://forecast.weather.gov/product.php?site=OKX&product=CLI&issuedby=NYC) |
| Chicago | `il` | `KXHIGHCHI` | MDW (Midway) | [LOT MDW CLI](https://forecast.weather.gov/product.php?site=LOT&product=CLI&issuedby=MDW) |
| Austin | `tx` | `KXHIGHAUS` | AUS (Bergstrom) | [EWX AUS CLI](https://forecast.weather.gov/product.php?site=EWX&product=CLI&issuedby=AUS) |
| Miami | `fl` | `KXHIGHMIA` | MIA (Miami Intl) | [MFL MIA CLI](https://forecast.weather.gov/product.php?site=MFL&product=CLI&issuedby=MIA) |

## Contract certification / resolution

Kalshi publishes product certification and contract terms PDFs per series. These define the resolution procedure: which NWS report/station is authoritative, when the report is final for settlement, and how edge cases are handled.

Examples (NYC): Product certification and contract terms are linked from the Kalshi API (e.g. `contract_url`, `contract_terms_url`).

## Contract selection (how the bot chooses a market)

- Build **event ticker**: series + date suffix `YYMONDD` (e.g. `KXHIGHNY-26JAN23`).
- Fetch event: `GET /trade-api/v2/events/{event_ticker}?with_nested_markets=true`.
- Each event has markets whose `subtitle` encodes the temperature bucket (e.g. `71° to 72°`, `40° or above`).
- `kalshi_trader.py` parses those subtitles and chooses the bucket that contains the predicted temperature (or closest by EV).

## Authentication

Kalshi Trade API v2 requests are signed with:

- `KALSHI-ACCESS-KEY` (your key id)
- `KALSHI-ACCESS-TIMESTAMP` (ms)
- `KALSHI-ACCESS-SIGNATURE` = RSA-PSS signature of `timestamp + HTTP_METHOD + PATH_WITHOUT_QUERY`

Demo and production use **separate API keys**. Use `--env prod` (or `WT_ENV=prod`) with production keys; use `--env demo` only with demo keys.

## Dry-run first

By default the trader does **not** place orders; it prints what it *would* submit. It only places orders if you pass `--send-orders`. Test with `--env demo` and dry-run before enabling live trading.
