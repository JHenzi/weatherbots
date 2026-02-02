-- Core relational schema for weather-trader Postgres migration.
-- This file is applied by scripts/init_postgres.py.

-- Cities: shared dimension for all city-scoped tables.
CREATE TABLE IF NOT EXISTS cities (
    id          SERIAL PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE, -- ny, il, tx, fl
    name        TEXT,
    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION,
    metadata    JSONB
);

-- Forecast providers / sources.
CREATE TABLE IF NOT EXISTS sources (
    id       SERIAL PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE, -- open-meteo, tomorrow, etc.
    kind     TEXT,
    metadata JSONB
);

-- Point-in-time consensus predictions (from predictions_history.csv).
CREATE TABLE IF NOT EXISTS predictions (
    id                     BIGSERIAL PRIMARY KEY,
    city_id                INTEGER NOT NULL REFERENCES cities(id),
    trade_date             DATE    NOT NULL,
    run_ts                 TIMESTAMPTZ NOT NULL,
    env                    TEXT,

    tmax_predicted         DOUBLE PRECISION,
    tmax_lstm              DOUBLE PRECISION,
    tmax_forecast          DOUBLE PRECISION,
    spread_f               DOUBLE PRECISION,
    confidence_score       DOUBLE PRECISION,
    conviction_score       DOUBLE PRECISION,
    forecast_sources       TEXT,

    -- Per-provider values, e.g. {\"open-meteo\": 72.3, ...}
    provider_values        JSONB,

    -- Raw text fields (for debugging / traceability).
    sources_used           TEXT,
    weights_used           TEXT,

    prediction_mode        TEXT,               -- forecast / blend / lstm
    blend_forecast_weight  DOUBLE PRECISION,
    refresh_history        BOOLEAN,
    retrain_lstm           BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_predictions_city_date
    ON predictions (city_id, trade_date);

-- Intraday snapshots (intraday_forecasts.csv).
CREATE TABLE IF NOT EXISTS intraday_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    city_id         INTEGER NOT NULL REFERENCES cities(id),
    trade_date      DATE    NOT NULL,
    snapshot_ts     TIMESTAMPTZ NOT NULL,

    mean_forecast   DOUBLE PRECISION,
    current_sigma   DOUBLE PRECISION,

    -- Per-provider instantaneous forecasts for this pulse.
    provider_values JSONB,

    sources_used    TEXT,
    weights_used    TEXT
);

CREATE INDEX IF NOT EXISTS idx_intraday_city_date_ts
    ON intraday_snapshots (city_id, trade_date, snapshot_ts);

-- Trade intents / executions (trades_history.csv).
CREATE TABLE IF NOT EXISTS trades (
    id               BIGSERIAL PRIMARY KEY,
    run_ts           TIMESTAMPTZ NOT NULL,
    env              TEXT        NOT NULL,
    trade_date       DATE        NOT NULL,
    city_id          INTEGER     REFERENCES cities(id),
    series_ticker    TEXT        NOT NULL,
    event_ticker     TEXT        NOT NULL,
    market_ticker    TEXT        NOT NULL,
    market_subtitle  TEXT,
    pred_tmax_f      DOUBLE PRECISION,
    side             TEXT,
    count            INTEGER,
    yes_price        INTEGER,
    no_price         INTEGER,
    send_orders      BOOLEAN     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_env_date_city
    ON trades (env, trade_date, city_id);

-- Trade decisions (decisions_history.csv).
CREATE TABLE IF NOT EXISTS decisions (
    id               BIGSERIAL PRIMARY KEY,
    run_ts           TIMESTAMPTZ NOT NULL,
    env              TEXT        NOT NULL,
    trade_date       DATE        NOT NULL,
    city_id          INTEGER     REFERENCES cities(id),
    series_ticker    TEXT        NOT NULL,
    event_ticker     TEXT        NOT NULL,
    pred_tmax_f      DOUBLE PRECISION,
    spread_f         DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    decision         TEXT        NOT NULL,
    reason           TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_env_date_city
    ON decisions (env, trade_date, city_id);

-- Per-trade evaluation context and realized outcomes (eval_history.csv).
CREATE TABLE IF NOT EXISTS eval_events (
    id                    BIGSERIAL PRIMARY KEY,
    run_ts                TIMESTAMPTZ NOT NULL,
    env                   TEXT        NOT NULL,
    trade_date            DATE        NOT NULL,
    city_id               INTEGER     REFERENCES cities(id),
    series_ticker         TEXT        NOT NULL,
    event_ticker          TEXT        NOT NULL,

    decision              TEXT,
    reason                TEXT,

    mu_tmax_f             DOUBLE PRECISION,
    sigma_f               DOUBLE PRECISION,
    spread_f              DOUBLE PRECISION,
    confidence_score      DOUBLE PRECISION,

    tmax_open_meteo       DOUBLE PRECISION,
    tmax_visual_crossing  DOUBLE PRECISION,
    tmax_tomorrow         DOUBLE PRECISION,
    tmax_weatherapi       DOUBLE PRECISION,
    tmax_lstm             DOUBLE PRECISION,

    sources_used          TEXT,
    weights_used          TEXT,

    chosen_market_ticker  TEXT,
    chosen_market_subtitle TEXT,
    bucket_lo             DOUBLE PRECISION,
    bucket_hi             DOUBLE PRECISION,

    model_prob_yes        DOUBLE PRECISION,
    yes_ask               INTEGER,
    yes_bid               INTEGER,
    yes_spread            INTEGER,
    ask_qty               INTEGER,
    market_prob_yes       DOUBLE PRECISION,
    edge_prob             DOUBLE PRECISION,
    ev_cents              DOUBLE PRECISION,
    count                 INTEGER,
    send_orders           BOOLEAN,

    -- Settlement/enrichment columns from settle_eval.py (optional).
    settlement_tmax_f     DOUBLE PRECISION,
    settlement_source_url TEXT,
    bucket_hit            BOOLEAN,
    realized_pnl_cents    INTEGER,
    realized_pnl_dollars  DOUBLE PRECISION,
    settled_at            TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_eval_events_env_date_city
    ON eval_events (env, trade_date, city_id);

-- Daily rollups from eval_history.csv (daily_metrics.csv).
CREATE TABLE IF NOT EXISTS eval_metrics (
    id          BIGSERIAL PRIMARY KEY,
    run_ts      TIMESTAMPTZ NOT NULL,
    trade_date  DATE        NOT NULL,
    city_id     INTEGER     REFERENCES cities(id),
    metric_type TEXT        NOT NULL,
    source_name TEXT        NOT NULL,
    value       DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_metrics_date_city_type
    ON eval_metrics (trade_date, city_id, metric_type, source_name);

-- Per-source absolute error log (source_performance.csv).
CREATE TABLE IF NOT EXISTS source_performance (
    id             BIGSERIAL PRIMARY KEY,
    date           DATE        NOT NULL,
    city_id        INTEGER     REFERENCES cities(id),
    source_name    TEXT        NOT NULL,
    predicted_tmax DOUBLE PRECISION,
    actual_tmax    DOUBLE PRECISION,
    absolute_error DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_source_performance_date_city_source
    ON source_performance (date, city_id, source_name);

-- Learned weights history (weights_history.csv).
CREATE TABLE IF NOT EXISTS weights_history (
    id          BIGSERIAL PRIMARY KEY,
    run_ts      TIMESTAMPTZ NOT NULL,
    as_of       DATE        NOT NULL,
    city_id     INTEGER     REFERENCES cities(id),
    window_days INTEGER,
    weights     JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_weights_history_asof_city
    ON weights_history (as_of, city_id);

-- Observation snapshots (observations_history.csv): projected high, delta, time temp will max.
CREATE TABLE IF NOT EXISTS observations (
    id                   BIGSERIAL PRIMARY KEY,
    observed_at           TIMESTAMPTZ NOT NULL,
    city_id               INTEGER NOT NULL REFERENCES cities(id),
    stid                  TEXT NOT NULL,
    temp                  DOUBLE PRECISION NOT NULL,
    observed_high_today   DOUBLE PRECISION,
    projected_high        DOUBLE PRECISION,
    trend_10m             DOUBLE PRECISION,
    trend_30m             DOUBLE PRECISION,
    trend_1h              DOUBLE PRECISION,
    acceleration          DOUBLE PRECISION,
    time_temp_will_max    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_observations_city_observed
    ON observations (city_id, observed_at);

