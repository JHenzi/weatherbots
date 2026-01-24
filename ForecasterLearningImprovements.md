## Forecaster learning: long-term improvements

This bot can “learn which forecasters to trust” by turning each provider into a continuously-scored expert, then using those scores to (a) weight the consensus mean, (b) estimate uncertainty (\(\sigma\)), and (c) decide when *not* to trade.

You already have the core learning loop implemented:
- Nightly truth ingestion via `truth_engine.py` (NWS CLI).
- Per-source error log: `Data/source_performance.csv`.
- Per-city weights updated nightly: `calibrate_sources.py` writes `Data/weights.json`.
- Trading uses `weights.json` in `run_daily.py` postprocess, and uses spread/confidence guardrails.

This doc describes the next upgrades that make the learning more robust, faster to adapt, and safer to deploy.

---

### 1) Make the scoring better than plain MAE

MAE is a good start but can be improved in ways that matter for trading.

- **Use robust loss for outliers**
  - **Problem**: A few bad days can whipsaw weights.
  - **Improvement**: Use Huber loss or winsorize daily errors (cap at e.g. 10°F) before computing MAE/RMSE.
  - **Implementation**: in `calibrate_sources.py`, cap `absolute_error` when aggregating windows.

- **Score by “bucket hit” relevance**
  - **Problem**: A 2°F error may be irrelevant if it doesn’t change the Kalshi bucket, and critical if it crosses a boundary.
  - **Improvement**: Track “boundary-crossing error”:
    - distance from predicted mean to nearest bucket edge
    - whether the realized temperature lands in the chosen bucket
  - **Implementation**: you already store bucket bounds and settlement in `Data/eval_history.csv`. Add a nightly metric: “bucket boundary loss”.

- **Probabilistic scoring (recommended)**
  - **Problem**: Trading depends on probabilities, not just point forecasts.
  - **Improvement**: Convert each provider into a distribution and evaluate with:
    - **Log loss** on bucket probabilities, or
    - **Brier score** on “bucket hit” events.
  - **Implementation**: store per-provider implied distribution parameters (or at least an estimated provider \(\sigma\)); compute per-bucket probs and score them once settlement is known.

---

### 2) Make weights adapt to regime changes

Fixed rolling windows (e.g. 14 days) adapt, but not optimally.

- **Exponentially decayed weights**
  - **Idea**: recent days matter more than old days.
  - **Implementation**: compute an exponentially-weighted MAE:
    - weight day \(k\) days ago by \(\lambda^k\) (e.g. \(\lambda=0.9\))
  - **Benefit**: faster adaptation after fronts, seasonal transitions, API changes.

- **Seasonal / temperature-regime specialization**
  - **Idea**: providers may be better in certain regimes (winter vs summer; stable vs volatile).
  - **Implementation**:
    - keep separate weight tables by “season bin” (month or day-of-year quartile), or
    - by temperature bin (e.g. <32, 32–60, 60–80, >80).

- **Lead-time specialization**
  - **Problem**: accuracy depends on horizon; “tomorrow” differs from “3 days out”.
  - **Implementation**: store **lead hours** for each provider call; maintain weights by lead-time buckets (0–24h, 24–48h, …).

---

### 3) Separate “mean trust” from “uncertainty trust”

Right now the system uses provider disagreement (`spread_f`) and historical MAE to set \(\sigma\). That can be improved:

- **Per-provider residual variance**
  - Track not just MAE but also residual stddev per provider/city.
  - Use that to estimate the consensus uncertainty via an inverse-variance ensemble:
    - \(w_i \propto 1/\sigma_i^2\)

- **Ensemble uncertainty calibration**
  - Calibrate the mapping from disagreement → \(\sigma\) using past data:
    - Fit a small regression: \(\sigma \approx a + b \cdot spread\)
  - This makes “confidence_score” and trade gating more stable.

---

### 4) Handle missingness and provider failures explicitly

Missing providers shouldn’t accidentally cause overconfidence.

- **Reliability score**
  - Track availability (% days present) per provider/city.
  - Penalize weights for flaky providers (or require a minimum availability before a provider is eligible).

- **Fallback policy**
  - If only 1–2 sources are present, either:
    - raise \(\sigma\) floor,
    - require higher EV,
    - or skip entirely.

---

### 5) Learn “which model family” to trust (LSTM vs forecasts) as a bandit

Instead of always excluding LSTM from voting (or always blending), let the system learn:

- **Contextual bandit over prediction modes**
  - Actions: `forecast`, `blend`, `lstm` (or “include LSTM in vote yes/no”).
  - Context: city, season, spread, temporal jitter, recent model performance.
  - Reward: realized PnL (or bucket hit), regularized by risk.

This is much simpler than full RL and fits your logging setup.

---

### 6) Make improvements safely (so learning upgrades don’t break trading)

- **Offline backtests**
  - Re-run the “decision engine” on historical `eval_history.csv` with different weight schemes and compare:
    - MAE/RMSE
    - bucket hit rate
    - simulated EV and PnL (using historical quotes if available; otherwise proxy)

- **Shadow mode**
  - Compute “new weights” daily but don’t trade them; log them to `Data/weights_history.csv` with a `variant` tag.

- **Canary rollout**
  - Enable new weights for one city first, or cap risk to tiny size until stable.

---

### 7) Concrete next changes (high ROI)

- **A. Add exponentially-decayed weighting to `calibrate_sources.py`**
  - Keep the existing MAE\(^-2\) scheme, but compute MAE with exponential weights.

- **B. Add per-provider reliability tracking**
  - New file: `Data/provider_uptime.csv` or a new metric type in `Data/daily_metrics.csv`.

- **C. Add probabilistic scoring**
  - Store per-provider (or per-consensus) bucket probabilities and score via Brier/log loss after settlement.

- **D. Add “mode selection bandit” (optional)**
  - Start with a simple epsilon-greedy policy and log outcomes.

