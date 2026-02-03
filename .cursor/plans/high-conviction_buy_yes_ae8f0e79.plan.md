---
name: High-conviction BUY YES
overview: Target BUY YES when the best predictor says temp in bracket and the warming trend is a "good time" to bet (no consensus). Add confluence detection and notifications; keep existing at-risk as the separate "exceed bracket → BUY NO" signal.
todos: []
isProject: false
---

# High-conviction BUY YES (confluence) — corrected plan

## Intent (critical)

- **We are not targeting NO for the alignment signal.** We don't care what consensus says. Find the **best predictor** (lowest MAE) for the city → what does it say? Is the **warming trend** a "good time" to bet (reliable trend bucket)? If both line up so temp is expected to land **inside** a bracket → **BUY YES** on that bracket.
- The **existing** at-risk logic (projected high **exceeds** bracket → bracket breaks → NO wins) remains as a **separate** signal: that is correctly "BUY NO" and stays as-is.
- The **new** signal is: best predictor says temp in bracket + trend says it's a good time to bet → **BUY YES**.

---

## Two distinct signals


| Signal                 | Condition                                                                                                                                   | Action                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **At-risk (exceed)**   | Projected high > bracket; warming; edge on NO                                                                                               | BUY NO (current implementation) |
| **Confluence (align)** | Best predictor forecasts **in** bracket; warming trend is a "good time" (reliable bucket for that city); optionally YES attractively priced | **BUY YES** (to add)            |


---

## Backend (`scripts/web_dashboard_api.py`)

1. **Source name → prediction key**
  Add `SOURCE_NAME_TO_PRED_KEY` (e.g. `"visual-crossing"` → `tmax_visual_crossing`) so we can read the best source’s forecast.
2. **MAE and trend-bucket data**
  - `_load_mae_by_city_source()` from `source_performance.csv` → per-city best source (lowest MAE).  
  - `_load_mae_by_city_trend_bucket()` from `observations_history.csv` + actuals, using `_trend_bucket()` to categorize; use this to define “reliable” (e.g. MAE ≤ median for that city’s buckets).
3. **Confluence (BUY YES) detection**
  In or alongside `get_at_risk_brackets()`:
  - For each city, find the **best predictor** (lowest MAE from `_load_mae_by_city_source()`). Consensus is **not** used. Consider brackets we care about (e.g. around current temp and projected high).
  - For each bracket, check:  
    - Best predictor's forecast is **in** `[bucket_base, bracket_high]`.  
    - Current warming trend (`trend_1h` or similar) is in a **"good time"** bucket for that city (reliable: e.g. MAE ≤ median for that city's trend buckets from `_load_mae_by_city_trend_bucket()`).  
    - Optionally: YES side has acceptable value (yes_ask present, etc.).
  - If both conditions hold → high-conviction BUY YES. Attach `confluence: true`, `best_mae_source: "<source_name>"`.
4. **API response**
  Either add a `confluence_yes` list (BUY YES opportunities) or add `confluence` + `best_mae_source` (and action hint) to existing items. Frontend will use this to show “High-conviction BUY YES” only for confluence items.

---

## Frontend (`scripts/dashboard_web.html`)

1. **High-conviction BUY YES notification**
  - Add `showConfluenceNotification(item)` used **only** when `item.confluence` (or item is in `confluence_yes` list).  
  - Title: e.g. `"Weather Trader — High-conviction BUY YES"`.  
  - Body: e.g. city, bracket, best source name, and “Conditions align for temp in bracket — consider BUY YES on &lt;ticker&gt;”.  
  - Do **not** use this for plain at-risk (exceed) items; those keep the existing “BUY NO” notification.
2. **Dashboard UI**
  - For items with `confluence === true` (or in `confluence_yes`): show a clear **“High-conviction BUY YES”** badge and copy (e.g. “Consider BUY YES when priced well”).  
  - Keep existing at-risk card text for exceed-bracket items: “BUY NO if you agree temp will exceed …”.
3. **Rendering logic**
  - If an item is confluence → call `showConfluenceNotification(item)` and render YES-focused copy and badge.  
  - If an item is at-risk only (exceed) → keep current `showAtRiskBracketNotification(item)` and “BUY NO” copy.

---

## Documentation

- `**documentation/dashboard.md**`  
Replace “at-risk brackets (BUY NO suggestions)” with wording that includes both signals, e.g.:  
“at-risk brackets (warm trend may exceed bracket → BUY NO) and high-conviction BUY YES when conditions align (best predictor in bracket + warming trend is a good time to bet).”

---

## Edge cases

- Missing MAE or best source: skip confluence for that city/bracket.  
- No reliable trend bucket for current trend: skip confluence.  
- Define “reliable” as e.g. MAE ≤ median for that city’s trend buckets (or document threshold in code comment).

---

## Summary

- **Confluence = best predictor says temp in bracket + warming trend is a good time to bet → BUY YES** (new). Consensus is not used.  
- **At-risk (exceed) = bracket likely broken → BUY NO** (existing).  
- Plan, backend, frontend, and docs must all say **BUY YES** for the confluence path and keep BUY NO only for the exceed-bracket path.

