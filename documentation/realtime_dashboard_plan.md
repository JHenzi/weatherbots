# Real-Time Web Dashboard Plan

This plan outlines the creation of a modern, web-based operational dashboard for **"near-automated" trading**. It serves as a command center where the bot calculates the math of every trend shift and presents a clear "Hedge or Hold" decision to the human, with the ability to transition into fully automated execution.

## 1. Tech Stack
- **Backend**: Python (FastAPI) to serve data and handle the real-time fetcher.
- **Frontend**: Tailwind CSS + Chart.js (served via simple HTML/JS or React).
- **Data Delivery**: JSON API with 30-second polling for real-time updates.
- **Deployment**: Localhost or a Dockerized sidecar service.

## 2. Dashboard Layout & Features

### A. Live Station Overviews (Top Bar)
- **Status Cards**: One card per city (NYC, CHI, AUS, MIA).
- **KPIs**: Current Temp, "Feels Like", and a 1-hour Sparkline.
- **Trending Indicators**: 
    - Dynamic arrows (↑, ↓) colored by intensity.
    - Textual status: "Warming rapidly", "Plateauing near high", "Cooling".
- **Visuals**: Large, color-coded temperature display.

### B. Interactive Trajectory Charts (Main View)
- **Live Line Chart**: Real-time temperature vs. your "YES" bucket boundaries.
- **Progress Gauge**: A "Sun Tracker" bar showing current time relative to the **Daily Peak Window** (e.g., "75% to Daily High").
- **Overlays**: Vertical markers for "Run Time" and "Settlement Time".
- **Trend Indicators**: Rolling 10m and 30m linear regression lines projected forward.

### C. Position & Hedge Console
- **Active Bets**: Table showing open tickers, buy-in price, and current "Live Value".
- **Trading Control Center**:
    - **Native Order Buttons**: Direct "Buy YES" / "Buy NO" / "Sell Position" buttons for each active market, allowing for manual intervention.
    - **Quick-Fill Presets**: Buttons for common share quantities (e.g., "Buy 10", "Buy 50", "Max allocation").
    - **Market Depth**: Mini orderbook view showing the best bid/ask directly inside the dashboard.
- **Notification Center**: 
    - Desktop-style "Toasts" for trend alerts.
    - Sound alerts for "High Risk" hedge signals.
- **Hedge Advisor**: 
    - A "Risk Meter" showing the probability of the current trend breaking the bucket.
    - **"Why Hedge?" Tooltip**: Displays the math: `(P_fail * Payout) - Hedge_Cost`.
    - **Decision Button**: "Near-Automated" mode presents an "Approve Hedge" button; "Full-Auto" mode executes and shows the "Hedged at $X.XX" status.

## 3. Data Flow

1. **`observation_monitor.py`**: Continues to run as a background service, fetching Synoptic data every 2 minutes and updating Postgres/JSON.
2. **API Endpoint**: The web server provides `/api/observations` and `/api/positions`.
3. **Frontend**: The browser-based dashboard polls the API and re-renders components.

## 4. Implementation Steps

1. **Phase 1: Web Server & API**
   - Create `scripts/web_dashboard.py` using FastAPI.
   - Implement JSON endpoints that aggregate `trades_history.csv` and `observations_latest.json`.

2. **Phase 2: Modern UI Shell**
   - Create a clean, dark-mode dashboard template using Tailwind CSS.
   - Integrate Chart.js for the live trajectory visualization.

3. **Phase 3: Risk & Projection Logic**
   - Implement the "Trend Projection" algorithm in the backend (using NumPy for regression).
   - Display "Projected Landing Temp" on the charts.

4. **Phase 4: Operational Integration**
   - Add a "Launch Dashboard" command to the main bot.
   - Configure a Docker port (e.g., `8080`) in `docker-compose.yml` for easy access.
