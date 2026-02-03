# Security and secrets

## If API keys or secrets were ever committed

**Repository history is permanent.** If `.env`, `gooony.txt`, or any file containing API keys, tokens, or private keys was ever committed (even in the past), those secrets exist in git history and should be treated as **compromised**.

### 1. Rotate or revoke every exposed secret

Assume any of the following that were ever in the repo or in a committed file are compromised. Rotate or revoke them **immediately**:

- **Kalshi**: Revoke the exposed API key in the Kalshi dashboard and create a new one. Generate a new RSA key pair and upload the new public key; stop using the old private key.
- **Visual Crossing**: Regenerate or revoke the API key in your Visual Crossing account.
- **Tomorrow.io, WeatherAPI, OpenWeatherMap, Pirate Weather, Google Weather**: Revoke or regenerate each key in the respective provider dashboard.
- **Synoptic/mesoToken**: If `SYNOPTIC_TOKEN` was committed, treat it as public; refresh it (e.g. via `scripts/refresh_synoptic_token.py`) and do not commit the new value.
- **Postgres**: If `PGDATABASE_URL` or `POSTGRES_PASSWORD` was committed, change the database password and update the URL in a non-committed `.env`.

Use only **new** keys and store them **only** in `.env` (or a secrets manager), and ensure `.env` is never committed.

### 2. Never commit secrets

- **`.env`** is in `.gitignore`. Do not remove it. Do not force-add `.env`. Never commit files that contain API keys, passwords, or private keys.
- **`gooony.txt`** and **`*.pem`**, **`*.key`** are gitignored; use them only for the Kalshi private key on the host and do not commit them.
- **`*.sqlite` and `.cache`** are gitignored. The project uses **requests_cache** (e.g. in `intraday_pulse.py`, `daily_prediction.py`, `hourly_pulse.py`, `truth_engine.py`) with the default **SQLite** backend. Those cache databases store the **full HTTP request**, including URL and query params — so **API keys that are passed in the URL or params are stored inside the SQLite files** (e.g. `Data/visualcrossing_cache.sqlite`, `Data/tomorrow_cache.sqlite`, `Data/weatherapi_cache.sqlite`, `Data/openweathermap_cache.sqlite`, `Data/pirateweather_cache.sqlite`, `Data/google_weather_cache.sqlite`, `Data/.cache.sqlite`). If any of these were ever committed, treat the keys as compromised, rotate them, and **delete the cache files** when rotating (they will be recreated without the old key). Do not commit `*.sqlite` or remove them from `.gitignore`.
- This codebase is written to use **environment variables only** (e.g. `os.getenv("VISUAL_CROSSING_API_KEY")`). There are no API keys or tokens hardcoded in source code. Keep it that way.

### 3. Delete request cache SQLite files when rotating keys

When you rotate API keys, **delete all requests_cache SQLite files** so they are not re-used with old keys and do not contain the new key in plain form. For example:

```bash
rm -f Data/*.sqlite Data/.cache.sqlite .cache.sqlite
```

After that, the next run will recreate caches using the new keys from `.env`. The cache files are now in `.gitignore` so they will not be committed.

### 4. Purging secrets from git history (optional but recommended if keys were committed)

If you need to remove a file (or its contents) from **all** git history so that no clone or fetch can ever recover it:

- **Cache SQLite files**: If any `*.sqlite` or `.cache.sqlite` were ever committed, remove them from history too (they may contain API keys). Example with BFG: `bfg --delete-files '*.sqlite'` (or delete specific paths). With git filter-repo: `git filter-repo --path-glob '*.sqlite' --invert-paths`.

1. **BFG Repo-Cleaner**  
   - Install [BFG](https://rtyley.github.io/bfg-repo-cleaner/).  
   - Replace secrets with placeholders:  
     `bfg --replace-text passwords.txt`  
     (where `passwords.txt` lists strings to replace with `***REMOVED***`).  
   - Or remove the file from history:  
     `bfg --delete-files .env`

2. **git filter-repo** (recommended by GitHub)  
   - Install [git-filter-repo](https://github.com/newren/git-filter-repo).  
   - To remove `.env` from entire history:  
     `git filter-repo --path .env --invert-paths`  
   - Then force-push (rewrites history; collaborators must re-clone):  
     `git push --force --all`

3. **After rewriting history**  
   - All users must re-clone or rebase onto the new history; old clones will still contain the removed data until they do.  
   - Rotate/revoke the exposed secrets regardless; purging history does not undo the fact that they were exposed.

### 5. Docker and Postgres

- **Postgres**: `docker-compose.yml` may reference a default password for local development. For any shared or production deployment, set `POSTGRES_PASSWORD` (and optionally `PGDATABASE_URL`) in `.env` so that no secret is stored in the repository. See the README for environment variables.
- **Kalshi key in Docker**: Mount the private key as a file (e.g. `/run/secrets/kalshi_key.pem`) and set `KALSHI_PRIVATE_KEY_PATH` to that path. Do not put the key contents in environment variables or in the image.

### 6. Reporting

If you discover a secret that was committed or a security issue, rotate the affected credentials immediately and, if appropriate, report it to the maintainers privately (e.g. via a secure channel rather than a public issue).
