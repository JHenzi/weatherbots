import datetime as dt
import html
import re
import time
from dataclasses import dataclass

import requests_cache


CITY_CLI = {
    "ny": {"site": "OKX", "issuedby": "NYC"},
    "il": {"site": "LOT", "issuedby": "MDW"},
    "tx": {"site": "EWX", "issuedby": "AUS"},
    "fl": {"site": "MFL", "issuedby": "MIA"},
}


@dataclass(frozen=True)
class CliTruth:
    city: str
    target_date: dt.date
    observed_max_f: int
    report_issued_ts: str
    version: int
    source_url: str


_SUMMARY_RE = re.compile(r"SUMMARY FOR ([A-Z]+) (\d{1,2}) (\d{4})", re.IGNORECASE)
_MAX_RE = re.compile(r"^MAXIMUM\s+(-?\d+)\b", re.IGNORECASE)
_PRE_RE = re.compile(r"<pre[^>]*>([\s\S]*?)</pre>", re.IGNORECASE)
_ISSUED_RE = re.compile(
    r"\b(\d{1,4}\s*(AM|PM)\s+[A-Z]{2,5}\s+\w{3}\s+\w{3}\s+\d{1,2}\s+\d{4})\b",
    re.IGNORECASE,
)


def _fetch_cli_text(
    *,
    site: str,
    issuedby: str,
    version: int,
    session: requests_cache.CachedSession,
    force_refresh: bool = False,
) -> str:
    url = "https://forecast.weather.gov/product.php"
    params = {
        "site": site,
        "issuedby": issuedby,
        "product": "CLI",
        "format": "txt",
        "version": str(version),
        "glossary": "0",
    }
    # NOTE: for "truth" we prefer correctness over caching. We keep a short cache for normal runs,
    # but when we're polling for newly-published reports we must force-refresh, otherwise we can
    # miss a report that was published within the cache TTL.
    req_kwargs = {"params": params, "timeout": 30, "headers": {"Cache-Control": "no-cache"}}
    if force_refresh:
        # Override session default TTL for this request.
        req_kwargs["expire_after"] = 0
    r = session.get(url, **req_kwargs)
    r.raise_for_status()
    # product.php returns HTML; the actual product text is inside a <pre> block.
    raw = r.text or ""
    m = _PRE_RE.search(raw)
    if not m:
        return raw
    # Unescape HTML entities and preserve newlines.
    return html.unescape(m.group(1))


def _parse_target_date(text: str) -> dt.date | None:
    m = _SUMMARY_RE.search(text)
    if not m:
        return None
    mon, day, year = m.group(1), int(m.group(2)), int(m.group(3))
    month = dt.datetime.strptime(mon[:3].title(), "%b").month
    return dt.date(year, month, day)


def _parse_yesterday_max(text: str) -> int | None:
    # We only trust the "TEMPERATURE (F) / YESTERDAY / MAXIMUM" block.
    lines = text.splitlines()
    in_temp = False
    in_yesterday = False
    for line in lines:
        s = line.strip()
        if s.upper().startswith("TEMPERATURE"):
            in_temp = True
            in_yesterday = False
            continue
        if in_temp and s.upper() == "YESTERDAY":
            in_yesterday = True
            continue
        if in_temp and in_yesterday:
            mm = _MAX_RE.match(s)
            if mm:
                return int(mm.group(1))
            # stop scanning if we moved to next section
            if s.upper().startswith("PRECIPITATION"):
                break
    return None


def _parse_issued_line(text: str) -> str:
    # Prefer a line that matches the typical CLI issued timestamp.
    for ln in text.splitlines():
        if _ISSUED_RE.search(ln or ""):
            return ln.strip()
    # Fallback: first line containing AM/PM.
    return next((ln.strip() for ln in text.splitlines() if (" AM " in ln or " PM " in ln)), "").strip()


def get_actual_tmax_from_nws_cli(
    city: str,
    target_date: dt.date,
    *,
    max_versions: int = 50,
    max_wait_seconds: int = 0,
    retry_every_seconds: int = 120,
) -> CliTruth:
    if city not in CITY_CLI:
        raise ValueError(f"Unknown city code: {city}")
    cfg = CITY_CLI[city]
    # Keep cache short; NWS may publish the target report minutes after we first check.
    session = requests_cache.CachedSession("Data/nws_cli_cache", expire_after=300)

    deadline = time.monotonic() + max(0, int(max_wait_seconds))
    attempt = 0
    last_seen: list[tuple[int, dt.date | None, str]] = []
    while True:
        attempt += 1
        force_refresh = attempt > 1  # first attempt can use cache; retries must refresh
        last_seen = []

        # Version=1 is "current", higher versions are older.
        for v in range(1, max_versions + 1):
            text = _fetch_cli_text(
                site=cfg["site"],
                issuedby=cfg["issuedby"],
                version=v,
                session=session,
                force_refresh=force_refresh,
            )
            rep_date = _parse_target_date(text)
            issued_line = _parse_issued_line(text)
            if rep_date is not None:
                last_seen.append((v, rep_date, issued_line))

            # The "SUMMARY FOR ..." date corresponds to the day being summarized (target_date).
            if rep_date != target_date:
                continue
            max_f = _parse_yesterday_max(text)
            if max_f is None:
                # We only accept reports that have a "YESTERDAY" temperature block (final day summary).
                continue
            source_url = (
                f"https://forecast.weather.gov/product.php?site={cfg['site']}&issuedby={cfg['issuedby']}"
                f"&product=CLI&format=txt&version={v}&glossary=0"
            )
            return CliTruth(
                city=city,
                target_date=target_date,
                observed_max_f=max_f,
                report_issued_ts=issued_line.strip(),
                version=v,
                source_url=source_url,
            )

        if time.monotonic() >= deadline:
            # Helpful diagnostics: what date is "current" right now?
            # Sort by newest date first, then lowest version number.
            newest = None
            if last_seen:
                newest = sorted(last_seen, key=lambda t: (t[1] or dt.date.min, -t[0]), reverse=True)[0]
            hint = ""
            if newest is not None:
                v, d, issued = newest
                hint = f" (latest_seen={d} at version={v}; issued='{issued}')"
            raise RuntimeError(
                f"Could not find CLI report for {city} date={target_date} within {max_versions} versions{hint}"
            )

        time.sleep(max(5, int(retry_every_seconds)))

