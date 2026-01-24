import datetime as dt
import html
import re
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


def _fetch_cli_text(*, site: str, issuedby: str, version: int, session: requests_cache.CachedSession) -> str:
    url = "https://forecast.weather.gov/product.php"
    params = {
        "site": site,
        "issuedby": issuedby,
        "product": "CLI",
        "format": "txt",
        "version": str(version),
        "glossary": "0",
    }
    r = session.get(url, params=params, timeout=30)
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


def get_actual_tmax_from_nws_cli(city: str, target_date: dt.date, *, max_versions: int = 50) -> CliTruth:
    if city not in CITY_CLI:
        raise ValueError(f"Unknown city code: {city}")
    cfg = CITY_CLI[city]
    session = requests_cache.CachedSession("Data/nws_cli_cache", expire_after=3600)
    # Version=1 is "current", higher versions are older.
    for v in range(1, max_versions + 1):
        text = _fetch_cli_text(site=cfg["site"], issuedby=cfg["issuedby"], version=v, session=session)
        rep_date = _parse_target_date(text)
        # The "SUMMARY FOR ..." date corresponds to the day being summarized (target_date).
        if rep_date != target_date:
            continue
        max_f = _parse_yesterday_max(text)
        if max_f is None:
            continue
        source_url = (
            f"https://forecast.weather.gov/product.php?site={cfg['site']}&issuedby={cfg['issuedby']}"
            f"&product=CLI&format=txt&version={v}&glossary=0"
        )
        # issued line is in header; keep it best-effort
        issued_line = next((ln for ln in text.splitlines() if "AM" in ln or "PM" in ln), "")
        return CliTruth(
            city=city,
            target_date=target_date,
            observed_max_f=max_f,
            report_issued_ts=issued_line.strip(),
            version=v,
            source_url=source_url,
        )
    raise RuntimeError(f"Could not find CLI report for {city} date={target_date} within {max_versions} versions")

