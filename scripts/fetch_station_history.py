"""
Fetch multi-year station observation history from Windguru.

The Windguru session cookies expire periodically. When they do, log in via
the browser, open DevTools → Network, find any /int/iapi.php request, copy
the Cookie header and x-wg-token header, and update the CREDENTIALS dict
below (or set as environment variables).

Usage:
    python fetch_station_history.py

Output:
    station_{id}_history.parquet  — full history, deduplicated and sorted
    station_{id}_history.csv      — same, for quick inspection

Fields returned by the API:
    datetime, wind_avg (kt), wind_max (kt), wind_min (kt),
    wind_direction (deg), temperature (°C), rh (%), mslp (hPa), gustiness (%)
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — override with environment variables or edit here
# ---------------------------------------------------------------------------

STATION_ID   = int(os.environ.get("WG_STATION_ID",   "571"))
STATION_NAME = os.environ.get("WG_STATION_NAME", "southend-on-sea_teyc-jetty")

# Date range — 5 years back from today
END_DATE   = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = END_DATE.replace(year=END_DATE.year - 5)

# Windguru session — copy fresh values from DevTools when cookies expire
WG_SESSION   = os.environ.get("WG_SESSION",    "")
WG_IDU       = os.environ.get("WG_IDU",        "")
WG_LOGIN_MD5 = os.environ.get("WG_LOGIN_MD5",  "")
WG_TOKEN     = os.environ.get("WG_TOKEN",      "")

# Fetch parameters
CHUNK_DAYS   = 30        # days per API request — 30 is reliable
AVG_MINUTES  = 60        # averaging window; 60 = hourly observations
SLEEP_SEC    = 1.5       # polite delay between requests
MAX_RETRIES  = 3

# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": f"https://www.windguru.cz/station/{STATION_ID}",
        "x-wg-token": WG_TOKEN,
        "DNT": "1",
    })
    s.cookies.update({
        "session":   WG_SESSION,
        "idu":       WG_IDU,
        "login_md5": WG_LOGIN_MD5,
        "langc":     "en-",
    })
    return s


# ---------------------------------------------------------------------------
# Fetch one chunk
# ---------------------------------------------------------------------------

def fetch_chunk(session: requests.Session, chunk_start: datetime, chunk_end: datetime) -> pd.DataFrame:
    params = {
        "q":           "station_data",
        "id_station":  str(STATION_ID),
        "from":        chunk_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to":          chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "avg_minutes": str(AVG_MINUTES),
        "graph_info":  "1",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(
                "https://www.windguru.cz/int/iapi.php",
                params=params,
                timeout=30,
            )
            if resp.status_code == 401:
                log.error("401 Unauthorised — session cookies have expired. Refresh WG_SESSION, WG_IDU, WG_LOGIN_MD5, WG_TOKEN.")
                raise SystemExit(1)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("datetime"):
                return pd.DataFrame()  # no data for this period

            df = pd.DataFrame({
                "datetime":      pd.to_datetime(data["datetime"]),
                "wind_avg_kt":   data.get("wind_avg"),
                "wind_max_kt":   data.get("wind_max"),
                "wind_min_kt":   data.get("wind_min"),
                "wind_dir_deg":  data.get("wind_direction"),
                "temperature_c": data.get("temperature"),
                "humidity_pct":  data.get("rh"),
                "pressure_hpa":  data.get("mslp"),
                "gustiness_pct": data.get("gustiness"),
            })
            df["station_id"] = STATION_ID
            return df

        except (requests.RequestException, ValueError) as e:
            log.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)

    log.error(f"All retries failed for {chunk_start.date()} – {chunk_end.date()}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_parquet = f"station_{STATION_ID}_{STATION_NAME}_history.parquet"
    out_csv     = f"station_{STATION_ID}_{STATION_NAME}_history.csv"

    session = make_session()
    all_chunks = []

    chunk_start = START_DATE
    total_chunks = ((END_DATE - START_DATE).days // CHUNK_DAYS) + 1

    log.info(f"Fetching station {STATION_ID} from {START_DATE.date()} to {END_DATE.date()}")
    log.info(f"  {total_chunks} chunks × {CHUNK_DAYS} days, avg_minutes={AVG_MINUTES}")

    chunk_n = 0
    while chunk_start < END_DATE:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), END_DATE)
        chunk_n += 1

        log.info(f"[{chunk_n}/{total_chunks}] {chunk_start.date()} → {chunk_end.date()}")
        df = fetch_chunk(session, chunk_start, chunk_end)

        if not df.empty:
            all_chunks.append(df)
            log.info(f"  {len(df)} rows")
        else:
            log.info("  no data")

        chunk_start = chunk_end
        time.sleep(SLEEP_SEC)

    if not all_chunks:
        log.error("No data fetched.")
        return

    full = (
        pd.concat(all_chunks, ignore_index=True)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    full.to_parquet(out_parquet, index=False)
    full.to_csv(out_csv, index=False)

    log.info(f"Done. {len(full)} rows saved to {out_parquet}")
    log.info(f"  Date range: {full['datetime'].min()} → {full['datetime'].max()}")
    log.info(f"  Null wind_avg: {full['wind_avg_kt'].isna().sum()} ({full['wind_avg_kt'].isna().mean():.1%})")
    print(full.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
