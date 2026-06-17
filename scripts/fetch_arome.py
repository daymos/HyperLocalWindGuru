"""
Fetch AROME forecast data from the Météo-France public API.

Free API — requires account registration at portail-api.meteofrance.fr
and subscription to the AROME product to get a client key.

Auth flow:
  POST https://portail-api.meteofrance.fr/token
  with Authorization: Basic <base64(client_id:client_secret)>
  → returns {"access_token": "...", "expires_in": 3600}

Data:
  WCS GetCoverage returns a GRIB2 file for one parameter / run / timestep / height / bbox.
  Parsed with cfgrib into an xarray Dataset, then point-extracted nearest to each station.

Coverage: lon [-12, 16], lat [37.5, 55.4] — covers UK south coast.
Retention: 5-day rolling window on this endpoint.
Resolution: 0.01° (~1.3 km) — MF-NWP-HIGHRES-AROME-001-FRANCE-WCS
"""

import os
import base64
import tempfile
import requests
import cfgrib
import xarray as xr
import pandas as pd
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOKEN_URL = "https://portail-api.meteofrance.fr/token"
BASE_URL = "https://public-api.meteofrance.fr/public/arome/1.0"
COVERAGE = "MF-NWP-HIGHRES-AROME-001-FRANCE-WCS"

PARAMETERS = {
    "wind_speed_ms":   "WIND_SPEED__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "wind_gust_ms":    "WIND_SPEED_GUST__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "temperature_k":   "TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
}

# UK south coast stations — add your spots here
STATIONS = {
    "spot_594225": {"lat": 50.7, "lon": -1.8},   # example: Bournemouth area
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_token(client_id: str, client_secret: str) -> str:
    """Exchange client credentials for a bearer token."""
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {credentials}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Fetch one GRIB2 field
# ---------------------------------------------------------------------------

def fetch_coverage(
    token: str,
    parameter: str,
    run_time: datetime,
    valid_time: datetime,
    height_m: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> bytes:
    """
    Download one GRIB2 field from AROME WCS GetCoverage.

    Args:
        token:      Bearer token
        parameter:  WCS parameter name e.g. WIND_SPEED__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND
        run_time:   Model run datetime (UTC)
        valid_time: Forecast valid datetime (UTC)
        height_m:   Height above ground in metres (e.g. 10, 100)
        lat_min/max, lon_min/max: Bounding box

    Returns:
        Raw GRIB2 bytes
    """
    run_str = run_time.strftime("%Y-%m-%dT%H.%M.%SZ")
    valid_str = valid_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    coverage_id = f"{parameter}___{run_str}"

    url = (
        f"{BASE_URL}/wcs/{COVERAGE}/GetCoverage"
        f"?service=WCS&version=2.0.1"
        f"&coverageid={coverage_id}"
        f"&subset=time({valid_str})"
        f"&subset=height({height_m})"
        f"&subset=lat({lat_min},{lat_max})"
        f"&subset=long({lon_min},{lon_max})"
        f"&format=application/wmo-grib"
    )

    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "accept": "application/octet-stream",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Point extraction
# ---------------------------------------------------------------------------

def grib_to_point(grib_bytes: bytes, lat: float, lon: float) -> float:
    """Write GRIB2 bytes to a temp file, open with cfgrib, extract nearest point."""
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        tmp_path = f.name

    try:
        ds = cfgrib.open_dataset(tmp_path)
        # Select nearest grid point
        da = list(ds.data_vars.values())[0]
        value = float(da.sel(latitude=lat, longitude=lon, method="nearest").values)
    finally:
        os.unlink(tmp_path)

    return value


# ---------------------------------------------------------------------------
# Main: fetch all parameters for all stations at one timestep
# ---------------------------------------------------------------------------

def fetch_arome_for_stations(
    token: str,
    run_time: datetime,
    valid_time: datetime,
    stations: dict = STATIONS,
    pad_deg: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch AROME wind speed, gust, and temperature for all stations at one timestep.

    Returns a DataFrame with one row per station:
        station_id, run_time, valid_time, lead_hours,
        wind_speed_ms, wind_gust_ms, temperature_k,
        wind_speed_10m_ms, wind_speed_100m_ms  (for shear feature)
    """
    records = []

    # Build a bbox that covers all stations with padding
    all_lats = [s["lat"] for s in stations.values()]
    all_lons = [s["lon"] for s in stations.values()]
    lat_min = min(all_lats) - pad_deg
    lat_max = max(all_lats) + pad_deg
    lon_min = min(all_lons) - pad_deg
    lon_max = max(all_lons) + pad_deg

    # Fetch each parameter once (shared bbox, one GRIB per param)
    gribs = {}
    for key, param in PARAMETERS.items():
        height = 100 if key == "wind_speed_ms" else 10  # also fetch 100m wind for shear
        print(f"  Fetching {key} at {height}m for run={run_time.isoformat()} valid={valid_time.isoformat()}...")
        gribs[key] = fetch_coverage(
            token, param, run_time, valid_time,
            height_m=height,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )

    # Also fetch 10m wind speed separately for shear calculation
    print("  Fetching wind_speed 10m for shear...")
    gribs["wind_speed_10m_ms"] = fetch_coverage(
        token, PARAMETERS["wind_speed_ms"], run_time, valid_time,
        height_m=10,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
    )

    lead_hours = (valid_time - run_time).total_seconds() / 3600

    for station_id, coords in stations.items():
        lat, lon = coords["lat"], coords["lon"]
        record = {
            "station_id": station_id,
            "run_time": run_time,
            "valid_time": valid_time,
            "lead_hours": lead_hours,
            "lat": lat,
            "lon": lon,
        }
        for key, grib_bytes in gribs.items():
            record[key] = grib_to_point(grib_bytes, lat, lon)

        # Derived: wind shear index (100m - 10m) / 90
        if "wind_speed_ms" in record and "wind_speed_10m_ms" in record:
            record["wind_shear_index"] = (record["wind_speed_ms"] - record["wind_speed_10m_ms"]) / 90.0

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Load credentials from environment — never hardcode
    client_id = os.environ.get("MF_CLIENT_ID")
    client_secret = os.environ.get("MF_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("Set MF_CLIENT_ID and MF_CLIENT_SECRET environment variables.")
        print("Get them from: https://portail-api.meteofrance.fr")
        sys.exit(1)

    token = get_token(client_id, client_secret)
    print("Token obtained.")

    # Example: fetch the most recent AROME run (00 UTC today) for +3h
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    run_time = now.replace(hour=0)       # 00Z run
    valid_time = run_time.replace(hour=3) # +3h forecast

    df = fetch_arome_for_stations(token, run_time, valid_time)
    print(df.to_string(index=False))
    df.to_parquet(f"arome_{run_time.strftime('%Y%m%d_%H%M')}.parquet", index=False)
    print("Saved.")
