# HyperLocal Wind Forecast — Project Brief

## Goal

Produce site-specific, short-horizon wind forecasts for a handful of spots on the UK south coast. The approach is a **residual correction model**: take an AROME NWP forecast as the base signal, train an ML model to predict the local error at each station, and add the correction back to get a sharper final forecast.

---

## Why this could work

AROME is already a high-resolution (1.3 km) mesoscale model that captures coastal effects well. But it still carries systematic local biases at specific spots: terrain funnelling, sea-breeze timing, station exposure. Those biases are more learnable than raw wind, so correcting the residual on top of AROME is cheaper and more accurate than predicting wind from scratch.

---

## Data sources

| Source | What we get | How |
|---|---|---|
| Météo-France open API | AROME forecast: wind speed, direction, gust, pressure, temp | REST API (portail-api.meteofrance.fr), GRIB2 / JSON |
| Windguru weather stations | Observed wind speed, direction, timestamp per station | Windguru station upload endpoint (we own the stations) |

AROME covers the UK south coast / Channel domain. Forecasts run every 3 h out to +42 h. We pull the fields at the grid point nearest each station.

Windguru Stations data is available to export. 5+ years of data with 1 hours resolution exists, but each stations was started at a different time. 

---

## Model options

One model per site × forecast horizon. Three realistic options, ordered by complexity:

### Option A — LightGBM ✅ recommended start

| | |
|---|---|
| **Output** | Point estimate (single wind value) |
| **Training hardware** | CPU only — standard GitHub Actions runner (free) |
| **Serving** | Tiny binary (~1 MB), load in Cloud Run with no GPU |
| **Storage** | GCS bucket, versioned by date prefix |
| **Pros** | Fast, low friction, no scaling required, SHAP interpretability, drop-in for XGBoost |
| **Cons** | Point estimate only — no confidence intervals |

### Option B — NGBoost

| | |
|---|---|
| **Output** | Full probability distribution (mean + uncertainty bounds) |
| **Training hardware** | CPU only — GitHub Actions runner works fine |
| **Serving** | Pure Python, slightly heavier but still Cloud Run / CPU |
| **Storage** | GCS bucket, serialise with joblib |
| **Pros** | Gives confidence intervals ("12 kt, 90% CI: 9–15 kt") — genuinely useful for wind sports |
| **Cons** | More complex to evaluate and serve; less mature ecosystem than LightGBM |

### Option C — Temporal Fusion Transformer (TFT)

| | |
|---|---|
| **Output** | Multi-horizon probabilistic forecasts in one model |
| **Training hardware** | GPU strongly preferred — dedicated container on GCE or Cloud Run GPU |
| **Training time** | Minutes to hours depending on dataset size |
| **Serving** | PyTorch model, ~100 MB+, needs GPU or beefy CPU Cloud Run instance |
| **Storage** | GCS bucket, checkpoint + final model saved as `.pt` |
| **Pros** | Architecturally designed for this — known future covariates (AROME), multi-horizon, attention weights show which lags matter |
| **Cons** | Needs 2+ years of data to reliably beat LightGBM; much more engineering overhead |

**Suggested path:** start with LightGBM, add NGBoost if uncertainty estimates are valuable, consider TFT only after you have enough data and LightGBM has hit its ceiling.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Data layer (GCP)                        │
│                                                                  │
│  Météo-France API ──► Cloud Function (ingest, every 3 h)        │
│                        • download AROME GRIB2 for each station   │
│                          grid point (wind, gust, temp, BLH…)    │
│                        • store raw fields → GCS/arome/           │
│                                                                  │
│  Station uploads  ──► Cloud Function (receive, on push)          │
│                        • receive Windguru station JSON           │
│                        • store raw observations → GCS/stations/  │
│                                                                  │
│                       Alignment step (inside Cloud Function):    │
│                        • snap station obs to nearest UTC hour    │
│                        • join on valid_time = observation time   │
│                        • drop rows with missing obs or forecast  │
│                        • result: one row per station × hour with │
│                          both forecast fields and observed wind  │
│                        • store aligned dataset → GCS/aligned/    │
│                          as Parquet, partitioned by station/year │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────┐
│                    Training pipeline (GitHub Actions)            │
│                                                                  │
│  on: schedule (daily 03:00 UTC)  |  workflow_dispatch            │
│                                                                  │
│  1. fetch-data       pull latest Parquet from GCS                │
│  2. feature-eng      lags, sin/cos direction, time features      │
│  3. compute-residuals  observed − AROME forecast per horizon     │
│  4. train            LightGBM per site × horizon (CPU, ~seconds) │
│  5. evaluate         RMSE/MAE corrected vs raw AROME baseline    │
│  6. push-model       write to GCS models/ only if RMSE improved  │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────┐
│                   Model registry (GCS bucket)                    │
│                                                                  │
│  gs://wind-forecast/models/                                      │
│    {site}/          one folder per station (camber-sands/ etc.)  │
│      {horizon}/     one folder per lead time (1h/, 3h/, 6h/)    │
│                     separate model per horizon because residual  │
│                     patterns change with lead time — at 1h,      │
│                     recent observed wind dominates; at 6h,       │
│                     the AROME signal matters much more           │
│        {date}/      dated training run for rollback history      │
│          model.pkl                                               │
│          feature_schema.json                                     │
│        latest.json  small file recording the active version:     │
│                     {"version":"2025-01-01","rmse":1.82}         │
│                     Cloud Run reads this on startup to know      │
│                     which model.pkl to load — update this file   │
│                     to promote or roll back without redeploying  │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────┐
│                      Serving (GCP Cloud Run)                     │
│                                                                  │
│  FastAPI container (CPU, scales to zero)                         │
│  - loads model from GCS on startup (service account auth)        │
│  - GET /forecast?site=xxx&horizon=3h                             │
│    → fetches latest AROME forecast from Météo-France API         │
│    → builds feature row                                          │
│    → returns { base_kt, corrected_kt, residual_kt }             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Feature set (per training row)

### Base AROME fields

| Feature | Notes |
|---|---|
| Wind speed 10 m (m/s) | Base signal for residual target |
| Wind direction → sin + cos | Cyclical encoding, avoids 359°/1° discontinuity |
| Gust speed | Often the operationally relevant number |
| Surface pressure | Synoptic context |
| 2 m air temperature | Input to land–sea contrast feature |
| Boundary layer height (BLH) | Controls whether surface decouples from synoptic flow; low BLH = local effects dominate |
| Wind speed at 100 m | Combined with 10 m to compute shear / stability proxy |
| Wind direction at 100 m → sin + cos | Directional veer; backing = unstable, veering = stable |

### Derived dynamic features

| Feature | Source | Why it matters |
|---|---|---|
| Solar elevation angle | Computed via pvlib from lat/lon + timestamp | More precise sea-breeze trigger than hour or day-of-year alone |
| Land–sea temperature contrast | AROME 2 m temp − CMEMS daily SST | Primary driver of sea-breeze strength and timing on the Channel coast |
| Pressure tendency dp/dt | Rolling diff on AROME surface pressure (1 h, 3 h) | Frontal passage detection; sharp fall = incoming front changes local flow |
| Wind shear index | (wind_100m − wind_10m) / 90 | Stability proxy derived from AROME multi-level output |

### Time features

| Feature | Notes |
|---|---|
| Hour of day → sin + cos | Diurnal sea-breeze cycle |
| Day of year → sin + cos | Seasonal heating and SST cycle |
| Month | Coarser seasonal signal, useful for tree models |

### Observation lags (temporal memory)

| Feature | Notes |
|---|---|
| Observed wind speed at t−1h, t−2h, t−3h | Critical for short horizons; persistence is a strong baseline |
| Observed direction (sin + cos) at t−1h | Recent local trend |

### Static site features (computed once per station)

| Feature | Source | Notes |
|---|---|---|
| Fetch distance in forecast wind direction | Lookup table computed from OS/GEBCO coastline (8 × 45° bins) | Dynamic at inference: look up the bin matching current forecast direction |
| Wind direction relative to coastline normal | Dot product of forecast wind vector and coastline perpendicular | +1 = fully onshore, −1 = offshore, ~0 = shore-parallel (often accelerates) |
| Directional exposure / shelter index per sector | Precomputed from OS Terrain 50 DEM (horizon angle or TPI per 30° bin) | Captures sheltering by cliffs, headlands, bays |
| Station elevation (m) | Station metadata | Exposure effect |
| Distance to nearest coastline (km) | Computed | Inland stations damp sea-breeze signal |
| Dominant land cover upwind at 1 km / 5 km / 10 km | CORINE or OS land use | Urban, agricultural, or water surface roughness |

> **Note on channelling:** for sites near gaps between headlands or valley mouths, terrain gradient data (DEM aspect and slope in the upwind fetch window) can reveal whether a given wind direction aligns with a channelling axis. This is the most physically meaningful directional feature but also the hardest to compute — it requires proper terrain analysis tooling (e.g. QGIS, GDAL, or WAsP).

### Tidal features

| Feature | Source | Notes |
|---|---|---|
| Tidal height (m) | NTSLF or Admiralty Tidal API | Low tide exposes intertidal flats, changes local roughness and thermal properties |
| Tidal phase (flood / ebb / slack) | Derived from height time series | Coarser signal, useful for estuarine sites |

---

Target: `residual = observed_wind[t+h] − arome_forecast[t, h]`

### Feature priority for initial build

> These features are listed in order of value and engineering effort. The full list above is **aspirational** — we may not be able to extract all of them, and the model will still work well with a subset. Start with the first two tiers and add more only if RMSE improvement stalls.

1. **Must have** — Base AROME fields + time features + observation lags. Enough for a working model.
2. **High value, low cost** — Solar elevation angle, land–sea contrast, pressure tendency, BLH, wind shear. All derivable from AROME or pvlib with minimal extra work.
3. **Worth the effort** — Directional fetch lookup, wind direction vs coastline normal. Requires one-off geometric computation per station but is then static.
4. **Add if relevant to the site** — Tidal state (estuarine or wide foreshore sites), land cover upwind.
5. **Aspirational** — Directional exposure/shelter index, channelling indicator from DEM. Meaningful physical signal but requires terrain analysis tooling and may be hard to validate.

---

## Validation

Chronological split — never shuffle time series:

- Train: oldest 70 %
- Validation: next 15 % (early stopping / hyperparameter tuning)
- Test: most recent 15 % (held out, reported once)

Key metrics: RMSE and MAE of raw AROME vs corrected forecast on the test set. Model only written to GCS if corrected RMSE < raw AROME RMSE on the latest validation window.

---

## Evaluation dashboard

A Streamlit app deployed on Cloud Run alongside the forecast API. Its job is to make it immediately obvious whether the correction is helping, where it is failing, and what the current live forecast looks like.

### Tech stack

**Streamlit** — Python, no frontend work, easy to deploy on Cloud Run. Reads directly from GCS (same Parquet files as the training pipeline). Plotly for interactive charts.

### Views

**Day plot — the main view**

Line chart, wind speed (kt) on Y, time on X, covering a selectable day or rolling 24 h / 48 h window.

Three series:
- `AROME forecast` — raw model output for that horizon
- `Station observed` — actual recorded wind
- `Corrected forecast` — our residual-adjusted prediction

Shaded error bands optional (if using NGBoost confidence intervals).

```
kt
20 │      ╭──╮
   │  ~~~~╯  ╰~~~AROME
15 │    ●●●●●●●●●●●● Observed  
   │  ──────────────── Corrected
10 │
   └─────────────────────── time
     06:00   12:00   18:00
```

Metric chips above the chart: `AROME RMSE: 3.2 kt` → `Corrected RMSE: 1.9 kt` → `↓ 41%`

**Site selector** — dropdown to switch between stations. Each site shows its own model's performance independently.

**Horizon selector** — toggle between 1 h, 3 h, 6 h forecasts. Useful for seeing where the model degrades with lead time.

**Model drift panel**

Rolling 7-day and 30-day RMSE trend lines for both AROME and corrected. If the corrected RMSE starts creeping up toward AROME RMSE, the model is drifting and needs retraining or the residual pattern has changed (e.g. seasonal shift).

**Direction breakdown**

Wind rose or bar chart showing RMSE by wind direction sector (e.g. 8 × 45° bins). Reveals if the model struggles in specific flow regimes (e.g. easterlies, which are less common in training data).

### Deployment

Separate Cloud Run service, same Docker image base as the forecast API. Reads from GCS with the same service account. No public auth needed initially — Cloud Run IAP or a simple token header is enough to keep it internal.

---

## GCP services used

| Service | Role |
|---|---|
| Cloud Storage | Raw GRIB/JSON from AROME, station observations as Parquet, versioned model artefacts |
| Cloud Functions | Ingest AROME every 3 h + receive station data |
| Cloud Run (forecast) | FastAPI forecast endpoint |
| Cloud Run (dashboard) | Streamlit evaluation dashboard |
| Artifact Registry | Docker images for both services |
| GitHub Actions secret | Météo-France API key stored as a repository secret (`MF_API_KEY`), injected as env var at deploy/run time |

---

## Phased build plan

| Phase | What |
|---|---|
| 0 | Collect 6+ months data, explore in notebook, sanity-check AROME vs station |
| 1 | Offline residual model, validate locally, confirm RMSE improvement |
| 2 | GitHub Actions pipeline, automated GCS model push |
| 3 | Cloud Functions for live AROME ingest + Cloud Run forecast API |
| 4 | Streamlit dashboard deployed on Cloud Run |
| 5 | Add lag features from nearby stations, monitor drift via dashboard |
