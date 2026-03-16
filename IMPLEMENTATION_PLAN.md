# Multi-Point Weather Data — Implementation Plan

**Status:** Approved by Captain — Dev deployment first
**Date:** 2026-03-15
**Scope:** BE, FR, DE — wind (onshore/offshore) + solar en priorité

## Architecture

### Principe clé : A/B side-by-side
- L'ancien pipeline (centroïde) reste **inchangé** et continue de tourner
- Le nouveau pipeline (multi-points) tourne **en parallèle**
- Les forecasts sont stockés avec un tag `weather_mode` pour comparaison
- On peut comparer en production avant de switcher

### Modifications par module

---

## 1. Module `energy-data-gathering` (collecte météo)

### 1a. Nouveau fichier : `src/fetch_weather_multipoint.py`
- Importe les zones depuis `weather_zones_real.py` (copié depuis energy-forecast)
- Fetch Open-Meteo pour tous les points d'un pays
- Calcule la moyenne pondérée par forecast_type
- Stocke dans une **nouvelle table** `weather_data_multipoint`

### 1b. Nouvelle table : `weather_data_multipoint`
```sql
CREATE TABLE weather_data_multipoint (
    country_code TEXT NOT NULL,
    forecast_type TEXT NOT NULL,  -- 'wind_onshore', 'wind_offshore', 'solar', 'load'
    timestamp_utc TEXT NOT NULL,
    forecast_run_time TEXT,
    temperature_2m_k REAL,
    dew_point_2m_k REAL,
    relative_humidity_2m_frac REAL,
    pressure_msl_hpa REAL,
    wind_speed_10m_ms REAL,
    wind_gusts_10m_ms REAL,
    wind_direction_10m_deg REAL,
    wind_speed_100m_ms REAL,
    wind_direction_100m_deg REAL,
    wind_speed_80m_ms REAL,
    wind_speed_120m_ms REAL,
    precip_mm REAL,
    rain_mm REAL,
    snowfall_mm REAL,
    shortwave_radiation_wm2 REAL,
    direct_radiation_wm2 REAL,
    direct_normal_irradiance_wm2 REAL,
    diffuse_radiation_wm2 REAL,
    model_name TEXT DEFAULT 'era5',
    data_quality TEXT DEFAULT 'multipoint_weighted',
    n_points INTEGER,  -- nombre de points utilisés
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (country_code, forecast_type, timestamp_utc)
);
```

### 1c. Intégration cron
- Ajouter un appel après le fetch centroïde existant
- `fetch_weather_multipoint.py` tourne pour BE, FR, DE
- Les autres pays continuent en centroïde uniquement

### 1d. Aucune modification des fichiers existants
- `fetch_weather.py` → inchangé
- `weather_data` table → inchangée
- Pipeline actuel → aucun risque de régression

---

## 2. Module `energy-forecast` (prédiction)

### 2a. Modification : `src/db.py` — nouvelle fonction
```python
def load_training_data_multipoint(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Load training data using multipoint weather instead of centroid."""
    # Charge energy data normalement
    # Charge weather depuis weather_data_multipoint au lieu de weather_data
    # Merge identique
```

### 2b. Modification : `src/forecaster.py` — paramètre weather_mode
```python
class Forecaster:
    def __init__(self, ..., weather_mode='centroid'):
        self.weather_mode = weather_mode  # 'centroid' ou 'multipoint'
    
    def train(self, ...):
        if self.weather_mode == 'multipoint':
            df = load_training_data_multipoint(...)
        else:
            df = load_training_data(...)  # existant, inchangé
```

### 2c. Modification : `scripts/forecast_daily.py`
- Après le forecast centroïde (inchangé), lancer un forecast multipoint pour BE/FR/DE
- Stocker avec `model_name = 'xgboost_multipoint'` dans la table forecasts

### 2d. Modification : `scripts/train.py`
- Nouveau flag `--weather-mode multipoint`
- Entraîne des modèles séparés stockés dans `models/{country}/{type}/multipoint/`
- Les modèles centroïde dans `models/{country}/{type}/` restent intacts

---

## 3. Stockage des modèles

```
models/
├── BE/
│   ├── wind_onshore/
│   │   ├── model.joblib              # centroïde (inchangé)
│   │   ├── production/model.joblib   # centroïde prod (inchangé)
│   │   └── multipoint/
│   │       └── model.joblib          # nouveau modèle multipoint
│   ├── solar/
│   │   ├── model.joblib
│   │   └── multipoint/model.joblib
│   ...
```

---

## 4. Comparaison / Monitoring

### 4a. Script : `scripts/compare_weather_modes.py`
- Charge les forecasts centroïde et multipoint pour la même période
- Compare MAE, MAPE, RMSE contre les actuals
- Génère un rapport markdown

### 4b. Dashboard (futur)
- Ajouter un toggle "Weather Mode" sur la page forecast
- Afficher les deux courbes side-by-side

---

## 5. Ordre d'exécution

### Phase 1 : Data pipeline (energy-data-gathering)
1. Copier `weather_zones_real.py` dans energy-data-gathering
2. Créer table `weather_data_multipoint`
3. Créer `fetch_weather_multipoint.py`
4. Backfill données historiques (90 jours minimum pour entraînement)
5. Ajouter au cron de collecte

### Phase 2 : Forecasting (energy-forecast)  
6. Ajouter `load_training_data_multipoint()` dans db.py
7. Ajouter `weather_mode` param dans Forecaster
8. Entraîner les modèles multipoint pour BE/FR/DE (wind + solar)
9. Ajouter le forecast multipoint dans forecast_daily.py

### Phase 3 : Validation
10. Lancer compare_weather_modes.py
11. Vérifier que les modèles centroïde n'ont PAS changé
12. Documenter les résultats

---

## Contraintes
- ⚠️ Aucun fichier existant ne doit être cassé
- ⚠️ Les modèles centroïde tournent en parallèle, TOUJOURS
- ⚠️ Dev uniquement — pas de déploiement prod tant que non validé
- ⚠️ Seulement BE, FR, DE pour l'instant
