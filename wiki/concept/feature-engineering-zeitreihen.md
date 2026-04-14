---
title: "Feature Engineering für Zeitreihen"
type: concept
tags: [feature-engineering, zeitreihen, forecasting]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---

# Feature Engineering für Zeitreihen

## Kernidee

Feature Engineering für Zeitreihen bedeutet: aus dem Datum und den historischen Werten zusätzliche Informationen ableiten, die dem Modell helfen, Muster zu erkennen. Ein rohes Datum ("2026-04-14") sagt einem Modell wenig — aber "Dienstag", "KW 16", "kein Feiertag", "7-Tage-Durchschnitt = X" sind verwertbare Signale.

**Wichtig**: Bei Zeitreihen-Features muss man **Data Leakage** vermeiden — kein Feature darf Informationen aus der Zukunft enthalten. Alle Features müssen zum Zeitpunkt der Vorhersage berechenbar sein.

## Feature-Kategorien

### Kalender-Features (deterministisch)
Zum Vorhersagezeitpunkt bekannt — kein Leakage-Risiko.

```python
import pandas as pd

df['dayofweek'] = df['date'].dt.dayofweek      # 0=Mo, 6=So
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
df['day_of_year'] = df['date'].dt.dayofyear
```

### Feiertags-Features
Besonders wichtig für Logistik- und Retail-Forecasting.

```python
import holidays

ch_holidays = holidays.Switzerland(prov='ZH', years=[2024, 2025, 2026])

df['is_holiday'] = df['date'].isin(ch_holidays).astype(int)
df['is_brueckentag'] = (
    df['date'].shift(-1).isin(ch_holidays) | 
    df['date'].shift(1).isin(ch_holidays)
).astype(int) & (~df['is_holiday'])

# Tage bis zum nächsten Feiertag
holiday_dates = sorted(ch_holidays.keys())
df['days_to_next_holiday'] = df['date'].apply(
    lambda d: min((h - d.date()).days for h in holiday_dates if h >= d.date())
)
```

### Lag-Features (historisch)
Vergangene Werte als Input — Kernidee der Autoregression.

```python
# Direkte Lags
for lag in [1, 7, 14, 28]:
    df[f'lag_{lag}'] = df['value'].shift(lag)

# Rolling Statistics
for window in [7, 14, 30]:
    df[f'rolling_mean_{window}'] = df['value'].shift(1).rolling(window).mean()
    df[f'rolling_std_{window}'] = df['value'].shift(1).rolling(window).std()

# WICHTIG: shift(1) bevor rolling() — sonst Data Leakage!
```

### Trend- und Saisonalitäts-Features

```python
import numpy as np

# Linearer Trend
df['trend'] = range(len(df))

# Sinusförmige Saisonalität (für Wochenmuster)
df['sin_weekday'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['cos_weekday'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

# Für Jahresmuster
df['sin_yearly'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['cos_yearly'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
```

## Wann einsetzen

- Bei Gradient-Boosting-Modellen ([[LightGBM Forecasting]]) — die können nicht selbst aus dem Datum lernen
- Als zusätzliche exogene Features für [[N-HiTS]] oder [[SARIMAX]]
- Zur explorativen Analyse: Feature-Korrelationen zeigen, welche Muster in den Daten stecken

## Wann NICHT einsetzen

- [[Foundation Models für Zeitreihen]] (Chronos-2, TimesFM) brauchen oft kein manuelles Feature Engineering — sie lernen Muster direkt aus den Rohwerten
- Zu viele Features → Overfitting-Risiko (→ [[Bias-Variance Tradeoff]])
- Perfekt korrelierte Features vermeiden (z.B. `month` UND `quarter` UND `sin_yearly` zusammen ist redundant)

## Praxisbeispiel

Vollständige Feature-Pipeline für einen täglichen Forecast:

```python
def create_features(df, target_col='value'):
    """Feature Engineering Pipeline für tägliche Zeitreihen.
    
    Alle Features sind zum Vorhersagezeitpunkt berechenbar (kein Leakage).
    """
    df = df.copy()
    
    # Kalender
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Feiertage (Schweiz, Kanton Zürich)
    ch_holidays = holidays.Switzerland(prov='ZH', years=df['date'].dt.year.unique())
    df['is_holiday'] = df['date'].dt.date.isin(ch_holidays).astype(int)
    
    # Lags (shift sichert gegen Leakage)
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling (auf geshiftete Daten!)
    shifted = df[target_col].shift(1)
    df['rolling_mean_7'] = shifted.rolling(7).mean()
    df['rolling_mean_28'] = shifted.rolling(28).mean()
    df['rolling_std_7'] = shifted.rolling(7).std()
    
    # Saisonalität (Sin/Cos Encoding)
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df.dropna()  # Erste Rows ohne Lag-Werte entfernen
```

## Eingesetzt in

- [[N-HiTS]] — Exogene Features (Kalender, Feiertage)
- [[LightGBM Forecasting]] — Vollständiges Feature-Set (Lags, Rolling, Kalender)
- [[SARIMAX]] — Exogene Variablen (X-Komponente)

## Verwandte Konzepte

- [[SHAP Explainability]] — Erklärt welche Features wie viel beitragen
- [[Bias-Variance Tradeoff]] — Zu viele Features = Overfitting
- [[Backtesting]] — Features müssen über den gesamten Backtest-Zeitraum stabil sein

## Quellen
