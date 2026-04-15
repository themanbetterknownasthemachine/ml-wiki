---
title: "Feature Engineering für Zeitreihen"
type: concept
tags: [feature-engineering, zeitreihen, forecasting]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-15
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

#### Erweiterte `day_before_holiday`-Logik

Ein einfaches "nächster Tag ist Feiertag" greift zu kurz. Für wochenendadjazente Feiertage braucht man 3 Fälle:

```python
def is_day_before_holiday(date, hol_all):
    # Fall 1: direkter nächster Tag ist Feiertag
    if (date + timedelta(days=1)).date() in hol_all:
        return 1.0
    # Fall 2: Freitag und nächster Montag ist Feiertag (langes Wochenende)
    if date.weekday() == 4 and (date + timedelta(days=3)).date() in hol_all:
        return 1.0
    # Fall 3: Donnerstag und Freitag ist Feiertag
    if date.weekday() == 3 and (date + timedelta(days=1)).date() in hol_all:
        return 1.0
    return 0.0
```

**Wichtig**: Nicht alle Feiertage bedeuten Betriebsschliessung. Es empfiehlt sich, zwei separate Sets zu pflegen:
- `holidays_closed`: Nur echte Schliessungstage
- `holidays_all`: Alle gesetzlichen Feiertage (für Vortags-Logik)

#### Kurzwochentage-Feature (`is_short_week`)

Wochen mit einem Feiertag haben weniger Arbeitstage — das Volumen wird auf weniger Tage verteilt:

```python
def is_short_week(date, hol_closed):
    monday = date - timedelta(days=date.weekday())
    return 1.0 if any(
        (monday + timedelta(days=i)).date() in hol_closed for i in range(5)
    ) else 0.0
```

Dieses Feature ist für alle Tage der Woche identisch — nicht nur den Feiertag selbst.

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

### Business-Day-Imputation für NeuralForecast

NeuralForecast erwartet einen lückenlosen Zeitreihen-Index. Fehlende Tage (Feiertage, Betriebsschliessungen) müssen vor dem Training aufgefüllt werden. Globaler Median ist dabei suboptimal — ein wochentag-spezifischer Median ist präziser:

```python
full_bday_idx = pd.bdate_range(df.index.min(), df.index.max(), freq="B")
df_full = df.reindex(full_bday_idx)

# Wochentag-spezifischer Median: Montag bekommt Montags-Median, etc.
weekday_medians = df.groupby(df.index.dayofweek)["value"].median()
df_full["value"] = df_full["value"].fillna(
    df_full.index.to_series().map(lambda d: weekday_medians[d.dayofweek])
)
```

**Wichtig**: Die aufgefüllten Tage sind technisches Hilfsmittel. Bei der Evaluation immer auf die ursprünglichen, echten Beobachtungen filtern (`original_dates_set`).

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

- [[nhits-tuning-dokumentation]] — Erweiterte Feiertagslogik, Kurzwochentage-Feature, Business-Day-Imputation
