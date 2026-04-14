---
title: "SARIMAX"
type: entity
tags: [forecasting, statistisch, zeitreihen, baseline]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
quellen: 0
---

# SARIMAX

## Überblick

SARIMAX (Seasonal ARIMA with eXogenous variables) ist ein klassisches statistisches Modell für Zeitreihen-Forecasting. Es modelliert die Zeitreihe als Kombination aus Autoregression (AR), Differenzierung (I), Moving Average (MA) und saisonalen Komponenten, plus optionale externe Variablen.

## Architektur / Funktionsweise

SARIMAX kombiniert vier Komponenten:

- **AR(p)**: Autoregression — der aktuelle Wert hängt von den letzten `p` Werten ab
- **I(d)**: Integration/Differenzierung — `d`-mal differenzieren um Stationarität herzustellen
- **MA(q)**: Moving Average — der aktuelle Wert hängt von den letzten `q` Vorhersagefehlern ab
- **Saisonal (P, D, Q, m)**: Gleiche Logik, aber auf saisonaler Ebene (z.B. `m=7` für Wochensaisonalität, `m=12` für Monatssaisonalität)
- **X**: Exogene Variablen (z.B. Feiertage, Wetter, Preise)

Die Notation `SARIMAX(p,d,q)(P,D,Q,m)` beschreibt das vollständige Modell.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    endog=train_series,
    exog=exog_features,         # Exogene Variablen (optional)
    order=(1, 1, 1),            # (p, d, q)
    seasonal_order=(1, 1, 1, 7) # (P, D, Q, m) — Wochensaisonalität
)

result = model.fit(disp=False)
forecast = result.forecast(steps=30, exog=exog_future)
```

**Wichtig: Stationarität prüfen** vor dem Modellbau mit ADF-Test oder KPSS-Test. Wenn die Reihe nicht stationär ist → differenzieren (`d` erhöhen) oder Transformation (Log, Box-Cox).

## Eingesetzte Konzepte

- [[Stationarität]] — Grundvoraussetzung für ARIMA-Modelle
- [[Feature Engineering für Zeitreihen]] — Exogene Features (X-Komponente)
- [[Backtesting]] — Evaluation mit zeitlich korrekter Train/Test-Aufteilung
- [[Informationskriterien AIC BIC]] — Zur Modellselektion der (p,d,q)-Parameter

## Stärken

- Interpretierbar: jeder Koeffizient hat eine klare statistische Bedeutung
- Unsicherheitsintervalle (Konfidenzintervalle) automatisch verfügbar
- Robust bei wenig Daten (funktioniert ab ~2 Jahre Monatsdaten)
- Schnelles Training (Sekunden statt Minuten/Stunden)
- Gute Baseline: jedes ML-Modell sollte SARIMAX schlagen, sonst lohnt sich die Komplexität nicht

## Schwächen / Limitierungen

- Schwächer bei komplexen, nichtlinearen Mustern (→ hier sind [[N-HiTS]] oder [[LightGBM Forecasting]] besser)
- Nur eine Zeitreihe pro Modell (kein Multi-Series Training)
- Manuelle Parameterwahl nötig (oder Auto-ARIMA via `pmdarima`, aber das kann langsam sein)
- Forecast-Qualität degradiert stark bei langem Horizont
- Saisonale Order mit hohem `m` (z.B. `m=365` für Jahressaisonalität) ist rechnerisch nicht praktikabel

## Verwandte Entities

- [[N-HiTS]] — Deep-Learning-Alternative, meist besser bei täglichen Forecasts
- [[Prophet]] — Facebooks Dekompositions-Modell, ähnliche Nische
- [[LightGBM Forecasting]] — Gradient Boosting auf engineered Features

## Offene Fragen / Nächste Schritte

- Vergleich mit [[Foundation Models für Zeitreihen]] auf monatlicher Granularität (wo SARIMAX traditionell stark ist)

## Quellen
