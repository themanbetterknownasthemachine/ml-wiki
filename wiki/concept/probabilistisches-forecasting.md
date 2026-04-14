---
title: "Probabilistisches Forecasting"
type: concept
tags: [forecasting, unsicherheit, quantile, probabilistisch, evaluation]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---

# Probabilistisches Forecasting

Statt eines einzelnen Punktwertes (`ŷ_t = 42`) liefert probabilistisches Forecasting eine **Verteilung** über mögliche Zukunftswerte — zum Beispiel als Quantile oder Konfidenzintervalle. Das ist oft wertvoller als der beste Schätzwert allein.

## Wann brauche ich Unsicherheitsintervalle?

**Immer dann wenn die Konsequenz von Über- vs. Unterschätzung asymmetrisch ist:**

| Szenario | Warum Verteilung wichtig |
|----------|------------------------|
| **Lagerhaltung** | Zu wenig bestellt → Stockout (teuer). Zu viel → Lagerkosten. Optimaler Bestand = Funktion des Quantils, nicht des Mittelwerts |
| **Ressourcenplanung** | "95% der Zeit reicht Kapazität X" ist ein anderes Statement als "Durchschnitt = X" |
| **Risikoabschätzung** | Worst-Case Szenarien (obere Quantile) für Puffer-Planung |
| **Anomalie-Erkennung** | Punkt ausserhalb des 99%-Intervalls = starkes Signal |
| **Modellvertrauen** | Breite Intervalle = Modell ist unsicher → manuelle Prüfung triggern |

Bei symmetrischen, linearen Kosten (MAE-Optimierung) reicht der Medianwert. Sobald Kosten asymmetrisch sind: Quantile nutzen.

## Quantile als Output

Ein Quantile-Forecast `q_τ` bei Niveau τ ∈ (0,1) bedeutet:
> "Mit Wahrscheinlichkeit τ liegt der echte Wert unter diesem Forecast."

```
τ = 0.10 → 10%-Quantil (untere Grenze)
τ = 0.50 → Median (bester Einzelpunkt-Schätzer unter MAE-Verlust)
τ = 0.90 → 90%-Quantil (obere Grenze)

90%-Prognoseintervall = [q_0.05, q_0.95]
```

## N-HiTS: Quantile Regression konfigurieren

[[N-HiTS]] liefert standardmässig nur Punktvorhersagen. Für Quantile muss die Loss-Funktion explizit gesetzt werden:

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

# MQLoss = Multi-Quantile Loss
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

model = NHITS(
    h=30,
    input_size=90,
    loss=MQLoss(quantiles=quantiles),   # statt default MAE
    n_pool_kernel_size=[8, 4, 1],
    n_freq_downsample=[4, 2, 1],
    max_steps=1000,
)

nf = NeuralForecast(models=[model], freq='D')
nf.fit(df=train_df)
forecast = nf.predict()
# forecast enthält jetzt: NHITS-q10, NHITS-q25, NHITS-median, NHITS-q75, NHITS-q90
```

**Quantile Crossing vermeiden**: MQLoss garantiert nicht automatisch `q_0.1 ≤ q_0.5 ≤ q_0.9`. Bei Bedarf:
```python
from neuralforecast.losses.pytorch import MQLoss, HuberMQLoss
# HuberMQLoss ist robuster gegenüber Ausreissern
```

## Chronos-2: Quantile out-of-the-box

[[Foundation Models für Zeitreihen|Chronos-2]] liefert **21 Quantile** ohne zusätzliche Konfiguration (aus [[chronos-2-paper]]):
`{0.01, 0.05, 0.1, 0.15, …, 0.85, 0.9, 0.95, 0.99}`

```python
from chronos import ChronosPipeline
import torch
import numpy as np

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

forecasts = pipeline.predict(
    context=torch.tensor(history),
    prediction_length=30,
    num_samples=100,       # interne Samples → stabiler Quantile-Schätzer
)
# forecasts.shape: (num_series, num_samples, prediction_length)

# Quantile extrahieren
low  = np.quantile(forecasts[0], 0.1, axis=0)
med  = np.quantile(forecasts[0], 0.5, axis=0)
high = np.quantile(forecasts[0], 0.9, axis=0)
```

## Direkter Vergleich: N-HiTS vs. Chronos-2

| Aspekt | N-HiTS | Chronos-2 |
|--------|--------|-----------|
| **Quantile verfügbar** | Nur mit `MQLoss` konfiguriert | Out-of-the-box (21 Quantile) |
| **Kalibrierung** | Abhängig von Trainingsdaten-Qualität | Vortrainiert auf breiten Daten — oft gut kalibriert |
| **Flexibilität** | Frei wählbare Quantile | Feste 21 Quantile (reicht meist) |
| **Rechenaufwand** | Marginal teurer als MAE-Training | `num_samples` skaliert Inferenz-Zeit |
| **Qualität** | Gut bei domänen-spezifischen Daten | Konkurrenzfähig im Zero-Shot |

**Kalibrierung** = Coverage trifft Erwartungswert: Das 90%-Intervall sollte ~90% der echten Werte enthalten. Unkalibrierte Modelle sind systematisch zu eng (overconfident) oder zu weit.

## Metriken für probabilistische Forecasts

### WQL (Weighted Quantile Loss)
```
WQL = (2/H) · Σ_t Σ_τ w_τ · [τ · max(y_t - q_τ, 0) + (1-τ) · max(q_τ - y_t, 0)]
```
- Mittlerer Quantile-Loss über alle vorhergesagten Quantile und Horizonte
- Niedrigerer WQL = besser
- Verwendet in allen grossen Foundation-Model-Benchmarks (fev-bench, GIFT-Eval)

### CRPS (Continuous Ranked Probability Score)
- Globales Mass für Verteilungsgenauigkeit
- Entspricht dem WQL wenn alle Quantile vorhergesagt werden
- Degeneriert zum MAE wenn nur Median vorhergesagt wird

### Coverage
```python
coverage_90 = np.mean((y_true >= low) & (y_true <= high))
# Sollte ≈ 0.90 sein für ein 90%-Intervall
```
Einfache Sanity-Check-Metrik. Schlechte Coverage deutet auf schlecht kalibriertes Modell hin.

### Intervallbreite
Durchschnittliche Breite des Prognoseintervalls — zeigt an wie stark das Modell "schwankt". Schmalere Intervalle bei gleicher Coverage = besser.

## Praktisches Vorgehen

```python
import pandas as pd
import matplotlib.pyplot as plt

# Backtest-Ergebnisse
cv_df = nf.cross_validation(df=train_df, n_windows=4, step_size=30)

# Coverage berechnen
cv_df['covered_90'] = (
    (cv_df['y'] >= cv_df['NHITS-q10']) &
    (cv_df['y'] <= cv_df['NHITS-q90'])
)
print(f"90%-Coverage: {cv_df['covered_90'].mean():.1%}")  # Ziel: ~90%

# Visualisierung
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(cv_df['ds'], cv_df['y'], label='Actual', color='black')
ax.plot(cv_df['ds'], cv_df['NHITS-median'], label='Median', color='blue')
ax.fill_between(
    cv_df['ds'],
    cv_df['NHITS-q10'],
    cv_df['NHITS-q90'],
    alpha=0.3, label='90% Intervall'
)
ax.legend()
```

## Verwandte Seiten

- [[N-HiTS]] — Konfiguration mit MQLoss
- [[Foundation Models für Zeitreihen]] — Chronos-2 mit 21 Quantilen out-of-the-box
- [[chronos-2-paper]] — WQL als Hauptmetrik in allen Benchmarks
- [[backtesting]] — Coverage und WQL im Cross-Validation Kontext
- [[n-hits-hyperparameter]] — `MQLoss` als Hyperparameter-Entscheidung
