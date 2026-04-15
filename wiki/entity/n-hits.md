---
title: "N-HiTS"
type: entity
tags: [forecasting, deep-learning, neuralforecast, zeitreihen]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-15
quellen: 2
---

# N-HiTS

## Überblick

N-HiTS (Neural Hierarchical Interpolation for Time Series) ist ein Deep-Learning-Modell für Zeitreihen-Forecasting, das durch hierarchische Zerlegung der Zeitreihe in verschiedene Frequenzkomponenten besonders gut mit Multi-Step-Forecasts umgehen kann.

## Architektur / Funktionsweise

N-HiTS baut auf dem älteren [[N-BEATS]] Modell auf und löst dessen Hauptproblem: bei langen Forecast-Horizonten musste N-BEATS jeden einzelnen Zeitpunkt vorhersagen, was rechenintensiv und fehleranfällig war.

Die Kernidee von N-HiTS ist **hierarchische Interpolation**:

- Das Modell besteht aus mehreren **Stacks**, wobei jeder Stack auf eine andere Frequenz/Granularität der Zeitreihe fokussiert.
- Jeder Stack hat einen eigenen **Expressiveness Ratio** — er gibt nur wenige Koeffizienten aus (z.B. 4 statt 30), die dann auf den vollen Forecast-Horizont hochinterpoliert werden.
- Stack 1 lernt den groben Trend (wenige Koeffizienten → lange, glatte Kurve), Stack 2 lernt mittlere Schwankungen, Stack 3 lernt feine Details.
- Die Stacks arbeiten **subtraktiv**: jeder Stack lernt aus dem Residuum des vorherigen.

Implementiert in der **NeuralForecast**-Bibliothek von Nixtla.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

models = [
    NHITS(
        h=30,                    # Forecast-Horizont (z.B. 30 Tage)
        input_size=90,           # Lookback-Window
        n_pool_kernel_size=[8, 4, 1],   # Pooling pro Stack
        n_freq_downsample=[4, 2, 1],    # Downsampling pro Stack
        max_steps=1000,
        learning_rate=1e-3,
        batch_size=32,
    )
]

nf = NeuralForecast(models=models, freq='D')
nf.fit(df=train_df)
forecast = nf.predict()
```

## Eingesetzte Konzepte

- [[Hierarchische Zerlegung]] — Kern-Architekturprinzip
- [[Hyperparameter-Tuning mit Optuna]] — zur Optimierung der Modellkonfiguration
- [[Feature Engineering für Zeitreihen]] — Exogene Features als Input
- [[SHAP Explainability]] — zur Erklärung der Vorhersagen
- [[Backtesting]] — zur robusten Evaluation

## Stärken

- Sehr gute Performance auf täglichen Forecasts mit mehrtägigem Horizont
- Effizient durch Interpolation (weniger Parameter als N-BEATS bei gleichem Horizont)
- Kann drei Typen exogener Features nutzen: **futr** (zukunftsbekannt), **hist** (nur Vergangenheit), **stat** (zeitinvariant pro Serie)
- Gut trainierbar auf mittleren Datenmengen (einige hundert bis tausende Beobachtungen)
- Multi-Series Training möglich (ein Modell für mehrere verwandte Zeitreihen)

## Schwächen / Limitierungen

- Benötigt mehr Daten als statistische Modelle wie [[SARIMAX]] (mind. ~2 Jahre tägliche Daten für stabile Ergebnisse)
- Weniger interpretierbar als klassische Modelle — daher [[SHAP Explainability]] wichtig
- Hyperparameter-Sensitivität: `n_pool_kernel_size` und `n_freq_downsample` müssen sorgfältig gewählt werden (→ [[Hyperparameter-Tuning mit Optuna]])
- Kein Unsicherheitsintervall out-of-the-box (Quantile Regression möglich, aber erfordert Konfiguration)
- Kann bei sehr kurzen Zeitreihen oder starken Strukturbrüchen instabil werden

## Verwandte Entities

- [[SARIMAX]] — Klassische Alternative für Zeitreihen-Forecasting
- [[PatchTST]] — Transformer-basierte Alternative, im Ensemble getestet
- [[Foundation Models für Zeitreihen]] — Potenzielle nächste Generation (Chronos-2, TimesFM)
- [[NeuralForecast Bibliothek]] — Implementierungs-Framework
- [[Airflow DAG Forecasting]] — Orchestrierung des Trainings- und Inference-Prozesses

## Ensemble

N-HiTS kann mit anderen Modellen (z.B. [[PatchTST]], TSMixerx) zu einem Ensemble kombiniert werden. Drei Strategien:

| Strategie | Formel | Wann sinnvoll |
|-----------|--------|---------------|
| Gleichgewichtet | `(A + B + C) / 3` | Einfacher Einstieg |
| MAE-gewichtet | `Σ(wᵢ * Mᵢ)` mit `wᵢ = 1/MAEᵢ` | Wenn Modelle klar unterschiedliche Qualität haben |
| Median | `median(A, B, C)` | Empfohlen — robust gegen Ausreisser in einem Modell |

## Modell-Persistenz

```python
# Speichern
nf.save(path="models/nhits_v3/", overwrite=True)

# Laden für neue Prognosen (kein Retraining nötig)
nf_loaded = NeuralForecast.load(path="models/nhits_v3/")
preds = nf_loaded.predict(futr_df=future_features_df)
```

Versionierte Verzeichnisnamen (z.B. `nhits_v3_20260410`) ermöglichen Rollback auf frühere Modelle.

## Praxiserfahrung: Tuning vs. Baseline

In einem durchgeführten Optuna-Tuning (40 Trials, TPE-Sampler) brachte die Suche **keinen Gewinn** gegenüber der bestehenden Baseline-Konfiguration. Interpretation: Die initiale Konfiguration war bereits nahe optimal. Tuning lohnt sich mehr, wenn das Modell noch nicht auf die Zeitreihen-Charakteristik angepasst wurde.

## Windows-spezifische Hinweise

```python
import os, torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Zwei OpenMP-Instanzen (conda + PyTorch)
torch.set_num_threads(1)                       # Deadlock-Vermeidung auf Windows
```

## Offene Fragen / Nächste Schritte

- Evaluation von [[Foundation Models für Zeitreihen]] als Ergänzung/Ablösung
- Quantile Regression für Unsicherheitsintervalle implementieren
- Multivariate Erweiterung: zusätzliche exogene Features evaluieren
- Cross-Validation-Strategie verfeinern (aktuell: Expanding Window)

## Quellen

- [[neuralforecast-nhits-docs]] — Offizielle NeuralForecast Dokumentation: vollständige Parameterliste, exogene Feature-Typen, scaler_type
- [[nhits-tuning-dokumentation]] — Praxisdokumentation: Tuning-Ergebnis, Ensemble-Strategien, Modell-Persistenz, Windows-Workarounds
