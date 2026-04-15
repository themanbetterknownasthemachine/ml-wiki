---
title: "Index"
type: index
aktualisiert: 2026-04-15
---

# Wiki Index — ML & Forecasting

## Entities

| Seite | Status | Beschreibung |
|-------|--------|-------------|
| [[n-hits]] | aktuell | Deep-Learning Forecasting-Modell mit hierarchischer Interpolation |
| [[sarimax]] | aktuell | Klassisches statistisches Zeitreihen-Modell (Baseline) |
| [[foundation-models-zeitreihen]] | aktuell | Vortrainierte Modelle für Zero-Shot Forecasting (Chronos-2, TimesFM) |

## Concepts

| Seite | Status | Beschreibung |
|-------|--------|-------------|
| [[backtesting]] | aktuell | Zeitreihen-gerechte Evaluation: Expanding/Sliding Window, Metriken, Fallstricke |
| [[transformer-architektur]] | aktuell | Q/K/V Attention, RoPE, Patching — Basis von Chronos-2 und TimesFM |
| [[probabilistisches-forecasting]] | aktuell | Quantile Forecasting: wann nötig, N-HiTS MQLoss vs. Chronos-2 out-of-the-box |
| [[n-hits-hyperparameter]] | aktuell | Kritische N-HiTS Parameter, Wechselwirkungen und Optuna-Suchraum |
| [[stationaritaet]] | aktuell | Voraussetzung für SARIMAX: ADF/KPSS-Tests, Differenzierung |
| [[foundation-model-vs-nhits]] | aktuell | Entscheidungsrahmen: Foundation Model vs. N-HiTS — wann was einsetzen |
| [[hyperparameter-tuning-optuna]] | aktuell | Bayesian Optimization für ML-Hyperparameter mit Optuna |
| [[tuning-vs-baseline]] | aktuell | Entscheidungsrahmen: Wann lohnt Tuning — und wann ist die Baseline bereits optimal? |
| [[shap-explainability]] | aktuell | Erklärbarkeit von ML-Vorhersagen über Shapley-Werte |
| [[feature-engineering-zeitreihen]] | aktuell | Lag-, Kalender- und Rolling-Features für Zeitreihen |

## Decisions

| Seite | Status | Beschreibung |
|-------|--------|-------------|
| [[adr-001-nhits-statt-prophet]] | aktuell | Warum N-HiTS statt Prophet für tägliche Forecasts gewählt wurde |

## Sources

| Seite | Typ | Beschreibung |
|-------|-----|-------------|
| [[chronos-2-paper]] | paper | Chronos-2: Group Attention, multivariate Foundation Model (Amazon, 2025) |
| [[neuralforecast-nhits-docs]] | dokumentation | NeuralForecast N-HiTS: vollständige Parameterliste, exogene Feature-Typen (Nixtla) |
| [[nhits-tuning-dokumentation]] | dokumentation | N-HiTS Tuning-Experiment: Optuna-Suchraum, Warm Start, Ensemble, Modell-Persistenz |

## Runbooks

_Noch keine Runbooks erstellt._

---

## Fehlende Seiten (erwähnt aber noch nicht erstellt)

Diese Seiten werden in bestehenden Pages referenziert, haben aber noch keine eigene Seite:

- ~~[[Backtesting]]~~ → erstellt als [[backtesting]]
- [[Bias-Variance Tradeoff]] — Overfitting vs. Underfitting verstehen
- [[Gradient Descent]] — Optimierungsalgorithmus für Neural Networks
- ~~[[Transformer Architektur]]~~ → erstellt als [[transformer-architektur]]
- [[N-BEATS]] — Vorgängermodell von N-HiTS
- [[PatchTST]] — Transformer-basiertes Forecasting-Modell
- [[LightGBM Forecasting]] — Gradient Boosting für Zeitreihen
- [[Prophet]] — Facebooks Dekompositions-Modell
- [[NeuralForecast Bibliothek]] — Nixtla Framework
- [[Airflow DAG Forecasting]] — Orchestrierung
- [[Transfer Learning]] — Wissenstransfer zwischen Domänen
- ~~[[Stationarität]]~~ → erstellt als [[stationaritaet]]
- [[Informationskriterien AIC BIC]] — Modellselektion
- [[Ensemble]] — Kombination mehrerer Modelle
- [[ADR-002 Monitoring Zwei-Stufen-Ansatz]] — Monitoring-Architektur
