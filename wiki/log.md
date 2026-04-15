---
title: "Log"
type: log
---

# Wiki Log — ML & Forecasting

## [2026-04-15] query | Signale für nahes Optimum

- Frage: Welche Signale zeigen an, dass ein Modell bereits nahe am Optimum ist?
- Antwort synthetisiert aus: `concept/tuning-vs-baseline`, `source/nhits-tuning-dokumentation`
- Keine neue Seite — Frage ist Teilmenge von `concept/tuning-vs-baseline` (Abschnitt "Signale: Baseline ist bereits optimal")
- Vier Signale zusammengefasst: flache Optuna-Kurve, geringe Parameter Importance, strukturlose Residuen, domänenwissensbasierte Baseline

## [2026-04-15] query | Wann lohnt sich Hyperparameter-Tuning?

- Frage: Wann bringt Tuning Gewinn — und wann ist die Baseline bereits optimal?
- Quellen gelesen: `source/nhits-tuning-dokumentation`, `concept/hyperparameter-tuning-optuna`, `concept/n-hits-hyperparameter`, `concept/backtesting`
- Neue Concept-Seite: `concept/tuning-vs-baseline.md`
- Inhalt: Signale für/gegen Tuning, Tuning-Canary (15 Trials), Entscheidungsrahmen, Alternativen (Ensemble, Features, Daten)
- Key Insight: Flache Optuna-Kurve nach Warm Start ist das stärkste Signal für eine bereits optimale Baseline
- Index aktualisiert

## [2026-04-15] ingest | N-HiTS Tuning-Dokumentation (internes Notebook)

- Source: `raw/NHITS_Tuning_Dokumentation.md`
- Source-Seite erstellt: `wiki/source/nhits-tuning-dokumentation.md`
- `entity/n-hits.md` aktualisiert: Ensemble-Strategien, Modell-Persistenz, Windows-Workarounds, Praxiserfahrung Tuning vs. Baseline
- `concept/n-hits-hyperparameter.md` erweitert: Per-Stack-Suchraum (Ansatz B), Warm Start mit `enqueue_trial()`, zweistufige max_steps, Parameter Importance
- `concept/hyperparameter-tuning-optuna.md` erweitert: Warm Start, zweistufige max_steps, Study-Persistenz als .pkl
- `concept/feature-engineering-zeitreihen.md` erweitert: erweiterte `day_before_holiday`-Logik (3 Fälle), `is_short_week`-Feature, Business-Day-Imputation mit wochentag-spezifischem Median
- Key Finding: Optuna-Tuning (40 Trials) fand keinen Gewinn gegenüber Baseline — Baseline war bereits nahe optimal

## [2026-04-14] ingest | NeuralForecast N-HiTS Dokumentation

- Source: nixtlaverse.nixtla.io/neuralforecast/models.nhits.html
- Source-Seite erstellt: `wiki/source/neuralforecast-nhits-docs.md`
- `concept/n-hits-hyperparameter.md` korrigiert: Default n_pool_kernel_size [2,2,1], neue Parameter (pooling_mode, interpolation_mode, num_lr_decays, scaler_type, drei exogene Feature-Typen)
- `entity/n-hits.md` aktualisiert: drei exogene Feature-Typen, Quelle verlinkt
- ⚠️ Korrektur: n_pool_kernel_size Default war falsch dokumentiert ([8,4,1] statt [2,2,1])

## [2026-04-14] query | Probabilistisches Forecasting + N-HiTS Hyperparameter erstellt

- `concept/probabilistisches-forecasting.md`: Wann Quantile nötig, N-HiTS MQLoss vs. Chronos-2, WQL/CRPS/Coverage Metriken
- `concept/n-hits-hyperparameter.md`: n_pool_kernel_size, n_freq_downsample, Wechselwirkungs-Matrix, Optuna-Suchraum
- mkdocs.yml: YAML-Strukturfehler behoben (features/icon hing unter extra_css statt theme)

## [2026-04-14] query | Transformer Architektur + Stationarität Concept-Seiten erstellt

- `concept/transformer-architektur.md`: Q/K/V, Multi-Head Attention, RoPE, Patching, Group Attention (Chronos-2)
- `concept/stationaritaet.md`: ADF/KPSS-Tests, Differenzierung, Bezug zu SARIMAX vs. Deep Learning
- Index + mkdocs.yml aktualisiert

## [2026-04-14] query | Backtesting Concept-Seite erstellt

- Neue Concept-Seite: `concept/backtesting.md`
- Inhalt: Expanding vs. Sliding Window, Metriken (MAE/MASE/WQL/CRPS), Implementierung NeuralForecast + Chronos-2, Fallstricke
- Index aktualisiert: Backtesting aus "Fehlende Seiten" entfernt, Concepts ergänzt

## [2026-04-14] query | Foundation Models vs. N-HiTS Vergleich

- Frage: Was wissen wir über Foundation Models? Vergleich Chronos-2 vs. N-HiTS
- Quellen gelesen: `entity/foundation-models-zeitreihen`, `entity/n-hits`, `source/chronos-2-paper`
- Ergebnis als Concept abgelegt: `concept/foundation-model-vs-nhits.md`
- Index aktualisiert

## [2026-04-14] ingest | Chronos-2 Paper (arXiv 2510.15821)

- Source-Seite erstellt: `wiki/source/chronos-2-paper.md`
- Entity `wiki/entity/foundation-models-zeitreihen.md` aktualisiert: Status draft → aktuell, Chronos-2 Architekturdetails (Group Attention, Patches, Quantile Head, Modellgrössen), Benchmark-Tabelle ergänzt
- Index aktualisiert: Foundation Models auf aktuell gesetzt, Sources-Sektion befüllt
- Key Finding: Chronos-2 führt auf allen 3 grossen Benchmarks (fev-bench 47.3% Skill Score); grösster Vorteil bei Covariate-Tasks (+13–15 Punkte)

## [2026-04-14] init | Wiki-Grundstruktur erstellt

- Schema (CLAUDE.md) definiert: Seitentypen, Frontmatter, Workflows, Domänenregeln
- Templates erstellt: entity, concept, decision, source, runbook
- Erste Entity-Seiten: N-HiTS, SARIMAX, Foundation Models für Zeitreihen
- Erste Concept-Seiten: Hyperparameter-Tuning mit Optuna, SHAP Explainability, Feature Engineering für Zeitreihen
- Erste Decision-Seite: ADR-001 N-HiTS statt Prophet
- Index erstellt mit 7 Seiten und 15 fehlenden Seiten identifiziert
- Scope: ML/Forecasting-Wissen ohne sensible Geschäftsdaten (PoC)
