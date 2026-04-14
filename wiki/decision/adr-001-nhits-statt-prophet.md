---
title: "ADR-001: N-HiTS statt Prophet für tägliche Forecasts"
type: decision
tags: [modellwahl, forecasting, architektur]
status: aktuell
entschieden: 2024-06-01
beteiligte: [ML-Spezialist, BI-Teamlead]
---

# ADR-001: N-HiTS statt Prophet für tägliche Forecasts

## Kontext

Für tägliche Mengenprognosen in der Logistik wurde ein Forecasting-Modell benötigt, das zuverlässig 30 Tage in die Zukunft vorhersagen kann. Die Anforderungen: tägliche Granularität, Berücksichtigung von Wochentags- und Feiertagsmustern, Integration in bestehende Infrastruktur (Snowflake, Airflow, Power BI), und niedrige MAE als primäre Metrik.

## Entscheidung

**N-HiTS** wurde als primäres Forecasting-Modell gewählt, implementiert über die NeuralForecast-Bibliothek von Nixtla.

## Alternativen

### Option A: Prophet (Facebook/Meta)
- **Vorteile**: Einfache API, eingebaute Feiertags- und Saisonalitäts-Modellierung, gute Uncertainty Intervals, weit verbreitet → viel Community-Support
- **Nachteile**: Schwächer bei komplexen nichtlinearen Mustern, limitierte Anpassbarkeit der Saisonalitäts-Dekomposition, für tägliche Daten mit starker Wochensaisonalität oft schlechter als spezialisierte Modelle, Entwicklung wurde von Meta verlangsamt

### Option B: N-HiTS (NeuralForecast)
- **Vorteile**: Sehr gute Performance bei täglichen Multi-Step-Forecasts (State of the Art auf M4-Benchmark), hierarchische Zerlegung passt konzeptuell zu den Daten (überlagerte Wochen-/Monats-/Jahresmuster), Python-native → passt in unseren Stack, aktive Entwicklung durch Nixtla
- **Nachteile**: Weniger interpretierbar (→ [[SHAP Explainability]] als Mitigation), braucht mehr Daten als Prophet, Hyperparameter-Tuning nötig (→ [[Hyperparameter-Tuning mit Optuna]])

### Option C: SARIMAX (Baseline)
- **Vorteile**: Höchste Interpretierbarkeit, bewährt bei monatlichen Forecasts
- **Nachteile**: Deutlich schlechter bei täglicher Granularität und langem Horizont, kann nur eine Serie pro Modell, manuelle Parameterauswahl

### Option D: LightGBM mit Lag-Features
- **Vorteile**: Schnell, robust, gutes Feature Engineering möglich
- **Nachteile**: Kein natives Multi-Step-Forecasting (recursive Strategy nötig → Error Accumulation), Feature Engineering manuell

## Begründung

Ausschlaggebend waren:

1. **Metrik**: In Backtest-Vergleichen über mehrere Monate erzielte N-HiTS konsistent die niedrigste MAE auf 30-Tage-Horizont, vor LightGBM, Prophet und SARIMAX.
2. **Architektur-Fit**: Die hierarchische Zerlegung in N-HiTS (verschiedene Stacks für verschiedene Frequenzen) passt konzeptuell gut zu Logistik-Zeitreihen, die überlagerte Wochen-, Monats- und Jahresmuster zeigen.
3. **Stack-Kompatibilität**: NeuralForecast ist Python-native und integriert sich nahtlos in den bestehenden Airflow/Snowflake-Stack.
4. **Zukunftsfähigkeit**: NeuralForecast bietet auch [[PatchTST]], [[Foundation Models für Zeitreihen|Foundation Models]] etc. — Modellwechsel ohne Infrastruktur-Umbau möglich.

## Konsequenzen

- [[SHAP Explainability]] muss eingebaut werden, da das Modell selbst nicht interpretierbar ist
- [[Hyperparameter-Tuning mit Optuna]] ist erforderlich für optimale Performance
- [[SARIMAX]] bleibt als Baseline erhalten (Sanity Check: wenn N-HiTS schlechter als SARIMAX → etwas stimmt nicht)
- Monitoring muss robuster sein als bei statistischen Modellen, da Neural Networks stille Degradation zeigen können
- Ein [[Ensemble]] aus N-HiTS und PatchTST wurde getestet, brachte aber keinen signifikanten Vorteil (N-HiTS dominierte das Gewicht)

## Verwandte Entscheidungen

- [[ADR-002 Monitoring Zwei-Stufen-Ansatz]] — Folgeentscheidung zur Überwachung

## Betroffene Entities

- [[N-HiTS]]
- [[SARIMAX]]
- [[PatchTST]]
