---
title: "SHAP Explainability"
type: concept
tags: [explainability, interpretierbarkeit, shap, ml-workflow]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---

# SHAP Explainability

## Kernidee

SHAP (SHapley Additive exPlanations) beantwortet die Frage: **"Wie viel hat jedes Feature zur Vorhersage dieses konkreten Datenpunkts beigetragen?"** — und zwar auf eine mathematisch faire Weise, basierend auf dem Shapley-Wert aus der Spieltheorie.

**Intuition**: Stell dir ein Team von 5 Leuten vor, das zusammen ein Projekt abschliesst. Wie viel hat jede Person beigetragen? Shapley-Werte lösen das Problem, indem sie alle möglichen Teamkombinationen durchspielen und den Durchschnittsbeitrag jeder Person berechnen. SHAP macht genau das mit Features statt Personen.

## Wie es funktioniert

Für jede Vorhersage berechnet SHAP einen **SHAP-Wert pro Feature**:

- Positiver SHAP-Wert → Feature hat die Vorhersage erhöht
- Negativer SHAP-Wert → Feature hat die Vorhersage gesenkt
- Summe aller SHAP-Werte = Differenz zwischen Vorhersage und Durchschnitt

Drei Visualisierungsebenen:

1. **Force Plot** (einzelne Vorhersage): Zeigt welche Features diese eine Vorhersage nach oben/unten schieben
2. **Summary Plot** (ganzes Modell): Zeigt welche Features global am wichtigsten sind und wie ihre Werte die Vorhersage beeinflussen
3. **Dependence Plot** (ein Feature): Zeigt die Beziehung zwischen einem Feature-Wert und seinem SHAP-Wert

## Wann einsetzen

- Immer bei Black-Box-Modellen ([[N-HiTS]], [[LightGBM Forecasting]], Neural Networks)
- Für Stakeholder-Kommunikation: "Warum hat das Modell X vorhergesagt?"
- Zum Debugging: wenn Vorhersagen komisch aussehen → SHAP zeigt welches Feature schuld ist
- Für Feature Selection: Features mit durchgehend null SHAP-Werten können entfernt werden

## Wann NICHT einsetzen

- Bei einfachen linearen Modellen → Koeffizienten sind direkt interpretierbar
- Bei [[SARIMAX]] → die Modellstruktur selbst ist die Erklärung
- Als alleiniges Kriterium für Feature-Wichtigkeit → SHAP zeigt Korrelation, nicht Kausalität

## Praxisbeispiel

SHAP-Analyse für ein LightGBM Forecasting-Modell:

```python
import shap
import lightgbm as lgb

# Modell trainieren
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train)

# SHAP-Werte berechnen
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot — globale Feature-Wichtigkeit
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Beeswarm Plot — Werte UND Richtung
shap.summary_plot(shap_values, X_test)

# Einzelne Vorhersage erklären (z.B. Datenpunkt 0)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

**Für [[N-HiTS]]** und andere Deep-Learning-Modelle: `shap.DeepExplainer` oder `shap.KernelExplainer` verwenden (langsamer, aber modell-agnostisch).

## Eingesetzt in

- [[N-HiTS]] — Erklärung der Forecast-Vorhersagen für Business-Stakeholder
- [[LightGBM Forecasting]] — Feature-Importance-Analyse

## Verwandte Konzepte

- [[Feature Engineering für Zeitreihen]] — Die Features, die SHAP dann erklärt
- [[Bias-Variance Tradeoff]] — SHAP hilft zu verstehen, was das Modell gelernt hat vs. Noise

## Quellen
