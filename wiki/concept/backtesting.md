---
title: "Backtesting für Zeitreihen"
type: concept
tags: [evaluation, backtesting, cross-validation, zeitreihen, forecasting]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---

# Backtesting für Zeitreihen

Backtesting ist die zeitreihen-gerechte Evaluierungsmethode für Forecast-Modelle. Sie simuliert, wie das Modell in der Vergangenheit performt hätte — mit strikter Trennung von Vergangenheit (Training) und Zukunft (Evaluation).

## Warum Standard-Kreuzvalidierung nicht funktioniert

Bei klassischer k-Fold Cross-Validation werden Daten zufällig aufgeteilt. Für Zeitreihen ist das **fatal**:

- Das Modell würde auf Daten aus der Zukunft trainieren, um Vergangenheit vorherzusagen (**Data Leakage**)
- Zeitliche Abhängigkeiten (Autokorrelation, Trend, Saisonalität) werden ignoriert
- Das Ergebnis ist zu optimistisch und in der Praxis wertlos

**Grundregel**: Der Evaluierungszeitraum muss immer **nach** dem Trainingszeitraum liegen.

## Backtesting-Strategien

### Expanding Window (Walk-Forward Validation)

```
Fold 1: Train [t=1..100]   → Test [t=101..130]
Fold 2: Train [t=1..130]   → Test [t=131..160]
Fold 3: Train [t=1..160]   → Test [t=161..190]
```

- Trainingsfenster **wächst** mit jedem Fold
- Modell sieht mit der Zeit mehr Daten — realistischer für Production
- Rechenaufwand steigt linear mit Anzahl Folds
- **Empfohlen wenn**: Modell von mehr Daten profitiert (z.B. [[N-HiTS]])

### Sliding Window (Rolling Window)

```
Fold 1: Train [t=1..100]   → Test [t=101..130]
Fold 2: Train [t=31..130]  → Test [t=131..160]
Fold 3: Train [t=61..160]  → Test [t=161..190]
```

- Trainingsfenster hat **feste Grösse**, verschiebt sich
- Fokus auf jüngste Daten — sinnvoll bei Nicht-Stationarität / Strukturbrüchen
- Rechenaufwand konstant
- **Empfohlen wenn**: Ältere Daten weniger relevant (z.B. nach Produktlaunch-Änderung)

### Einfacher Train/Val/Test Split

```
Train: t=1..600   Val: t=601..700   Test: t=701..800
```

- Für schnelle Experimente und Hyperparameter-Suche (→ [[Hyperparameter-Tuning mit Optuna]])
- Kein Aufschluss über Robustheit über verschiedene Zeiträume
- **Nachteil**: Ein schlechter Testzeitraum (z.B. Ausreisser-Periode) verzerrt die gesamte Bewertung

## Forecast-Horizont und Gap

```
Train [..........]
                  ← Gap (optional) →
                                     Test [h=1, h=2, ..., h=H]
```

- **Forecast-Horizont H**: Wie viele Schritte in die Zukunft vorhergesagt werden
- **Gap**: Lücke zwischen Training und Test (sinnvoll wenn reale Deployment-Latenz existiert)
- **Origin-Shift**: Bei mehreren Folds ändert sich der Forecast-Origin (Startpunkt der Vorhersage)

Für tägliche Forecasts mit H=30 bedeutet das: jeder Fold testet, ob das Modell die nächsten 30 Tage korrekt vorhersagt.

## Metriken

### Punkt-Metriken (Point Forecasts)

| Metrik | Formel | Eigenschaft |
|--------|--------|-------------|
| **MAE** | `mean(|y - ŷ|)` | Robust gegenüber Ausreissern, gleiche Einheit wie Zielgrösse |
| **RMSE** | `sqrt(mean((y-ŷ)²))` | Bestraft grosse Fehler stärker |
| **MAPE** | `mean(|y-ŷ|/y) × 100%` | Relativ, aber instabil bei y ≈ 0 |
| **MASE** | `MAE / MAE_Naïve` | Skalenfrei, vergleichbar über Serien; >1 = schlechter als Naïve |

**MASE** (Mean Absolute Scaled Error) ist besonders nützlich für Benchmarks über verschiedene Zeitreihen, weil er relativ zur simplen Naïve-Baseline normiert (= "was würde passieren wenn wir einfach den letzten Wert kopieren").

### Probabilistische Metriken (Quantile Forecasts)

| Metrik | Beschreibung |
|--------|-------------|
| **WQL** (Weighted Quantile Loss) | Mittlerer Quantile-Loss über alle vorhergesagten Quantile |
| **CRPS** | Continuous Ranked Probability Score — globales Mass für Verteilungsgenauigkeit |
| **Coverage** | Anteil echter Werte innerhalb des Prognoseintervalls (z.B. 90%-Intervall) |

[[Foundation Models für Zeitreihen|Chronos-2]] liefert 21 Quantile out-of-the-box → WQL direkt berechenbar. [[N-HiTS]] benötigt dafür explizite Quantile-Regression-Konfiguration.

## Implementierung mit NeuralForecast (N-HiTS)

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE

# Expanding Window Cross-Validation
nf = NeuralForecast(
    models=[NHITS(h=30, input_size=90, max_steps=500)],
    freq='D'
)

# n_windows: Anzahl Backtest-Folds
# step_size: Schrittweite zwischen Folds (in Zeitschritten)
cv_df = nf.cross_validation(
    df=train_df,
    n_windows=4,       # 4 Folds
    step_size=30,      # alle 30 Tage ein neuer Fold
    refit=True         # Expanding Window (False = Sliding Window)
)

# Metriken berechnen
from neuralforecast.losses.numpy import mae, mase
print(mae(cv_df['y'], cv_df['NHITS']))
```

## Typische Fallstricke

| Fehler | Beschreibung | Lösung |
|--------|-------------|--------|
| **Data Leakage** | Zukunfts-Features im Training (z.B. Rolling Mean über Testperiode) | Features nur auf Training-Fenster berechnen |
| **Look-Ahead Bias** | Saisonale Normalisierung über ganzen Datensatz | Normalisierung im Fold berechnen |
| **Zu wenige Folds** | 1-2 Folds sind kein stabiler Schätzer | Mindestens 4–5 Folds |
| **Zu kurze Testfenster** | Kürzer als Saisonalitätsperiode | Testfenster ≥ 1 Saison |
| **Metrik-Leakage** | Hyperparameter auf Testset optimiert | Val-Set für Tuning, Test nur zur finalen Bewertung |

## Backtesting bei Foundation Models

Bei [[Foundation Models für Zeitreihen|Foundation Models wie Chronos-2]] entfällt das Modell-Training, aber Backtesting bleibt relevant:

- **Zero-Shot Evaluation**: Modell direkt auf historische Perioden anwenden und mit [[N-HiTS]] vergleichen
- **Gleiche Folds nutzen**: Um Vergleichbarkeit zu gewährleisten, exakt dieselben Train/Test-Splits verwenden
- **In-Context Learning berücksichtigen**: Chronos-2 nutzt den Kontext-Input (historische Werte) als "implizites Training" — Kontextlänge hat Einfluss auf Performance

```python
from chronos import ChronosPipeline
import torch

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# Für jeden Backtest-Fold:
# context = historische Werte bis Fold-Startpunkt
forecasts = pipeline.predict(
    context=torch.tensor(context_values),
    prediction_length=30,
    num_samples=20,        # für probabilistische Vorhersagen
)
```

## Verbindung zu anderen Konzepten

- [[N-HiTS]] — nutzt Expanding Window, `cross_validation()` in NeuralForecast
- [[SARIMAX]] — einfache Walk-Forward-Splits für statistische Modelle
- [[Foundation Models für Zeitreihen]] — Zero-Shot Backtesting für Chronos-2/TimesFM
- [[Hyperparameter-Tuning mit Optuna]] — Val-Folds als Optimierungsziel, Test-Folds für finale Bewertung
- [[Feature Engineering für Zeitreihen]] — Features müssen fold-korrekt berechnet werden (kein Leakage)
- [[foundation-model-vs-nhits]] — Backtesting ist der Mechanismus für diesen Modellvergleich
