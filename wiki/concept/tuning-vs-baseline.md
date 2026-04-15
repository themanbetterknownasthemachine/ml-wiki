---
title: "Wann lohnt sich Hyperparameter-Tuning?"
type: concept
tags: [hyperparameter, optuna, tuning, modellwahl, ml-workflow]
status: aktuell
erstellt: 2026-04-15
aktualisiert: 2026-04-15
---

# Wann lohnt sich Hyperparameter-Tuning?

## Kernfrage

Hyperparameter-Tuning ist teuer (Rechenzeit, Komplexität). Die Frage ist nicht "wie tune ich?" (→ [[hyperparameter-tuning-optuna]]), sondern **wann** es überhaupt Sinn macht — und wann die Baseline bereits ausreicht.

Die Antwort ist nicht trivial: Ein systematisches Optuna-Tuning ([40 Trials, TPE-Sampler](nhits-tuning-dokumentation)) kann ergeben, dass die Baseline-Konfiguration bereits nahe am Optimum lag und kein Gewinn erzielt wurde. Das ist ein legitimes und wertvolles Ergebnis.

## Signale: Tuning lohnt sich

### 1. Default-Parameter wurden noch nie angepasst

Bibliotheks-Defaults sind generisch, nicht domänenspezifisch. Beispiel [[N-HiTS]]:

```
Default: n_pool_kernel_size=[2, 2, 1]
```

Für eine Zeitreihe mit starker Wochensaisonalität ist `[2, 2, 1]` fast sicher suboptimal — Stack 0 sieht nach Pooling kaum mehr Datenpunkte. Wurde die Baseline mit Defaults gebaut, lohnt Tuning fast immer.

### 2. Steile Optuna-Optimierungskurve in frühen Trials

Wenn der MAE nach den ersten 5–10 Trials deutlich unter den Ausgangswert fällt, signalisiert das: der Suchraum ist noch nicht ausgeschöpft, Tuning bringt Gewinn.

```
Trial 0 (Zufall):   MAE = 18.2
Trial 3 (Zufall):   MAE = 14.5   ← starke Verbesserung → weitertunen
Trial 10 (TPE):     MAE = 13.1
...
```

### 3. Hohe Parameter Importance

```python
importances = optuna.importance.get_param_importances(study)
```

Wenn ein oder zwei Parameter hohe Importance haben (z.B. `learning_rate`, `pool_k_0`), gibt es noch viel zu gewinnen durch gezieltes Feintuning in diesen Dimensionen.

### 4. Systematische Residuen / Wochentags-Bias

Wenn die Wochentags-Analyse zeigt, dass das Modell z.B. jeden Montag systematisch unterschätzt (Bias >> 0), hat das Modell ein strukturelles Problem. Tuning kann helfen — oder es fehlt ein Feature.

```python
# Bias pro Wochentag prüfen bevor man tunet:
for wd in range(5):
    mask = [d.weekday() == wd for d in dates]
    bias = np.mean(actual[mask] - pred[mask])
    print(f"{'MTWRF'[wd]}: Bias = {bias:+.1f}")
```

### 5. Erste Tuning-Runde überhaupt

Das erste Tuning gibt immer Erkenntnisse — selbst wenn kein Gewinn entsteht: Parameter Importance zeigt, was zählt und was vernachlässigt werden kann.

---

## Signale: Baseline ist bereits optimal

### 1. Flache Optimierungskurve nach Warm Start

```
Trial 0 (Baseline): MAE = 12.5   ← bester Wert bleibt Trial 0
Trial 5 (TPE):      MAE = 12.7
Trial 15 (TPE):     MAE = 12.4
Trial 30 (TPE):     MAE = 12.6
```

Wenn die Kurve nach dem Warm-Start-Trial nicht mehr deutlich fällt und die TPE-Verbesserungen im Bereich des Messrauschens liegen, ist das Optimum nahe.

**Empfehlung**: Nach 15–20 Trials ohne >2–3% Verbesserung → Stop.

### 2. Geringe Parameter Importance überall

Wenn alle Parameter geringe Importance haben, ist die Verlustlandschaft relativ flach — kleine Änderungen haben kaum Effekt. Das Modell ist robust gegenüber der Parameterwahl.

### 3. Baseline basiert auf Domänenwissen

Eine Baseline, die bewusst auf die Zeitreihe abgestimmt wurde, ist schwer zu schlagen:

| Eigenschaft | Zeichen für gut kalibrierte Baseline |
|-------------|--------------------------------------|
| `n_pool_kernel_size` | Abstimmung auf Saisonalität (z.B. `[4,4,1]` für Wochenmuster + Details) |
| `input_size` | ≥ `2 * horizon`, bei Jahressaisonalität ≥ `4 * horizon` |
| `scaler_type` | `'robust'` bei verrauschten Daten mit Ausreissern |
| `learning_rate` | 1e-3 (bewährt für Adam bei Daily-Forecasting) |

### 4. Residualanalyse zeigt keine Struktur

Wenn Residuen symmetrisch und unkorreliert sind — kein Wochentags-Bias, kein Trend in den Fehlern — arbeitet das Modell bereits gut. Mehr Tuning findet dann nur noch Rauschen.

---

## Entscheidungsrahmen

```
Baseline vorhanden?
├── Nein → Default-Parameter? → Tuning fast immer sinnvoll
└── Ja
    ├── Residualanalyse: Systematischer Bias?
    │   ├── Ja → erst Features prüfen, dann Tuning
    │   └── Nein → Tuning-Canary starten (15 Trials)
    │       ├── Kurve flach / Baseline bleibt beste → Stop
    │       │   └── Ensemble als nächster Schritt
    │       └── Kurve fällt → weitertunen (40–100 Trials)
```

### Tuning-Canary: 15 Trials als Vorstudie

Bevor man 40+ Trials startet, lohnt sich ein Mini-Lauf:

```python
study.enqueue_trial(baseline_params)  # Warm Start
study.optimize(objective, n_trials=15)

# Auswertung:
best_improvement = (baseline_mae - study.best_value) / baseline_mae
if best_improvement < 0.02:  # weniger als 2% Verbesserung
    print("Baseline nahe optimal — Tuning wahrscheinlich nicht lohnend")
```

---

## Alternativen zu mehr Tuning

Wenn Tuning keinen Gewinn bringt, gibt es drei sinnvolle nächste Schritte:

### 1. Ensemble (oft besser als weiteres Tuning desselben Modells)

```python
# Median-Ensemble: robust gegen Ausreisser in einem Modell
ens_median = np.median([nhits_preds, patchtst_preds, tsm_preds], axis=0)
```

Verschiedene Modellarchitekturen machen unterschiedliche Fehler — ein Ensemble gleicht diese aus, ohne das Einzelmodell weiter zu optimieren. → [[N-HiTS]] Ensemble-Sektion.

### 2. Feature Engineering verbessern

Systematische Residuen deuten oft auf ein fehlendes Feature hin, nicht auf falsche Hyperparameter:

- Bias an bestimmten Wochentagen → Wochentags-Feature fehlt oder ist falsch
- Saisonales Muster in Residuen → Jahressaison-Feature ergänzen (z.B. `is_july`)
- Spike vor Feiertagen → `day_before_holiday`-Logik verfeinern

→ [[feature-engineering-zeitreihen]]

### 3. Mehr Trainingsdaten

Bei kleinen Datensätzen findet Tuning Rauschen, nicht Signal. Faustregel: Mindestens 3–5 vollständige Saisonalitätszyklen in den Trainingsdaten, bevor Tuning zuverlässige Ergebnisse liefert.

---

## Zusammenfassung

| Situation | Empfehlung |
|-----------|-----------|
| Erste Konfiguration, Default-Parameter | Tuning sinnvoll |
| Basis gut kalibriert, geringe Importance | Stop — Baseline ist optimal |
| Systematische Residuen (Wochentags-Bias) | Erst Features prüfen, dann tunen |
| Kein Einzelmodell-Gewinn mehr | Ensemble (Median) als nächster Schritt |
| Wenig Daten (<2 Jahre täglich) | Tuning vorsichtig — Overfitting-Risiko |

## Verwandte Konzepte

- [[hyperparameter-tuning-optuna]] — Wie man Optuna konkret einsetzt
- [[n-hits-hyperparameter]] — N-HiTS-spezifischer Suchraum
- [[feature-engineering-zeitreihen]] — Alternative zu Tuning bei Residual-Mustern
- [[backtesting]] — Evaluation muss tuning-sicher sein (kein Leakage)
- [[probabilistisches-forecasting]] — Wechsel auf MQLoss kann eine eigene Tuning-Dimension öffnen

## Quellen

- [[nhits-tuning-dokumentation]] — Praxisbeispiel: 40 Trials, kein Gewinn, Baseline war optimal
- [[hyperparameter-tuning-optuna]] — Konzeptionelle Grundlagen
