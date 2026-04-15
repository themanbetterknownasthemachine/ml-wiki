---
title: "N-HiTS Tuning – Interne Notebook-Dokumentation"
type: source
tags: [n-hits, optuna, hyperparameter, feature-engineering, ensemble, neuralforecast]
quelle_typ: dokumentation
url: ""
autoren: []
datum: 2026-04-10
---

# N-HiTS Tuning – Interne Notebook-Dokumentation

## Zusammenfassung

Schrittweise Dokumentation eines N-HiTS Hyperparameter-Tuning-Experiments für einen täglichen Forecast in einem Logistikkontext. Das Notebook deckt den vollständigen ML-Workflow ab: Datenbereinigung, Feature Engineering, Optuna-Tuning, Analyse, Ensemble-Test, Modellpersistenz.

**Kernaussage**: Optuna-Tuning (40 Trials, TPE-Sampler) fand keinen Gewinn gegenüber der Baseline-Konfiguration — die Baseline war bereits nahe optimal. Das ist ein positives Ergebnis: Die initiale Konfiguration war robuster als erwartet.

## Kontext

- **Forecast-Aufgabe**: Tagesgenaue Prognose einer betrieblichen Mengengrösse in einem Logistikbetrieb
- **Horizont**: 60 Arbeitstage (~3 Monate)
- **Input Size**: 120 Arbeitstage (~6 Monate Lookback)
- **Baseline**: Bestehendes N-HiTS-Modell aus einem vorangehenden Modellvergleich
- **Ziel**: Systematische Verbesserung durch Hyperparameter-Suche mit Optuna

## Datenaufbereitung

### Bereinigungsschritte

```python
df = df[df["tonnage"] > 0]           # Nullwerte = fehlende Daten
df = df[df.index.dayofweek < 5]       # Nur Arbeitstage (Mo–Fr)
# Jahresende 21.–31. Dez ausschliessen (Ausreisser durch Saisonspitzen)
year_end_mask = [(d.month == 12 and d.day >= 21) for d in df.index]
df = df[~year_end_mask]
```

**Warum Jahresende ausschliessen?** Die Weihnachtsperiode enthält starke Ausreisser und unregelmässige Betriebszeiten. Das Modell wird auf Normalperioden optimiert. Für die Weihnachtsperiode wird kein Modell eingesetzt.

### Business-Day-Lücken füllen

NeuralForecast erwartet einen lückenlosen Business-Day-Index. Fehlende Tage (Feiertage, Betriebsschliessungen) werden mit wochentag-spezifischen Medianwerten aufgefüllt:

```python
full_bday_idx = pd.bdate_range(df.index.min(), df.index.max(), freq="B")
df_full = df[["tonnage"]].reindex(full_bday_idx)

weekday_medians = df.groupby(df.index.dayofweek)["tonnage"].median()
df_full["tonnage"] = df_full["tonnage"].fillna(
    df_full.index.to_series().map(lambda d: weekday_medians[d.dayofweek])
)
```

**Warum wochentag-spezifisch?** Montage haben systematisch andere Werte als Freitage. Ein globaler Median würde an einem Montag-Feiertag einen verzerrten Wert einsetzen. Evaluation erfolgt ausschliesslich auf den echten, ursprünglichen Arbeitstagen.

## Feature Engineering

12 binäre Kalender-Features (alle `futr_exog` — zum Vorhersagezeitpunkt aus dem Kalender berechenbar):

| Feature | Erklärung |
|---------|-----------|
| `is_monday` bis `is_friday` | Wochentags-Flags; starker Einfluss auf tägliche Mengen |
| `day_before_holiday` | Vortag vor Schliesstag → oft erhöhte Menge (Kompensationseffekt) |
| `is_short_week` | Woche mit Feiertag → weniger Arbeitstage, höhere Tagesmenge |
| `is_july` | Ferienmonat → systematisch tiefere Mengen |
| `is_first_bday_of_month` | Monatsanfang-Muster (Logistik-Zyklen) |
| `is_last_bday_of_month` | Monatsende-Muster (Logistik-Zyklen) |
| `is_karwoche_dienstag` | Karwoche-Dienstag: erhöhte Menge vor kurzem Karfreitag |
| `is_tag_vor_1mai` | Letzter Arbeitstag vor 1. Mai (Feiertag in gewissen Regionen) |

### Komplexe Feiertagslogik

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

### Feiertags-Differenzierung (Betrieb offen vs. geschlossen)

Ein nationaler Feiertag muss nicht bedeuten, dass der Betrieb geschlossen ist. Daher zwei separate Feiertagssets:
- `ch_holidays_closed`: Nur echte Schliessungstage (z. B. ohne Auffahrt, falls Betrieb offen)
- `ch_holidays_all`: Alle Feiertage (für `day_before_holiday`-Logik)

```python
# Auffahrt ist national, aber Betrieb offen → aus "closed" entfernen
ch_holidays_closed = {d: n for d, n in national.items() if "Ascension Day" not in n}
```

## Optuna Hyperparameter-Tuning

### Suchraum

| Parameter | Typ | Wertebereich |
|-----------|-----|-------------|
| `pool_k_0/1/2` | kategorisch | `[1,2,4,8,16]` / `[1,2,4,8]` / `[1,2]` |
| `freq_d_0/1/2` | kategorisch | `[4,8,16,24]` / `[2,4,8]` / `[1,2]` |
| `n_blocks_0/1/2` | integer | 1–4 |
| `lr` | float (log) | 1e-4 bis 1e-2 |
| `scaler` | kategorisch | `robust`, `standard`, `minmax` |
| `dropout` | float | 0.0–0.3 |
| `interp_mode` | kategorisch | `linear`, `nearest` |
| `max_steps` | kategorisch | 200, 300, 400 (Tuning) → 500 (finales Modell) |

**Wichtig**: Per-Stack-Parameter werden individuell gesucht (nicht als gemeinsame kategorische Liste). `lr` logarithmisch skaliert.

### Warm Start

```python
study.enqueue_trial({
    "pool_k_0": 4, "pool_k_1": 4, "pool_k_2": 1,
    "freq_d_0": 8, "freq_d_1": 2, "freq_d_2": 1,
    "lr": 1e-3, "scaler": "robust", "dropout": 0.0,
    "interp_mode": "linear", "max_steps": 400,
})
```

Der erste Trial ist die bekannte Baseline-Konfiguration — gibt Optuna sofort einen guten Referenzpunkt.

### Zweistufige max_steps

- **Tuning-Trials**: `max_steps ∈ [200, 300, 400]`, `early_stop_patience_steps=5` → Geschwindigkeit
- **Finales Modell**: `max_steps=500`, `early_stop_patience_steps=10` → Stabilität

### Analyse-Plots

1. Optimierungsverlauf (MAE pro Trial)
2. Parameter Importance (Fano-Importance via Random Forest)
3. Parallel Coordinates (alle Parameter gleichzeitig)
4. Contour-Plots (Parameterinteraktionen)

## Finales Training

```python
nhits_best = NHITS(
    h=HORIZON,
    input_size=INPUT_SIZE,
    max_steps=500,                    # volle Steps
    early_stop_patience_steps=10,     # stabiler als 5
    val_check_steps=50,
    random_seed=42,                   # Reproduzierbarkeit
    **best_params_from_optuna,
)
nf_best.fit(df=nf_df_train, val_size=60)  # letzte 60 Tage als internes Val-Set
```

**Evaluation**: Nur auf echten Arbeitstagen gefiltert (`original_dates_set`).

**Wochentags-Analyse**: MAE und Bias separat pro Wochentag — zeigt systematische Über-/Unterschätzungen:
- `Bias > 0`: Modell unterschätzt
- `Bias < 0`: Modell überschätzt

## Ensemble-Test

Drei Modelle kombiniert: [[N-HiTS]] + PatchTST + TSMixerx

| Strategie | Formel | Eigenschaft |
|-----------|--------|-------------|
| Gleichgewichtet | `(A + B + C) / 3` | Einfach, robust |
| MAE-gewichtet | `Σ(wᵢ * Mᵢ) / Σwᵢ` mit `wᵢ = 1/MAEᵢ` | Bessere Modelle dominieren |
| Median | `median(A, B, C)` | Robustester Ansatz, resistent gegen Ausreisser |

**Empfehlung**: Median-Ensemble für maximale Robustheit.

## Modell-Persistenz

```python
# Modell speichern
nf_best.save(path=MODEL_DIR, overwrite=True)

# Optuna Study speichern (für spätere Analyse/Fortsetzung)
with open("nhits_study.pkl", "wb") as f:
    pickle.dump(study, f)

# Modell laden
nf_loaded = NeuralForecast.load(path=MODEL_DIR)
```

## Windows-spezifische Hinweise

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Zwei OpenMP-Instanzen (conda + PyTorch)
torch.set_num_threads(1)                       # Deadlock-Vermeidung auf Windows
```

## Key Takeaways

1. Optuna-Tuning bestätigt Baseline → nicht immer bringt Tuning Verbesserungen
2. Warm Start mit `enqueue_trial()` spart Trials für Random-Exploration
3. Zweistufige max_steps (Tuning vs. finales Modell) ist bewährtes Pattern
4. `is_short_week` und differenzierte `day_before_holiday`-Logik haben starken Einfluss
5. Business-Day-Imputation mit wochentag-spezifischem Median ist wichtig für NeuralForecast
6. Optuna-Study als `.pkl` persistieren → kann für Warm-Restart oder nachträgliche Analyse genutzt werden

## Verknüpfte Wiki-Seiten

- [[N-HiTS]] — Entity-Seite
- [[n-hits-hyperparameter]] — Hyperparameter-Details
- [[hyperparameter-tuning-optuna]] — Optuna Framework
- [[feature-engineering-zeitreihen]] — Feature Engineering Patterns
- [[backtesting]] — Evaluation-Strategie
