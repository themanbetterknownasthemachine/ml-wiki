---
title: "NeuralForecast N-HiTS Dokumentation"
type: source
tags: [dokumentation, n-hits, neuralforecast, nixtla, hyperparameter]
quelle_typ: dokumentation
url: "https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html"
autoren: [Nixtla]
datum: 2026-04-14
---

# NeuralForecast N-HiTS Dokumentation

**Autoren**: Nixtla · **Datum**: abgerufen 2026-04-14  
**Quelle**: [nixtlaverse.nixtla.io/neuralforecast/models.nhits.html](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html)

## Kernaussagen

1. N-HiTS ist MLP-basiert (kein Transformer) mit doppelt-residualem Stacking — Backcast- und Forecast-Outputs pro Stack
2. Drei Typen exogener Features: **futur** (zukunftsbekannt), **historisch** (nur Vergangenheit), **statisch** (zeitinvariant)
3. `pooling_mode` und `interpolation_mode` sind tunable Parameter — nicht nur `n_pool_kernel_size`/`n_freq_downsample`
4. LR-Decay via `num_lr_decays` (Standard: 3 Decays über `max_steps`)
5. `scaler_type` normalisiert Inputs intern — wichtig für gemischte Skalen

## Vollständige Parameterliste

### Architektur

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|-------------|
| `h` | int | — | Forecast-Horizont (Pflicht) |
| `input_size` | int | — | Lookback-Fenster (Pflicht) |
| `stack_types` | List[str] | `['identity','identity','identity']` | Basis-Funktion pro Stack |
| `n_blocks` | List[int] | `[1, 1, 1]` | Anzahl Blocks pro Stack |
| `mlp_units` | List[List[int]] | `[[512,512]]*3` | MLP-Schichten pro Stack |
| `n_pool_kernel_size` | List[int] | `[2, 2, 1]` | MaxPool-Fenstergrösse pro Stack |
| `n_freq_downsample` | List[int] | `[4, 2, 1]` | Output-Downsampling pro Stack |
| `pooling_mode` | str | `'MaxPool1d'` | `'MaxPool1d'` oder `'AvgPool1d'` |
| `interpolation_mode` | str | `'linear'` | `'linear'`, `'nearest'` oder `'cubic'` |
| `dropout_prob_theta` | float | `0.0` | Dropout auf Theta-Output der Stacks |
| `activation` | str | `'ReLU'` | ReLU, Softplus, Tanh, SELU, LeakyReLU, PReLU, Sigmoid |

### Training & Optimierung

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|-------------|
| `learning_rate` | float | `0.001` | Adam-Lernrate |
| `num_lr_decays` | int | `3` | Anzahl LR-Decay-Schritte über max_steps |
| `max_steps` | int | `1000` | Maximale Trainingsschritte |
| `batch_size` | int | `32` | Training Batch-Grösse |
| `windows_batch_size` | int | `1024` | Anzahl Fenster pro Batch (Sampling) |
| `valid_batch_size` | int | `None` | Validation Batch-Grösse (None = ganzes Val-Set) |
| `val_check_steps` | int | `100` | Alle N Steps auf Validation prüfen |
| `early_stop_patience_steps` | int | `-1` | Early Stopping Patience (-1 = deaktiviert) |
| `scaler_type` | str | `'identity'` | Input-Normalisierung (`'standard'`, `'robust'`, `'minmax'`, `'identity'`) |

### Exogene Features

| Parameter | Typ | Beschreibung |
|-----------|-----|-------------|
| `futr_exog_list` | List[str] | Zukunftsbekannte Variablen (Feiertage, geplante Promotionen) |
| `hist_exog_list` | List[str] | Nur historisch verfügbare Variablen (vergangene Temperaturen, etc.) |
| `stat_exog_list` | List[str] | Zeitinvariante Variablen (Produkt-Kategorie, Region) |

### Loss & Monitoring

| Parameter | Default | Optionen |
|-----------|---------|---------|
| `loss` | `MAE()` | MAE, MSE, RMSE, MAPE, SMAPE, MQLoss, HuberLoss, ... |
| `valid_loss` | wie `loss` | separater Loss für Validation möglich |
| `val_monitor` | `'ptl/val_loss'` | `'valid_loss'`, `'train_loss'` |

## Exogene Features im Detail

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

model = NHITS(
    h=30,
    input_size=90,
    # Zukunftsbekannt: wird für Forecast-Horizont mit übergeben
    futr_exog_list=['is_holiday', 'day_of_week'],
    # Nur historisch: nur im Lookback-Fenster verfügbar
    hist_exog_list=['temperature', 'competitor_price'],
    # Statisch: ein Wert pro Serie (nicht pro Zeitpunkt)
    stat_exog_list=['product_category'],
)

# train_df muss diese Spalten enthalten
# futr_df (für predict) muss futr_exog für Forecast-Horizont enthalten
nf = NeuralForecast(models=[model], freq='D')
nf.fit(df=train_df)
forecast = nf.predict(futr_df=futr_df)
```

## LR-Decay Schema

`num_lr_decays=3` mit `max_steps=1000` → LR wird bei ca. Step 333, 666 und 1000 um Faktor 10 reduziert:
```
Step 0–333:   lr = 0.001
Step 333–666: lr = 0.0001
Step 666–999: lr = 0.00001
```
Für aggressiveres Training: `num_lr_decays=0` (kein Decay) oder `num_lr_decays=1`.

## scaler_type Optionen

| Option | Beschreibung | Wann sinnvoll |
|--------|-------------|---------------|
| `'identity'` | Keine Normalisierung | Daten bereits normalisiert |
| `'standard'` | Z-Score Normalisierung | Unterschiedliche Skalen zwischen Serien |
| `'robust'` | Median/IQR-basiert | Viele Ausreisser in den Trainingsdaten |
| `'minmax'` | [0,1]-Skalierung | Bounded Outputs gewünscht |

## Relevanz für uns

- **Vollständige Parameterliste** für [[n-hits-hyperparameter|Hyperparameter-Tuning]]
- **Drei Exogene-Typen** wichtig: `futr_exog_list` für Feiertage/Promotionen, `hist_exog_list` für beobachtete externe Variablen — direkter Bezug zu [[Feature Engineering für Zeitreihen]]
- **`scaler_type='standard'`** einfache Verbesserung wenn Zeitreihen sehr unterschiedliche Skalen haben
- **`pooling_mode='AvgPool1d'`** als Alternative zu MaxPool — weniger aggressiv bei verrauschten Daten

## Aktualisierte Wiki-Seiten

- [[n-hits-hyperparameter]] — Default-Werte korrigiert, neue Parameter ergänzt
- [[n-hits]] — exogene Feature-Typen ergänzt

## Offene Fragen

- Welche `scaler_type`-Option ist optimal für unsere Daten?
- `futr_exog_list` für Feiertage: ist die Qualität besser als manuell berechnete binäre Features in [[Feature Engineering für Zeitreihen]]?
- `AvgPool1d` vs. `MaxPool1d` — lohnt ein Vergleich?
