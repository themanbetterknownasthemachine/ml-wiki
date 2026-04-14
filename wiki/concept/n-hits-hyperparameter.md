---
title: "N-HiTS Hyperparameter"
type: concept
tags: [n-hits, hyperparameter, tuning, neuralforecast, forecasting]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
quellen: 1
---

# N-HiTS Hyperparameter

[[N-HiTS]] hat mehrere stark interagierende Hyperparameter. Diese Seite erklärt was jeder Parameter steuert, welche Wechselwirkungen kritisch sind, und wie man sinnvoll mit [[Hyperparameter-Tuning mit Optuna|Optuna]] sucht.

## Überblick aller Parameter

```python
NHITS(
    # --- Pflicht ---
    h=30,                              # Forecast-Horizont
    input_size=90,                     # Lookback-Fenster

    # --- Architektur (kritisch) ---
    n_pool_kernel_size=[2, 2, 1],      # Default lt. Docs; anpassen an Saisonalität
    n_freq_downsample=[4, 2, 1],       # Output-Downsampling pro Stack
    pooling_mode='MaxPool1d',          # oder 'AvgPool1d'
    interpolation_mode='linear',       # 'linear', 'nearest', 'cubic'
    stack_types=['identity'] * 3,      # Basis-Funktion pro Stack
    n_blocks=[1, 1, 1],                # Anzahl Blocks pro Stack
    mlp_units=[[512, 512]] * 3,        # MLP-Grösse pro Stack
    activation='ReLU',                 # ReLU, Softplus, Tanh, SELU, ...

    # --- Exogene Features ---
    futr_exog_list=['is_holiday'],     # zukunftsbekannte Variablen
    hist_exog_list=['temperature'],    # nur historisch verfügbar
    stat_exog_list=['category'],       # zeitinvariant pro Serie

    # --- Training ---
    max_steps=1000,
    learning_rate=1e-3,
    num_lr_decays=3,                   # LR-Decay-Schritte über max_steps
    batch_size=32,
    windows_batch_size=1024,           # Fenster-Sampling pro Batch
    scaler_type='identity',            # 'standard', 'robust', 'minmax', 'identity'
    loss=MAE(),                        # oder MQLoss für Quantile

    # --- Regularisierung ---
    dropout_prob_theta=0.0,
    val_check_steps=100,
    early_stop_patience_steps=-1,      # -1 = kein Early Stopping
)
```

> ⚠️ Korrektur: Default `n_pool_kernel_size` ist laut offizieller Dokumentation `[2, 2, 1]`, nicht `[8, 4, 1]`. Für domänen-spezifische Saisonalitäten sollte der Wert explizit gesetzt werden.

## Die drei kritischsten Parameter

### `n_pool_kernel_size` — Was wird in jedem Stack betrachtet?

Steuert das **Pooling** (MaxPool oder AvgPool, je nach `pooling_mode`) vor jedem Stack. Grössere Kernel = Stack sieht gröbere, geglättete Version der Zeitreihe. **Default ist `[2, 2, 1]`** — für echte Saisonalitäten fast immer zu klein.

```
input_size=90, n_pool_kernel_size=[8, 4, 1]:

Stack 1 (Pool=8): Sieht 90/8 ≈ 11 Datenpunkte → lernt groben Trend
Stack 2 (Pool=4): Sieht 90/4 ≈ 22 Datenpunkte → lernt mittelfristige Muster
Stack 3 (Pool=1): Sieht alle 90 Punkte         → lernt kurzfristige Details
```

**Faustregel**: Absteigend von der Saisonalitätslänge bis 1.
- Tägliche Daten, Wochenmuster (s=7): `[7, 3, 1]` oder `[8, 4, 1]`
- Tägliche Daten, Jahresmuster (s=365): `[30, 7, 1]` oder `[52, 12, 1]`

**`pooling_mode`**: `'MaxPool1d'` (aggressiv, robust gegen Spikes) vs. `'AvgPool1d'` (weicher, besser bei verrauschten Daten).

### `n_freq_downsample` — Wie viele Koeffizienten gibt jeder Stack aus?

Steuert die **Interpolation**: Jeder Stack gibt `h / n_freq_downsample[i]` Koeffizienten aus, die dann auf Horizont `h` interpoliert werden.

```
h=30, n_freq_downsample=[4, 2, 1]:

Stack 1: gibt 30/4 = 7.5 ≈ 8 Koeffizienten aus → glatte, grobe Kurve
Stack 2: gibt 30/2 = 15 Koeffizienten aus      → mittlere Auflösung
Stack 3: gibt 30/1 = 30 Koeffizienten aus      → volle Auflösung
```

**Wechselwirkung mit `n_pool_kernel_size`**: Konsistenz ist wichtig — grösseres Pooling sollte mit grösserem Downsampling kombiniert werden:
```
n_pool_kernel_size=[8, 4, 1]  ←→  n_freq_downsample=[4, 2, 1]  ✓
n_pool_kernel_size=[8, 4, 1]  ←→  n_freq_downsample=[1, 1, 1]  ✗ (Stack 1 sieht grob aber gibt fein aus)
```

### `input_size` — Wie viel Vergangenheit sieht das Modell?

```
input_size sollte ≥ h sein (mindestens einen Forecast-Horizont zurückblicken)
input_size = 2*h bis 5*h ist typisch
```

**Wechselwirkung**: Grosse `n_pool_kernel_size` mit kleinem `input_size` → Stack sieht nur wenige Datenpunkte nach dem Pooling. Faustregel: `input_size / max(n_pool_kernel_size) ≥ 5`.

## Trainings-Hyperparameter

### `max_steps` vs. Early Stopping

```python
# Option A: Feste Steps
NHITS(max_steps=1000)

# Option B: Early Stopping
NHITS(
    max_steps=5000,
    val_check_steps=100,           # alle 100 Steps auf Val prüfen
    early_stop_patience_steps=10,  # stoppe nach 10 mal keine Verbesserung
)
```

Early Stopping ist robuster gegen Overfitting, verlängert aber den Tuning-Prozess (da das Modell unterschiedlich lange trainiert).

### `learning_rate` und `num_lr_decays`

Typischer Bereich: `1e-4` bis `1e-2`. Interagiert mit `batch_size`:
- Grosses `batch_size` + hohes `lr` = instabiles Training
- Kleines `batch_size` + niedriges `lr` = langsam aber stabil

**`num_lr_decays=3`** (Default): LR wird 3× über `max_steps` um Faktor 10 reduziert:
```
max_steps=1000, num_lr_decays=3:
Step 0–333:   lr = 1e-3
Step 333–666: lr = 1e-4
Step 666–999: lr = 1e-5
```
Für stabiles Training mit `early_stop_patience_steps` empfiehlt sich `num_lr_decays=0` oder `1`, da der Decay-Zeitpunkt nicht mehr mit der tatsächlichen Trainings-Dauer synchronisiert ist.

### `scaler_type` — Interne Input-Normalisierung

| Option | Wann sinnvoll |
|--------|---------------|
| `'identity'` (Default) | Daten bereits normalisiert oder gleiche Skala |
| `'standard'` | Zeitreihen mit sehr unterschiedlichen Niveaus (Multi-Serie) |
| `'robust'` | Viele Ausreisser in Trainingsdaten |
| `'minmax'` | Bounded Outputs gewünscht |

## Wechselwirkungs-Matrix

| Parameter A | Parameter B | Wechselwirkung |
|-------------|-------------|----------------|
| `n_pool_kernel_size` | `n_freq_downsample` | Müssen konsistent absteigen — Stack-Granularität sollte übereinstimmen |
| `n_pool_kernel_size` | `input_size` | `input_size / max_pool` sollte ≥ 5 bleiben |
| `n_freq_downsample` | `h` | `h` muss durch `n_freq_downsample[i]` teilbar sein (oder NeuralForecast rundet) |
| `max_steps` | `learning_rate` | Hohes `lr` + viele Steps = divergiert; niedriges `lr` + wenige Steps = untertrainiert |
| `max_steps` | `num_lr_decays` | Decays werden gleichmässig verteilt — bei Early Stopping kann letzter Decay zu spät kommen |
| `mlp_units` | `batch_size` | Grösseres MLP braucht mehr Samples pro Batch für stabile Gradienten |
| `loss=MQLoss` | `mlp_units` | Quantile-Loss braucht mehr Kapazität als MAE → grössere MLPs testen |
| `scaler_type` | `learning_rate` | Mit `'standard'`-Scaler kann `lr` oft etwas höher angesetzt werden |

## Optuna-Suchraum

```python
import optuna
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE

def objective(trial):
    # Architektur
    pool_sizes = trial.suggest_categorical(
        'pool_sizes',
        [[8, 4, 1], [16, 8, 1], [4, 2, 1], [7, 3, 1]]
    )
    freq_down = trial.suggest_categorical(
        'freq_down',
        [[4, 2, 1], [8, 4, 1], [2, 1, 1]]
    )
    input_mult = trial.suggest_int('input_mult', 2, 6)

    # Training
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    mlp_size = trial.suggest_categorical('mlp_size', [256, 512, 1024])

    model = NHITS(
        h=H,
        input_size=input_mult * H,
        n_pool_kernel_size=pool_sizes,
        n_freq_downsample=freq_down,
        mlp_units=[[mlp_size, mlp_size]] * 3,
        learning_rate=lr,
        batch_size=batch_size,
        max_steps=500,
        loss=MAE(),
    )

    nf = NeuralForecast(models=[model], freq='D')
    cv = nf.cross_validation(df=train_df, n_windows=3, step_size=H)

    return cv['y'].sub(cv['NHITS']).abs().mean()  # MAE

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

**Wichtig**: Architektur-Parameter (`pool_sizes`, `freq_down`) als `suggest_categorical` mit vordefinierten gültigen Kombinationen — nicht als unabhängige Integer, da die Konsistenz-Constraint sonst verletzt wird.

## Empfehlungen nach Datencharakteristik

| Zeitreihen-Typ | `n_pool_kernel_size` | `input_size` | Begründung |
|---------------|---------------------|--------------|------------|
| Starke Wochensaisonalität | `[7, 3, 1]` | `3*h` bis `4*h` | Stack 1 aligned auf 7-Tage-Periode |
| Starke Jahressaisonalität | `[30, 7, 1]` | `5*h` bis `7*h` | Grober Stack muss Jahrestrend erfassen |
| Kein klares Muster | `[4, 2, 1]` | `2*h` | Konservativ, lässt Modell selbst lernen |
| Kurze Zeitreihen (<1 Jahr) | `[4, 2, 1]` + kleines `mlp_units` | `2*h` | Weniger Kapazität gegen Overfitting |

## Quellen

- [[neuralforecast-nhits-docs]] — Offizielle NeuralForecast Dokumentation, vollständige Parameterliste

## Verwandte Seiten

- [[N-HiTS]] — Entity-Seite mit Code-Beispiel
- [[Hyperparameter-Tuning mit Optuna]] — Framework für systematische Suche
- [[backtesting]] — Cross-Validation als Objective für Optuna
- [[probabilistisches-forecasting]] — `MQLoss` als alternative Loss-Funktion
- [[Feature Engineering für Zeitreihen]] — Exogene Features als zusätzliche Inputs
