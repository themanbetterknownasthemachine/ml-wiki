---
title: "Hyperparameter-Tuning mit Optuna"
type: concept
tags: [hyperparameter, optimierung, optuna, ml-workflow]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---

# Hyperparameter-Tuning mit Optuna

## Kernidee

Hyperparameter sind die "Stellschrauben" eines ML-Modells, die VOR dem Training festgelegt werden und nicht durch den Trainingsprozess selbst gelernt werden. Beispiele: Learning Rate, Batch Size, Anzahl Layer, Dropout Rate. Optuna ist ein Framework, das diese Stellschrauben automatisch und intelligent durchsucht — besser als Grid Search (zu langsam) und Random Search (zu zufällig).

**Intuition**: Stell dir vor, du suchst den tiefsten Punkt in einem Tal bei Nebel. Grid Search geht systematisch Punkt für Punkt ab (langsam). Random Search springt zufällig herum (ineffizient). Optuna dagegen nutzt die bisherigen Messungen um informiert zu entscheiden, wo als nächstes gemessen wird — Bayesian Optimization.

## Wie es funktioniert

Optuna nutzt den **TPE-Algorithmus** (Tree-structured Parzen Estimator):

1. Starte mit einigen zufälligen Hyperparameter-Kombinationen
2. Teile die bisherigen Versuche in "gute" und "schlechte" auf
3. Modelliere die Verteilung der guten vs. schlechten Hyperparameter
4. Wähle die nächste Kombination dort, wo gute Ergebnisse wahrscheinlicher sind
5. Wiederhole bis Budget erschöpft oder Konvergenz erreicht

**Pruning** ist ein Killer-Feature: Optuna kann einen Trial vorzeitig abbrechen, wenn er nach wenigen Epochen schon schlechter aussieht als die besten bisherigen Trials. Das spart enorm Zeit.

## Wann einsetzen

- Immer wenn ein Modell mehr als 3-4 Hyperparameter hat
- Vor allem bei Deep-Learning-Modellen (wo Training teuer ist → Pruning spart Zeit)
- Wenn die Suchräume gross sind (z.B. Learning Rate von 1e-5 bis 1e-1)
- Nach dem initialen Modellbau, nicht vorher (erst ein lauffähiges Modell, dann optimieren)

## Wann NICHT einsetzen

- Bei sehr schnellen Modellen (z.B. Lineare Regression) — Grid Search reicht
- Bei extrem wenig Daten — Overfitting-Gefahr durch zu viel Tuning
- Ohne klare Evaluation-Metrik — Optuna optimiert was du ihm sagst
- Wenn der Suchraum nicht gut definiert ist (Garbage in → Garbage out)

## Praxisbeispiel

Optuna-Studie für ein [[N-HiTS]]-Modell:

```python
import optuna
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

def objective(trial):
    # Suchraum definieren
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    input_size = trial.suggest_categorical("input_size", [60, 90, 120, 180])
    n_blocks_1 = trial.suggest_int("n_blocks_1", 1, 3)
    n_blocks_2 = trial.suggest_int("n_blocks_2", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    model = NHITS(
        h=30,
        input_size=input_size,
        n_pool_kernel_size=[8, 4, 1],
        n_freq_downsample=[4, 2, 1],
        stack_types=3 * ["identity"],
        n_blocks=[n_blocks_1, n_blocks_2, 1],
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_steps=500,       # Weniger Steps fürs Tuning
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    # Cross-Validation statt einfacher Train/Test Split
    cv_results = nf.cross_validation(df=train_df, n_windows=3)
    mae = cv_results['NHITS'].sub(cv_results['y']).abs().mean()
    
    return mae  # Optuna minimiert standardmässig

# Studie starten
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=3600)  # Max 50 Trials oder 1 Stunde

print(f"Bester MAE: {study.best_value:.2f}")
print(f"Beste Parameter: {study.best_params}")
```

**Wichtig**: Für [[Backtesting|zeitreihen-gerechte Evaluation]] immer Cross-Validation mit `n_windows` statt einfachem Train/Test Split verwenden.

## Eingesetzt in

- [[N-HiTS]] — Optimierung der Modellarchitektur und Trainingsparameter
- [[LightGBM Forecasting]] — Tuning von `num_leaves`, `max_depth`, `learning_rate`

## Verwandte Konzepte

- [[Bias-Variance Tradeoff]] — Warum zu viel Tuning zu Overfitting führen kann
- [[Backtesting]] — Die richtige Evaluationsstrategie für Zeitreihen
- [[Gradient Descent]] — Der Optimierungsprozess innerhalb des Trainings (vs. Hyperparameter-Tuning ausserhalb)

## Quellen
