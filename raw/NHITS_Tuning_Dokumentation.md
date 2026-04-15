# N-HiTS Tuning – Schritt-für-Schritt Dokumentation
## Pistor WUZ West Kommissionierung

**Notebook:** `notebooks/nhits_production/NHITS_Tuning.ipynb`  
**Ziel:** Tagesweise Prognose der Rüstmenge (Tonnen) für die WUZ West Kommissionierung mit dem neuronalen Zeitreihenmodell N-HiTS – systematisch optimiert via Optuna Hyperparameter-Tuning.

---

## Inhaltsverzeichnis

1. [Kontext & Ausgangslage](#1-kontext--ausgangslage)
2. [Imports & Setup (Zelle 2–3)](#2-imports--setup)
3. [Konstanten (Zelle 4)](#3-konstanten)
4. [Daten laden & bereinigen (Zelle 6)](#4-daten-laden--bereinigen)
5. [Feiertags-Kalender (Zelle 8)](#5-feiertags-kalender)
6. [Feature Engineering (Zelle 10)](#6-feature-engineering)
7. [Train/Test Split & NeuralForecast-Format (Zelle 12)](#7-traintest-split--neuralforecast-format)
8. [Baseline N-HiTS (Zelle 14)](#8-baseline-n-hits)
9. [Optuna Hyperparameter-Tuning (Zelle 16–17)](#9-optuna-hyperparameter-tuning)
10. [Analyse der Optuna-Trials (Zelle 19–21)](#10-analyse-der-optuna-trials)
11. [Finales Training & Evaluation (Zelle 23–25)](#11-finales-training--evaluation)
12. [Ensemble-Test (Zelle 27–29)](#12-ensemble-test)
13. [Zusammenfassung (Zelle 31)](#13-zusammenfassung)
14. [Modell speichern (Zelle 33)](#14-modell-speichern)

---

## 1. Kontext & Ausgangslage

### Was wird vorhergesagt?

Die tägliche **Rüstmenge** (in Tonnen) der WUZ West Kommissionierung bei Pistor. Das ist die Menge an Waren, die pro Arbeitstag kommissioniert (zusammengestellt) werden muss. Eine genaue Prognose ermöglicht bessere Personalplanung.

### Was ist N-HiTS?

**N-HiTS (Neural Hierarchical Interpolation for Time Series)** ist ein modernes, tiefes neuronales Netz für Zeitreihenprognosen. Es zerlegt die Zeitreihe hierarchisch in mehrere Frequenzkomponenten (z. B. wöchentliche, monatliche Muster) und interpoliert diese zu einer finalen Prognose.

Im Vergleich zu klassischen Modellen (OLS, ARIMA) kann N-HiTS:
- Nichtlineare Muster erkennen
- Wochentags- und Kalendereffekte via externe Features (Exogenous Variables) nutzen
- Mehrere Schritte gleichzeitig vorhersagen (Horizon = 60 Arbeitstage)

### Warum Hyperparameter-Tuning?

Ein neuronales Netz hat viele Stellschrauben (Lernrate, Netzwerkarchitektur, Normalisierung, ...). Die richtigen Werte sind nicht im Voraus bekannt. **Optuna** automatisiert die Suche durch intelligentes Ausprobieren (Bayesian Optimization / TPE-Sampler) – statt alle Kombinationen zu testen, lernt Optuna, welche Kombinationen vielversprechend sind.

### Ausgangspunkt (Baseline)

Das Notebook baut auf dem Gewinner aus `BestModelsComparison.ipynb` auf:

| Modell | MAE | MAPE | RMSE |
|--------|-----|------|------|
| Baseline N-HiTS (v2, nhits_v2_20260326) | 12.5 t | 3.7% | 16.8 t |

**MAE** = Mean Absolute Error (durchschnittlicher absoluter Fehler in Tonnen)  
**MAPE** = Mean Absolute Percentage Error (prozentualer Fehler)  
**RMSE** = Root Mean Squared Error (bestraft grosse Fehler stärker)

---

## 2. Imports & Setup

### Zelle 2 – Leichte Bibliotheken

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

**Warum?** Windows verwendet manchmal zwei OpenMP-Instanzen gleichzeitig (eine von conda, eine von PyTorch). Das führt zu einem Absturz. Diese Umgebungsvariable erlaubt es, beide koexistieren zu lassen.

```python
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
```

**Warum Optuna?** Optuna ist eine Python-Bibliothek für automatisches Hyperparameter-Tuning. Der **TPE-Sampler** (Tree-structured Parzen Estimator) ist ein Bayesian Optimization Algorithmus: Er baut ein probabilistisches Modell darüber, welche Parameter gute Ergebnisse liefern, und schlägt neue Kombinationen gezielt vor – effizienter als Grid Search oder Random Search.

Das Logging wird auf WARNING gesetzt, damit nicht jeder Trial-Fortschritt als Text ausgegeben wird.

### Zelle 3 – PyTorch & NeuralForecast

```python
torch.set_num_threads(1)
```

**Warum?** Auf Windows kann PyTorch mit mehreren Threads in einen Deadlock geraten (gegenseitiges Blockieren von Prozessen). `set_num_threads(1)` erzwingt Single-Threading und vermeidet das Problem. Das Training wird dadurch zwar nicht parallelisiert, aber es läuft stabil.

```python
from neuralforecast.models import NHITS
```

**NeuralForecast** ist eine Open-Source-Bibliothek (von Nixtla), die moderne neuronale Zeitreihenmodelle einheitlich implementiert. N-HiTS ist eines davon.

---

## 3. Konstanten

### Zelle 4 – Zentrale Parameter

```python
HORIZON = 60  # Wie weit vorausschauen?
INPUT_SIZE = 120  # Wie viele Vergangenheitswerte nutzen?
MAX_STEPS = 500  # Wie viele Trainingsschritte (Epochen)?
```

**HORIZON = 60:** Das Modell prognostiziert 60 Arbeitstage auf einmal – das entspricht ca. 3 Monaten. Dieser Horizont wird auch für den Testdatensatz verwendet, damit Prognose und Evaluation auf derselben Länge basieren.

**INPUT_SIZE = 120:** Das Modell schaut auf die letzten 120 Arbeitstage zurück (ca. 6 Monate), um Muster zu erkennen. Weil Jahresmuster wichtig sind (z. B. Sommer, Ostern), braucht das Modell einen langen Rückblick. `INPUT_SIZE = 2 × HORIZON` ist eine gängige Faustregel.

**MAX_STEPS = 500:** Das Netz wird maximal 500 Mal über die Trainingsdaten iteriert. Mit Early Stopping kann es früher abbrechen, wenn keine Verbesserung mehr eintritt.

```python
FUTR_EXOG = [
    "is_monday",
    "is_tuesday",
    "is_wednesday",
    "is_thursday",
    "is_friday",
    "day_before_holiday",
    "is_short_week",
    "is_july",
    "is_first_bday_of_month",
    "is_last_bday_of_month",
    "is_karwoche_dienstag",
    "is_tag_vor_1mai",
]
```

**FUTR_EXOG (Future Exogenous Features):** Das sind bekannte Zukunftsinformationen, die dem Modell als zusätzliche Inputs übergeben werden. Sie sind "exogen" (von aussen bekannt) und "future" (auch für die Prognoseperiode verfügbar, weil sie aus dem Kalender berechnet werden).

Jedes Feature erklärt ein spezifisches Muster in der Rüstmenge (mehr dazu in Abschnitt 6).

---

## 4. Daten laden & bereinigen

### Zelle 6 – Einlesen und Bereinigung

```python
df = pd.read_excel(
    "data/Tagesverlauf Kommissionierung WUZ West bearbeitet.xlsx", index_col=0
)
df.index = pd.to_datetime(df.index)
df = df.rename(columns={"Rüstmenge": "tonnage"})
```

Die Excel-Datei enthält die historischen Tageswerte der Rüstmenge. Der Index (erste Spalte) sind die Daten. Die einzige relevante Spalte `Rüstmenge` wird in `tonnage` umbenannt (englisch, wegen einheitlicher Benennung im Code).

**Bereinigungsschritte:**

```python
# 1. Nur Werte > 0
df = df[df["tonnage"] > 0]
```

Tage mit Rüstmenge = 0 sind Fehler (Betriebsschliessung) oder fehlende Daten, keine echten Arbeitstage.

```python
# 2. Wochenenden entfernen
df = df[df.index.dayofweek < 5]
```

Wochenenden (Samstag = 5, Sonntag = 6) werden entfernt, weil WUZ West nur wochentags arbeitet.

```python
# 3. Jahresende ausschliessen (21.–31. Dezember)
year_end_mask = [(d.month == 12 and d.day >= 21) for d in df.index]
df = df[~year_end_mask]
```

**Warum Jahresende entfernen?** In der Weihnachtszeit gibt es starke Ausreisser in der Rüstmenge (Weihnachtsgeschäft) und unregelmässige Betriebszeiten. Da das Modell auf Normalperioden optimiert wird und Weihnachten ein Sonderfall ist, werden diese Tage aus dem Trainings- und Testdatensatz ausgeschlossen. Konsequenz: Für die Weihnachtsperiode wird kein Modell eingesetzt.

---

## 5. Feiertags-Kalender

### Zelle 8 – Pistor-spezifischer Feiertagskalender

```python
_ch_national = holidays.Switzerland(years=YEARS)
_ch_lu = holidays.Switzerland(years=YEARS, prov="LU")
```

Die Python-Bibliothek `holidays` liefert alle Schweizer Feiertage. Da Pistor im Kanton Luzern ansässig ist, werden zusätzlich die LU-spezifischen Feiertage geladen.

```python
# Nationale Feiertage OHNE Auffahrt (Ascension Day)
ch_holidays_closed = {d: n for d, n in _ch_national.items() if "Ascension Day" not in n}
```

**Warum Auffahrt raus?** Auffahrt (Christi Himmelfahrt) ist in der Schweiz ein nationaler Feiertag – aber Pistor WUZ West hat an diesem Tag geöffnet. Würde Auffahrt als geschlossener Feiertag behandelt, wäre das `day_before_holiday`-Feature falsch gesetzt: Der Mittwoch vor Auffahrt würde fälschlicherweise als "Vortag vor Schliesstag" markiert, obwohl am Donnerstag normal gearbeitet wird.

```python
CLOSED_LU_NAMES = {"Good Friday", "Easter Monday", "Whit Monday", "Saint Stephen's Day"}
ch_holidays_closed.update({d: n for d, n in _ch_lu.items() if n in CLOSED_LU_NAMES})
```

Zusätzlich werden LU-spezifische Schliessungen hinzugefügt: Karfreitag, Ostermontag, Pfingstmontag und Stephanstag.

```python
ch_holidays_all = dict(_ch_national)
ch_holidays_all.update(_ch_lu)
```

`ch_holidays_all` enthält alle Feiertage (auch Auffahrt). Es wird für das `day_before_holiday`-Feature verwendet – aber da Auffahrt in `ch_holidays_closed` fehlt, hat der Vortag von Auffahrt keinen Effekt (Auffahrt ist offen, daher kein Vortags-Effekt).

---

## 6. Feature Engineering

### Zelle 10 – Erklärung aller Features

Die Funktion `create_features()` berechnet 12 binäre (0/1) Kalender-Features für jeden Arbeitstag. Binäre Features sind einfach zu interpretieren und für neuronale Netze geeignet.

#### Wochentags-Features

```python
dataframe["is_monday"] = (dataframe.index.dayofweek == 0).astype(float)
dataframe["is_tuesday"] = (dataframe.index.dayofweek == 1).astype(float)
# ... bis Freitag
```

**Warum?** Die Rüstmenge variiert stark nach Wochentag. Montag und Dienstag haben typischerweise höhere Mengen (Nachhol-Effekte nach dem Wochenende), Freitag oft tiefere Mengen. Das Netz lernt diese Muster durch die expliziten Wochentags-Flags.

Beachte: `is_friday` ist im Code zwar berechnet, aber in einem früheren Notebook wurde festgestellt, dass Freitag wenig zusätzliche Erklärungskraft hat – trotzdem ist es im Feature-Set enthalten.

#### Vortag vor Feiertag

```python
def is_day_before_holiday(date, hol_all):
    if (date + timedelta(days=1)).date() in hol_all:
        return 1.0
    if date.weekday() == 4 and (date + timedelta(days=3)).date() in hol_all:
        return 1.0
    if date.weekday() == 3 and (date + timedelta(days=1)).date() in hol_all:
        return 1.0
    return 0.0
```

**Warum?** Am Arbeitstag vor einem Feiertag wird häufig mehr kommissioniert, weil die Lücke des Folgetages kompensiert werden muss. Die Logik berücksichtigt drei Fälle:
1. Der direkte nächste Tag ist ein Feiertag
2. Es ist Freitag und der darauffolgende Montag ist Feiertag (langes Wochenende)
3. Es ist Donnerstag und Freitag ist ein Feiertag

#### Kurze Woche

```python
def is_short_week(date, hol_closed):
    monday = date - timedelta(days=date.weekday())
    return (
        1.0
        if any((monday + timedelta(days=i)).date() in hol_closed for i in range(5))
        else 0.0
    )
```

**Warum?** Eine Woche mit einem Feiertag hat weniger Arbeitstage. Das Volumen wird auf weniger Tage verteilt, was zu höheren täglichen Rüstmengen führt. Dieses Feature markiert alle Tage einer solchen Woche.

#### Juli-Feature

```python
dataframe["is_july"] = (dataframe.index.month == 7).astype(float)
```

**Warum?** Im Juli ist die Rüstmenge deutlich tiefer (Ferienzeit, weniger Lieferungen). Da das Modell sonst kaum ein "Monatsmuster" lernt, bekommt es explizit ein Signal für den Ferienmonat.

#### Erster / Letzter Arbeitstag des Monats

```python
dataframe["is_first_bday_of_month"] = ...
dataframe["is_last_bday_of_month"] = ...
```

**Warum?** Am Monatsanfang und -ende gibt es oft spezifische Logistik-Muster (z. B. Monatsabschlüsse im Handel), die die Rüstmenge beeinflussen.

#### Karwoche Dienstag

```python
def is_karwoche_dienstag(date, hol_closed):
    if date.weekday() != 1:
        return 0.0
    return (
        1.0
        if hol_closed.get((date + timedelta(days=3)).date(), "") == "Good Friday"
        else 0.0
    )
```

**Warum?** In der Karwoche ist Karfreitag ein Schliessungstag. Die Woche hat nur 4 Arbeitstage (Mo–Do). Der Dienstag liegt dabei mitten in einer Woche mit erhöhter Vorbeladung und Nachholdruck – das ergibt erfahrungsgemäss eine besonders hohe Rüstmenge an genau diesem Tag.

#### Tag vor dem 1. Mai

```python
def is_last_bday_vor_1mai(date):
    mai1 = pd.Timestamp(year=date.year, month=5, day=1)
    if mai1.dayofweek >= 5:
        return 0.0  # 1. Mai auf WE → kein Effekt
    last_bday = mai1 - pd.tseries.offsets.BDay(1)
    return 1.0 if date == last_bday else 0.0
```

**Warum?** Der 1. Mai ist im Kanton Zürich (wichtiger Pistor-Liefergebiet) ein Feiertag. Am letzten Arbeitstag davor ist die Rüstmenge tendenziell tiefer (Kunden reduzieren Bestellungen). Wenn der 1. Mai auf ein Wochenende fällt, hat er keinen Effekt.

---

## 7. Train/Test Split & NeuralForecast-Format

### Zelle 12 – Datenaufbereitung für NeuralForecast

#### Train/Test Split

```python
TEST_SIZE = 60
test_idx = df.index[-TEST_SIZE:]  # letzte 60 Arbeitstage
train_idx = df.index[:-TEST_SIZE]  # alle früheren Tage
```

Die letzten 60 Arbeitstage werden für die Evaluation zurückgehalten. Das Modell sieht diese Daten beim Training nicht.

#### Lücken auffüllen (Business Day Index)

```python
full_bday_idx = pd.bdate_range(df.index.min(), df.index.max(), freq="B")
df_full = df[["tonnage"]].reindex(full_bday_idx)
```

**Warum?** NeuralForecast erwartet einen lückenlosen Business-Day-Index (`freq='B'`). In den Rohdaten fehlen einzelne Arbeitstage (Feiertage, Betriebsschliessungen). Diese Lücken müssen gefüllt werden.

```python
weekday_medians = df.groupby(df.index.dayofweek)["tonnage"].median()
df_full["tonnage"] = df_full["tonnage"].fillna(
    df_full.index.to_series().map(lambda d: weekday_medians[d.dayofweek])
)
```

**Warum wochentag-spezifischer Median?** Montage haben systematisch andere Mengen als Freitage. Ein globaler Median würde an einem Montag-Feiertag einen falschen Wert einsetzen (zu tief oder zu hoch). Stattdessen wird für jeden fehlenden Montag der Montags-Median eingesetzt, für fehlende Dienstage der Dienstags-Median usw.

**Wichtig:** Die gefüllten Tage sind nur ein technisches Hilfsmittel für das NeuralForecast-Framework. Bei der Evaluation werden nur die echten, ursprünglichen Arbeitstage verwendet.

#### NeuralForecast-Format

```python
nf_df = pd.DataFrame({
    "unique_id": "WUZ_West",  # Identifikator der Zeitreihe
    "ds": df_full.index,  # Datum
    "y": df_full["tonnage"],  # Zielgrösse
    # ... alle FUTR_EXOG-Features
})
```

NeuralForecast erwartet ein Long-Format mit drei Pflicht-Spalten: `unique_id`, `ds`, `y`. Mehrere Zeitreihen würden verschiedene `unique_id`-Werte haben. Hier gibt es nur eine Zeitreihe: `WUZ_West`.

```python
nf_df_train = nf_df[nf_df["ds"] < test_start]  # Trainingsdaten
nf_df_future = nf_df[nf_df["ds"] >= test_start][["unique_id", "ds"] + FUTR_EXOG]
```

`nf_df_future` enthält nur die exogenen Features für die Testperiode – keine `y`-Werte. Das Modell bekommt also nur die Kalender-Informationen, aber nicht die echten Tonnagen für den Testzeitraum.

---

## 8. Baseline N-HiTS

### Zelle 14 – Referenzmodell

Bevor das Tuning beginnt, wird die Baseline-Konfiguration aus dem Vorgänger-Notebook repliziert:

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `n_pool_kernel_size` | `[4, 4, 1]` | Max-Pooling-Fenstergrösse je Stack |
| `n_freq_downsample` | `[8, 2, 1]` | Downsampling-Faktor je Stack |
| `learning_rate` | `1e-3` | Lernrate des Adam-Optimierers |
| `scaler_type` | `'robust'` | Normalisierung (robuster Scaler) |
| `max_steps` | `500` | Maximale Trainingsschritte |
| `early_stop_patience_steps` | `10` | Stopp wenn 10× keine Verbesserung |

**Warum replizieren?** Um sicherzustellen, dass die Evaluation hier identisch mit dem Vorgänger-Notebook ist. Wenn die Baseline-Replikation ähnliche Metriken liefert, ist die gesamte Pipeline konsistent und die Vergleiche mit dem optimierten Modell sind fair.

**N-HiTS Architektur erklärt:**

```
Eingabe (INPUT_SIZE=120 Vergangenheitswerte + FUTR_EXOG)
    ↓
Stack 0 (grobe Trends):  pool_k=4, freq_d=8  → lernt langsame, grosse Muster
    ↓
Stack 1 (mittlere Muster): pool_k=4, freq_d=2  → lernt wöchentliche Zyklen
    ↓
Stack 2 (feine Details): pool_k=1, freq_d=1   → lernt kurzfristige Schwankungen
    ↓
Ausgabe: HORIZON=60 Prognosen (additive Kombination der 3 Stacks)
```

Jeder Stack hat seinen eigenen `pool_kernel_size` (wie breit der Max-Pooling-Filter ist, der irrelevante Details herausschmeisst) und `freq_downsample` (wie stark zeitlich vereinfacht wird).

---

## 9. Optuna Hyperparameter-Tuning

### Zelle 16 – Die Objective-Funktion

Die Objective-Funktion ist das Herzstück des Tunings. Optuna ruft sie für jeden Trial auf, übergibt Parameter-Vorschläge, und bewertet das Ergebnis (MAE):

```python
def nhits_objective(trial):
    # Optuna schlägt Parameter vor:
    n_pool_kernel_size = [
        trial.suggest_categorical("pool_k_0", [1, 2, 4, 8, 16]),  # Stack 0
        trial.suggest_categorical("pool_k_1", [1, 2, 4, 8]),  # Stack 1
        trial.suggest_categorical("pool_k_2", [1, 2]),  # Stack 2
    ]
```

**Suchraum (alle Parameter):**

| Parameter | Typ | Wertebereich | Bedeutung |
|-----------|-----|-------------|-----------|
| `pool_k_0/1/2` | kategorisch | `[1,2,4,8,16]` / `[1,2,4,8]` / `[1,2]` | Max-Pooling-Fenster je Stack |
| `freq_d_0/1/2` | kategorisch | `[4,8,16,24]` / `[2,4,8]` / `[1,2]` | Downsampling-Faktor je Stack |
| `n_blocks_0/1/2` | integer | 1–4 | Anzahl sequenzieller Blocks je Stack |
| `lr` | float (log) | 1e-4 bis 1e-2 | Lernrate |
| `scaler` | kategorisch | `robust`, `standard`, `minmax` | Normalisierungsmethode |
| `dropout` | float | 0.0–0.3 | Dropout-Rate (Regularisierung) |
| `interp_mode` | kategorisch | `linear`, `nearest` | Interpolationsmodus |
| `max_steps` | kategorisch | 200, 300, 400 | Trainingsschritte (reduziert für Speed) |

**Warum `suggest_float('lr', ..., log=True)`?** Die Lernrate wird auf logarithmischer Skala gesucht. Der Unterschied zwischen 1e-4 und 1e-3 ist genauso relevant wie zwischen 1e-3 und 1e-2. Lineare Suche würde den Bereich kleiner Lernraten kaum erkunden.

**Warum `max_steps ∈ [200, 300, 400]` statt 500?** Jeder Optuna-Trial dauert auf CPU 1–3 Minuten. Mit 500 Steps wäre ein 40-Trial-Lauf 33–100 Stunden lang. Die reduzierte Schrittzahl beschleunigt das Tuning deutlich. Das beste Modell wird danach mit den vollen 500 Steps final trainiert.

**Early Stopping bei Optuna-Trials:**
```python
early_stop_patience_steps = (5,)  # Optuna: nur 5 (schnell)
# finales Modell: 10 (stabiler)
```

### Zelle 17 – Study starten

```python
study = optuna.create_study(
    direction="minimize",  # MAE soll minimiert werden
    sampler=TPESampler(seed=42),  # TPE = Bayesian Optimization
    study_name="NHITS_WUZ_West",
)
```

**TPE-Sampler (Tree-structured Parzen Estimator):** Nach den ersten paar zufälligen Trials baut der Algorithmus ein Wahrscheinlichkeitsmodell: "Bei welchen Parameterwerten war das Ergebnis gut / schlecht?" Er schlägt dann neue Parameter vor, die mit hoher Wahrscheinlichkeit gut sind – effizienter als rein zufälliges Suchen.

**Seed = 42:** Sorgt für Reproduzierbarkeit. Mit demselben Seed, denselben Daten und derselben Reihenfolge der Trials kommt man immer zum selben Ergebnis.

```python
study.enqueue_trial({
    "pool_k_0": 4,
    "pool_k_1": 4,
    "pool_k_2": 1,
    "freq_d_0": 8,
    "freq_d_1": 2,
    "freq_d_2": 1,
    "lr": 1e-3,
    "scaler": "robust",
    "dropout": 0.0,
    "interp_mode": "linear",
    "max_steps": 400,
})
```

**Warm Start:** Der erste Trial ist kein Zufall, sondern exakt die bekannte Baseline-Konfiguration. So hat Optuna sofort einen Referenzpunkt. Wenn die Baseline bereits gut ist, muss Optuna nicht erst zufällig auf ähnliche Parameter stossen.

```python
study.optimize(nhits_objective, n_trials=40, show_progress_bar=True)
```

Optuna führt 40 Trials aus. Jeder Trial trainiert ein N-HiTS-Modell mit neuen Parametern und misst den MAE auf dem Testdatensatz.

---

## 10. Analyse der Optuna-Trials

### Zelle 19 – Trials-Tabelle

```python
trials_df = study.trials_dataframe()
trials_df = trials_df[trials_df["state"] == "COMPLETE"]
trials_df = trials_df.sort_values("value")
print(trials_df[cols].head(10))
```

Zeigt die Top-10 Trials nach MAE. So sieht man, welche Parameterkombinationen am besten abgeschnitten haben.

### Zelle 20 – Optuna-Visualisierungen

Sofern Plotly installiert ist, werden 5 interaktive Diagramme erzeugt:

1. **Optimierungsverlauf:** MAE pro Trial + bester MAE bis dahin. Zeigt, ob Optuna noch lernt oder konvergiert ist.
2. **Parameter Importance:** Welche Hyperparameter haben den grössten Einfluss auf den MAE?
3. **Parallel Coordinates:** Übersicht über alle Parameter und ihre MAE-Werte gleichzeitig.
4. **Contour-Plot (pool_k_0 vs. freq_d_0):** Wie interagieren diese zwei Parameter?
5. **Contour-Plot (lr vs. scaler):** Welche Lernraten funktionieren mit welchem Scaler?

Wenn Plotly fehlt, gibt es einen matplotlib-Fallback:

```python
# Optimization History + MAE-Verteilung als Balkendiagramme
```

### Zelle 21 – Parameter Importance (matplotlib)

```python
importances = optuna.importance.get_param_importances(study)
```

Optuna berechnet, welche Parameter den MAE am stärksten beeinflusst haben (basierend auf Fano-Importance, einem Random-Forest-basierten Wichtigkeitsmass). Hohe Importance = kleine Parameteränderungen haben grosse Auswirkung.

---

## 11. Finales Training & Evaluation

### Zelle 23 – Finales Modell trainieren

```python
bp = study.best_params  # beste Parameter aus Optuna

nhits_best = NHITS(
    h=HORIZON,
    input_size=INPUT_SIZE,
    max_steps=MAX_STEPS,  # jetzt volle 500 Steps
    futr_exog_list=FUTR_EXOG,
    scaler_type=bp["scaler"],
    n_pool_kernel_size=[bp["pool_k_0"], bp["pool_k_1"], bp["pool_k_2"]],
    n_freq_downsample=[bp["freq_d_0"], bp["freq_d_1"], bp["freq_d_2"]],
    n_blocks=[bp["n_blocks_0"], bp["n_blocks_1"], bp["n_blocks_2"]],
    dropout_prob_theta=bp["dropout"],
    interpolation_mode=bp["interp_mode"],
    early_stop_patience_steps=10,  # finales Modell: 10 statt 5
    val_check_steps=50,
    learning_rate=bp["lr"],
    random_seed=42,
)
```

Die beste Konfiguration aus Optuna wird mit `max_steps=500` (statt 200–400 in den Trials) und `early_stop_patience_steps=10` (stabiler) final trainiert. So hat das Modell mehr Zeit zum Konvergieren.

**Warum `random_seed=42`?** Neuronale Netze starten mit zufällig initialisierten Gewichten. Mit festem Seed ist das Ergebnis reproduzierbar.

```python
nf_best.fit(df=nf_df_train, val_size=VAL_SIZE)
```

`val_size=60`: Die letzten 60 Tage der Trainingsdaten werden intern als Validation Set verwendet. NeuralForecast stoppt das Training automatisch, wenn der Validation Loss sich `early_stop_patience_steps` Mal hintereinander nicht verbessert (Early Stopping).

```python
preds_best = nf_best.predict(futr_df=nf_df_future)
preds_best_orig = preds_best[preds_best["ds"].isin(original_dates_set)]
```

**Wichtig:** Gefiltert auf `original_dates_set` – die Evaluation findet nur auf echten Arbeitstagen statt, nicht auf den künstlich aufgefüllten Business Days.

**Ergebnisinterpretation:**
```
delta_mae = best_mae - BASELINE_NHITS['MAE']
if delta_mae < 0:
    → Verbesserung: {abs(delta_mae):.1f} t weniger MAE als Baseline
else:
    → Kein Gewinn vs. Baseline (Baseline ist bereits optimal)
```

**Fazit aus dem Tuning:** Optuna fand keinen Gewinn gegenüber der Baseline-Konfiguration – die ursprünglichen Parameter waren bereits nahe optimal. Das ist ein positives Resultat: Es bedeutet, dass die Baseline-Konfiguration robuster ist als erwartet.

### Zelle 24 – Vergleichs-Plot

Zwei Subplots:

1. **Zeitreihen-Plot:** Tatsächliche Rüstmenge vs. Baseline N-HiTS vs. Optimiertes N-HiTS über die 60 Testtage. Zeigt visuell, wo die Modelle gut/schlecht liegen.

2. **Residuenplot:** `Fehler = Actual – Predicted` für beide Modelle. Systematische Abweichungen in eine Richtung (Bias) sind problematischer als zufällige Fehler.

### Zelle 25 – Wochentags-Analyse

```python
for wd in range(5):  # Mo=0, Di=1, Mi=2, Do=3, Fr=4
    mask = [d.weekday() == wd for d in common_d]
    mae_wt.append(np.mean(np.abs(actual[mask] - pred[mask])))
    bias_wt.append(np.mean(actual[mask] - pred[mask]))
```

Zeigt den MAE und Bias getrennt für jeden Wochentag. Aufschlussreich für die Frage: "An welchem Wochentag liegt das Modell systematisch daneben?"

**Bias > 0:** Modell unterschätzt (sagt weniger voraus als tatsächlich)  
**Bias < 0:** Modell überschätzt (sagt mehr voraus als tatsächlich)

---

## 12. Ensemble-Test

### Zellen 27–29 – Ensemble aus drei Modellen

```python
from neuralforecast.models import PatchTST, TSMixerx

tsm = TSMixerx(h=60, input_size=120, max_steps=500, ...)
patchtst = PatchTST(h=60, input_size=120, max_steps=500, ...)
```

Zum Vergleich werden zwei weitere Modelle aus `BestModelsComparison.ipynb` trainiert:

- **TSMixerx:** Mischarchitektur für multivariate Zeitreihen, gut für komplexe Interaktionen zwischen Features
- **PatchTST:** Transformer-basiertes Modell, das Zeitreihen in "Patches" (Blöcke) aufteilt und mit einem Self-Attention-Mechanismus verarbeitet

**Drei Ensemble-Strategien:**

```python
ens_equal = (nhits + tsm + patchtst) / 3  # Gleiches Gewicht
ens_weighted = (w_nhits * nhits + w_tsm * tsm + ...) / ...  # MAE-gewichtet
ens_median = np.median([nhits, tsm, patchtst], axis=1)  # Median
```

1. **Gleichgewichtetes Mittel:** Alle drei Modelle gleich wichtig. Einfach, aber robust.
2. **MAE-gewichtetes Mittel:** Modelle mit tieferem MAE erhalten mehr Gewicht (Gewicht = 1/MAE). Das beste Modell dominiert stärker.
3. **Median:** Robustester Ansatz – ein Ausreisser in einem Modell kann das Ergebnis nicht stark verzerren (im Gegensatz zum Mittelwert).

**Warum Ensemble?** Einzelne Modelle haben unterschiedliche Stärken und Schwächen. Durch Kombination mehrerer Modelle werden Fehler teils ausgeglichen. Ensembles sind in der Praxis oft besser als das beste Einzelmodell.

---

## 13. Zusammenfassung

### Zelle 31 – Ergebnistabelle

Am Ende werden alle Modell-Ergebnisse in einer sortierten Tabelle ausgegeben:

```
N-HiTS TUNING – ZUSAMMENFASSUNG
──────────────────────────────────────────────────────────────────────
Modell                          MAE (t)    MAPE (%)   RMSE (t)
─────────────────────────────────────────────────────────────────
...
```

**Fazit der Zusammenfassung:**
- Das optimierte N-HiTS hat keinen Gewinn gegenüber der Baseline gebracht (Baseline war bereits optimal)
- Als Produktionsmodell wird das N-HiTS mit der Baseline-Konfiguration (`nhits_v3_20260410`) empfohlen
- Optional: Ensemble (Median) für noch mehr Robustheit gegenüber Ausreissern

---

## 14. Modell speichern

### Zelle 33 – Persistenz für die Produktion

```python
MODEL_DIR = r"...\models\nhits_v3_20260410"
nf_best.save(path=MODEL_DIR, overwrite=True)
```

Das finale Modell wird in einem versionierten Verzeichnis gespeichert. Der Versionsname `nhits_v3_20260410` enthält das Datum der letzten Trainingsläufe (10.04.2026).

```python
study_path = os.path.join(RESULTS_DIR, "nhits_study.pkl")
with open(study_path, "wb") as f:
    pickle.dump(study, f)
```

Die Optuna Study wird als `.pkl`-Datei gespeichert. So kann man später:
- Die Trial-Geschichte analysieren
- Das Tuning fortsetzen (`study.optimize(..., n_trials=20)`)
- Die Parameter Importance nochmal visualisieren

```python
results_export.to_csv(results_path, index=False, sep=";", decimal=",")
preds_export.to_csv(preds_path, index=False, sep=";", decimal=",")
```

Ergebnistabelle und Prognosewerte werden als CSV gespeichert (Semikolon-getrennt, Komma als Dezimalzeichen – Excel-kompatibel).

**Modell laden für neue Prognosen:**

```python
nf_loaded = NeuralForecast.load(path=MODEL_DIR)
preds = nf_loaded.predict(futr_df=nf_df_future)
```

Das gespeicherte Modell kann für neue Prognosen geladen werden, ohne das Modell neu trainieren zu müssen.

---

## Anhang: Modell-Dateien und Resultate

| Datei | Inhalt |
|-------|--------|
| `models/nhits_v3_20260410/` | Gespeichertes NeuralForecast-Modell (finale Produktion) |
| `results/nhits_study.pkl` | Optuna Study (40 Trials, alle Parameter und Ergebnisse) |
| `results/nhits_tuning_results.csv` | Ergebnistabelle: MAE/MAPE/RMSE aller Modelle |
| `results/nhits_best_predictions.csv` | Prognosewerte vs. Actual für die 60 Testtage |

## Anhang: Glossar

| Begriff | Erklärung |
|---------|-----------|
| **MAE** | Mean Absolute Error – Durchschnittlicher absoluter Fehler in Tonnen. Einfach interpretierbar: "Im Schnitt liegt die Prognose X Tonnen daneben." |
| **MAPE** | Mean Absolute Percentage Error – Prozentualer Fehler. Unabhängig von der absoluten Mengengrösse. |
| **RMSE** | Root Mean Squared Error – Wie MAE, aber grosse Fehler werden stärker bestraft. |
| **Horizon** | Prognose-Horizont: Wie viele Schritte voraus wird prognostiziert? |
| **Input Size** | Lookback-Fenster: Wie viele historische Werte bekommt das Modell? |
| **Exogenous Features** | Zusatzinformationen, die das Modell nutzen kann (hier: Kalender-Features). |
| **Early Stopping** | Das Training stoppt automatisch, wenn keine Verbesserung mehr eintritt. |
| **N-HiTS Stack** | N-HiTS hat 3 Stacks, die unterschiedliche Frequenzkomponenten modellieren. |
| **Max-Pooling** | Reduziert die Zeitreihe auf das Maximum in einem Fenster – filtert unwichtige Details heraus. |
| **Dropout** | Regularisierungstechnik: Neuronen werden während des Trainings zufällig deaktiviert, um Overfitting zu verhindern. |
| **Scaler** | Normalisierung der Daten vor dem Training (robust: widerstandsfähig gegen Ausreisser). |
| **TPE-Sampler** | Bayesian Optimization Algorithmus von Optuna für intelligentes Hyperparameter-Tuning. |
| **Ensemble** | Kombination mehrerer Modelle zu einem gemeinsamen Ergebnis. |
| **Bias** | Systematische Über- oder Unterschätzung eines Modells. |
