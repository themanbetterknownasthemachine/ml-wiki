---
title: "Stationarität"
type: concept
tags: [zeitreihen, statistik, sarimax, stationarität, differenzierung]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---

# Stationarität

Stationarität ist eine statistische Grundeigenschaft von Zeitreihen. Viele klassische Modelle wie [[SARIMAX]] setzen sie voraus — ohne Stationarität sind die Modellparameter nicht stabil und Prognosen unzuverlässig.

## Definition

Eine Zeitreihe ist **schwach stationär** wenn:

1. **Konstanter Erwartungswert**: `E[y_t] = μ` für alle t
2. **Konstante Varianz**: `Var(y_t) = σ²` für alle t
3. **Zeitunabhängige Kovarianz**: `Cov(y_t, y_{t+k})` hängt nur von k ab, nicht von t

Intuitiv: Die statistische Struktur der Zeitreihe **verschiebt sich nicht im Zeitverlauf**.

### Nicht-stationäre Muster

```
Nicht-stationär (Trend):        Nicht-stationär (Varianzwachstum):
    ↗                               /\/\/\/\
   ↗                            /\/\
  ↗                          /\
─────────────────            ─────────────────────────────────

Stationär:
  /\/\/\/\/\/\/\/\/\
─────────────────────────────────
```

Typische Ursachen für Nicht-Stationarität:
- **Trend** (steigender/fallender Mittelwert)
- **Saisonalität** (periodische Schwankungen im Mittelwert)
- **Heteroskedastizität** (wachsende Varianz)
- **Strukturbrüche** (plötzliche Niveau-Änderungen)

## Warum braucht SARIMAX Stationarität?

[[SARIMAX]] modelliert Zeitreihen als lineare Kombination vergangener Werte und Fehlerterme:

```
y_t = c + φ₁·y_{t-1} + φ₂·y_{t-2} + θ₁·ε_{t-1} + ... + ε_t
```

Wenn `y_t` einen Trend hat (nicht-stationär), dann:
- Wächst `y_{t-1}` mit der Zeit → Koeffizient φ₁ ist nicht konstant
- Das Modell hat **keine stabilen Parameter** → Vorhersage ausserhalb des Trainings-Niveaus
- **Spurious Regression**: scheinbar gute Fits die in Wirklichkeit nur den gemeinsamen Trend widerspiegeln

Das `I(d)` in ARIMA/SARIMAX steht für **Integration** — die Differenzierungsordnung die nötig ist um Stationarität herzustellen.

## Stationarität testen

### ADF-Test (Augmented Dickey-Fuller)

**Nullhypothese H₀**: Die Zeitreihe hat eine Einheitswurzel (= nicht-stationär)

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(y, autolag='AIC')
print(f"ADF Statistik: {result[0]:.4f}")
print(f"p-Wert:        {result[1]:.4f}")
print(f"Krit. Werte:   {result[4]}")

# Interpretation:
# p < 0.05 → H₀ ablehnen → Zeitreihe ist stationär
# p > 0.05 → H₀ nicht ablehnen → wahrscheinlich nicht-stationär
```

**Vorsicht**: ADF hat geringe Power bei kurzen Zeitreihen — lehnt H₀ zu selten ab.

### KPSS-Test (Kwiatkowski-Phillips-Schmidt-Shin)

**Nullhypothese H₀**: Die Zeitreihe ist stationär (umgekehrt zum ADF!)

```python
from statsmodels.tsa.stattools import kpss

result = kpss(y, regression='c', nlags='auto')
print(f"KPSS Statistik: {result[0]:.4f}")
print(f"p-Wert:         {result[1]:.4f}")

# Interpretation:
# p > 0.05 → H₀ nicht ablehnen → Zeitreihe ist stationär
# p < 0.05 → H₀ ablehnen → nicht-stationär
```

### ADF und KPSS kombinieren

Da beide Tests verschiedene H₀ haben, ergibt die Kombination mehr Sicherheit:

| ADF | KPSS | Schluss |
|-----|------|---------|
| p < 0.05 (stationär) | p > 0.05 (stationär) | ✓ Stationär |
| p > 0.05 (nicht-stat.) | p < 0.05 (nicht-stat.) | ✗ Nicht-stationär |
| p < 0.05 (stationär) | p < 0.05 (nicht-stat.) | Trendstationär (Trend vorhanden aber stationär um Trend) |
| p > 0.05 (nicht-stat.) | p > 0.05 (stationär) | Schwache Aussage, mehr Daten nötig |

## Stationarität herstellen

### 1. Differenzierung (für Trend)

```python
# Erste Differenz: entfernt linearen Trend
y_diff = y.diff().dropna()

# Zweite Differenz: entfernt quadratischen Trend (selten nötig)
y_diff2 = y.diff().diff().dropna()
```

Das `d` in ARIMA(p,d,q): d=1 bedeutet erste Differenz, d=2 zweite.

### 2. Saisonale Differenzierung (für Saisonalität)

```python
# Saisonale Differenz mit Periode s (z.B. s=7 für Wochenmuster)
y_seasonal_diff = y.diff(7).dropna()
```

Das `D` in SARIMA(p,d,q)(P,D,Q)s: saisonale Differenzierungsordnung.

### 3. Log-Transformation (für wachsende Varianz)

```python
import numpy as np
y_log = np.log(y)  # stabilisiert multiplikative Saisonalität
```

Oft als erster Schritt vor Differenzierung, wenn Varianz mit dem Niveau wächst.

### Vorgehen in der Praxis

```python
# Schritt 1: Plot der Zeitreihe (visuell beurteilen)
y.plot()

# Schritt 2: ADF + KPSS
adf_result = adfuller(y)
kpss_result = kpss(y, regression='c')

# Schritt 3: Falls nicht-stationär → differenzieren
if adf_result[1] > 0.05:
    y_transformed = y.diff().dropna()
    # Schritt 2 wiederholen

# Schritt 4: SARIMAX mit korrektem d/D
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
```

## Stationarität und Deep Learning

[[N-HiTS]] und [[Foundation Models für Zeitreihen|Foundation Models]] brauchen **keine explizite Stationarität** — sie lernen Trend und Saisonalität direkt aus den Daten. Aber:

- **N-HiTS**: Interne Normalisierung (Instance Normalization) kompensiert Niveau-Unterschiede
- **Chronos-2**: Standardisierung + sinh⁻¹-Transformation beim Preprocessing (aus [[chronos-2-paper]])
- Dennoch: Starke Strukturbrüche können beide Modelle überfordern

**Praktische Konsequenz**: Stationarität ist ein SARIMAX-spezifisches Thema — bei modernen Deep-Learning-Ansätzen ist es weniger kritisch, aber das Verständnis hilft bei der Fehleranalyse.

## Verwandte Seiten

- [[SARIMAX]] — benötigt Stationarität als Voraussetzung; d und D in der Modellspezifikation
- [[N-HiTS]] — umgeht das Problem durch interne Normalisierung
- [[Feature Engineering für Zeitreihen]] — Detrending und Deseasonalisierung als Feature-Schritt
- [[backtesting]] — Stationarität im Trainings-Fold separat prüfen (nicht über gesamten Datensatz)
- [[Informationskriterien AIC BIC]] — für Modellselektion nach Differenzierung
