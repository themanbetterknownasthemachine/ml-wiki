---
title: "Chronos-2: Learning General-Purpose Time Series Forecasting via In-Context Ensembling"
type: source
tags: [paper, foundation-model, chronos, multivariate, covariate, group-attention]
quelle_typ: paper
url: "https://arxiv.org/abs/2510.15821"
autoren: [Ansari et al., Amazon]
datum: 2025-10-01
---

# Chronos-2: Learning General-Purpose Time Series Forecasting via In-Context Ensembling

## Kernbotschaft

Chronos-2 erweitert [[Foundation Models für Zeitreihen|Foundation Models]] von univariaten auf **multivariate Zeitreihen und Covariates** durch einen neu eingeführten **Group Attention Mechanismus**, der In-Context Learning (ICL) über verwandte Serien ermöglicht — ohne task-spezifisches Training.

## Architektur

### Tokenisierung & Patches
- Eingabewerte werden **standardisiert + sinh⁻¹-transformiert** (stabilisiert Varianz, reduziert Ausreisser-Einfluss)
- Zeitreihe wird in **nicht-überlappende Patches** aufgeteilt
- Jeder Patch erhält Meta-Features: **relativer Zeitindex** und **Mask-Feature** (beobachtet vs. fehlend)
- Residualnetzwerk mappt auf Embeddings der Dimension `D_model`

### Group Attention (Kerninnovation)
Der zentrale neue Mechanismus für multivariate/covariate Unterstützung:
- Teilt Serien in **Gruppen** ein und aggregiert Information innerhalb der Gruppe
- Unterstützte Gruppen-Typen:
  - Einzelne Serie (univariat)
  - Mehrere verwandte Serien (Cross-Learning)
  - Multivariate Variaten mit gemeinsamer Dynamik
  - Ziele + Past-Only Covariates
  - Ziele + Known (zukunftsbekannte) Covariates
- **Memory-Skalierung**: O(V) — linear in Anzahl Variaten, kein O(V²) wie Moirai
- Keine Positions-Embeddings innerhalb der Gruppen (keine natürliche Ordnung)

### Transformer-Architektur
- **Encoder-only**, T5-ähnliches Design
- Alternierende **Time Attention** (über Patches einer Serie) und **Group Attention** (über Serien in der Gruppe)
- **Rotary Position Embeddings (RoPE)** statt T5-relatives Positioning
- **Quantile Head**: produziert 21 Quantile (`{0.01, 0.05, 0.1, …, 0.9, 0.95, 0.99}`)

### Modellgrössen
| Variante | Parameter |
|----------|-----------|
| Base     | 120M      |
| Small    | 28M (~1% schlechter auf GIFT-Eval) |

## Training

### Datenzusammensetzung
**Real (univariat):**
- 22 Datensätze (M4, Electricity, Solar, Taxi, USHCN, Weatherbench, Wiki, …)
- ~700K Zeitreihen über verschiedene Frequenzen

**Synthetisch (multivariate Strukturen):**
- **TSI-Generator**: zufällige Trend+Saisonalität+Irregularität-Kombinationen
- **TCM (Temporal Causal Model)**: Autoregression aus zufälligen kausalen Graphen
- **Multivariatizers**: erzeugen Dependencies zwischen Serien
  - *Cotemporal*: lineare/nichtlineare Transformationen gleichzeitig
  - *Sequenziell*: Lead-Lag-Effekte und Kointegration

### Loss-Funktion
Quantile Regression Objective:
```
L = Σ_q∈Q [ q·max(z - ẑ^q, 0) + (1-q)·max(ẑ^q - z, 0) ]
```
- Nur auf Target-Dimensionen berechnet (Missing Values und Covariates ausgenommen)

### Zwei-Stufen-Training
| Stage | Kontext | Zweck |
|-------|---------|-------|
| 1 | 2048 Token | Basistraining |
| 2 | 8192 Token | Lange Horizonte, mehr Output-Patches |

### Covariate Encoding
- **Kategorische Covariates (univariat)**: Target Encoding
- **Kategorische Covariates (multivariat)**: Ordinal Encoding
- Eingabematrix `U = [Ṽ, W̃]`: `Ṽ` = historische Targets + Past-Covariates, `W̃` = Known Covariates (Targets/Past-Only als NULL maskiert)

## Benchmark-Ergebnisse (Zero-Shot)

### fev-bench (100 Tasks: 32 univariat, 26 multivariat, 42 mit Covariates)
| Modell | Win Rate SQL | Skill Score SQL | WQL Win Rate | MASE Win Rate |
|--------|-------------|----------------|--------------|---------------|
| **Chronos-2** | **90.7%** | **47.3%** | **88.5%** | **87.9%** |
| TiRex | 80.8% | 42.6% | 79.0% | 75.1% |
| TimesFM-2.5 | 75.9% | 42.3% | 76.8% | 74.4% |
| Chronos-Bolt | — | 38.9% | — | — |

### GIFT-Eval (97 Tasks)
| Modell | WQL Win Rate | MASE Win Rate |
|--------|-------------|---------------|
| **Chronos-2** | **81.9%** | **83.8%** |
| TimesFM-2.5 | 77.5% | 77.7% |
| TiRex | 76.5% | 71.9% |

### Chronos Benchmark II (27 Tasks)
| Modell | WQL Win Rate | MASE Win Rate |
|--------|-------------|---------------|
| **Chronos-2** | **79.8%** | **81.5%** |
| TiRex | 70.4% | 71.6% |
| TimesFM-2.5 | 70.0% | 71.6% |

**Covariate-Gains**: +13–15 Skill-Score-Punkte durch ICL auf covariate-informierten Tasks.

## Vergleich Chronos-Bolt → Chronos-2

| Fähigkeit | Chronos-Bolt | Chronos-2 |
|-----------|-------------|-----------|
| Multivariate Zeitreihen | ✗ | ✓ |
| Past-Only Covariates | ✗ | ✓ |
| Known Covariates | ✗ | ✓ |
| Kategorische Covariates | ✗ | ✓ |
| Cross-Learning | ✗ | ✓ |
| fev-bench Skill Score | 38.9% | 47.3% |
| Memory-Skalierung | — | O(V) |

## Key Takeaways

1. **Group Attention** ist die Schlüsselinnovation: flexibles ICL ohne Architekturumbauten für verschiedene Task-Typen
2. **Synthetische Multivariatizers** sind effektiv: reale multivariate Trainingsdaten nicht zwingend nötig
3. **Covariate-Unterstützung** ist praktisch hochrelevant — grösste Gewinne gegenüber Konkurrenz genau dort
4. **Small-Variante** (28M) ist praktisch fast gleich gut wie Base (120M) — gut für ressourcenbeschränkte Deployments
5. **Durchsatz**: ~300 Serien/Sekunde auf A10G GPU (Batch 1024)

## Verwandte Wiki-Seiten

- [[Foundation Models für Zeitreihen]] — Entity-Seite mit Einordnung
- [[Transformer Architektur]] — Basis-Architektur
- [[Transfer Learning]] — Konzept dahinter
- [[Feature Engineering für Zeitreihen]] — Was Foundation Models ggf. überflüssig machen
