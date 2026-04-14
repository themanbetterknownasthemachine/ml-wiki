---
title: "Foundation Models für Zeitreihen"
type: entity
tags: [forecasting, foundation-model, chronos, timesfm, deep-learning]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
quellen: 1
---

# Foundation Models für Zeitreihen

## Überblick

Foundation Models für Zeitreihen sind vortrainierte Modelle, die auf grossen Mengen diverser Zeitreihen trainiert wurden und ohne oder mit minimalem Fine-Tuning auf neue, ungesehene Zeitreihen angewendet werden können (Zero-Shot oder Few-Shot Forecasting). Sie übertragen die Idee von GPT/BERT auf die Zeitreihen-Domäne.

## Kandidaten (Stand April 2026)

### Chronos-2 (Amazon)
- Nachfolger von Chronos-Bolt, **Encoder-only Transformer** (T5-ähnlich, [[Transformer Architektur|RoPE]]-Embeddings)
- Kernidee: **Group Attention** — ermöglicht In-Context Learning über verwandte Serien (multivariate + covariates)
- Unterstützt: univariat, multivariat, Past-Only Covariates, Known (zukunftsbekannte) Covariates, kategorische Covariates
- Training: ~700K reale univariate Zeitreihen + synthetische Multivariatizers (TCM, TSI-Generator)
- Grössen: Base (120M), Small (28M — nur ~1% schlechter)
- Output: 21 Quantile für probabilistisches Forecasting
- Durchsatz: ~300 Serien/Sekunde auf A10G GPU
- Open Source (Apache 2.0)
- Quelle: [[chronos-2-paper]]

### TimesFM 2.5 (Google)
- Decoder-only Transformer, speziell für Zeitreihen designt
- Patching-Ansatz (ähnlich [[PatchTST]]): Zeitreihe wird in Patches zerlegt
- Vortrainiert auf Google-internen + öffentlichen Zeitreihen-Datensätzen
- Unterstützt variable Frequenzen und Forecast-Horizonte
- Verfügbar auf Hugging Face

### Moirai (Salesforce)
- Universelles Forecasting-Modell mit "Any-Variate" Unterstützung
- Kann univariate und multivariate Zeitreihen gleichzeitig verarbeiten
- Masked Encoder Architektur

## Architektur / Funktionsweise

Der zentrale Unterschied zu [[N-HiTS]] oder [[SARIMAX]]:

| Aspekt | Klassisch (N-HiTS) | Foundation Model |
|--------|-------------------|------------------|
| Training | Auf DEINEN Daten | Auf Millionen diverser Zeitreihen |
| Feature Engineering | Nötig (→ [[Feature Engineering für Zeitreihen]]) | Oft nicht nötig |
| Fine-Tuning | — | Optional, oft Zero-Shot möglich |
| Daten-Anforderung | Hunderte bis tausende Beobachtungen | Kann mit wenig Daten umgehen |
| Domänen-Spezifität | Hoch (gelernt auf deinen Daten) | Breit (generalisiert über Domänen) |

**Zero-Shot-Workflow**: Du gibst dem Modell einfach deine historische Zeitreihe und sagst "forecast 30 Tage". Kein Training, kein Feature Engineering. Das Modell nutzt sein vortrainiertes Wissen über generelle Zeitreihenmuster.

## Eingesetzte Konzepte

- [[Transformer Architektur]] — Basis der meisten Foundation Models
- [[Transfer Learning]] — Vortrainiertes Wissen auf neue Domäne übertragen
- [[Backtesting]] — Evaluation im Vergleich zu bestehenden Modellen

## Stärken

- Zero-Shot: sofort einsetzbar ohne Training auf eigenen Daten
- Kein oder minimales [[Feature Engineering für Zeitreihen|Feature Engineering]] nötig
- Besonders stark bei wenig historischen Daten (Cold Start Problem)
- Oft gute Unsicherheitsintervalle (probabilistic Forecasting)
- Schnelle Iteration: neues Modell testen = ein API-Call

## Schwächen / Limitierungen

- Möglicherweise schlechter als domänen-spezifisch trainierte Modelle bei genug Daten
- Black Box auf Steroiden: noch weniger interpretierbar als [[N-HiTS]]
- Modellgrösse: die grossen Varianten brauchen GPU-Infrastruktur
- Noch relativ jung: weniger Production-Erfahrung als etablierte Methoden
- Risiko der "Halluzination": bei Zeitreihen die fundamental anders sind als die Trainingsdaten
- Fine-Tuning kann teuer sein und erfordert eigene GPU-Ressourcen

## Verwandte Entities

- [[N-HiTS]] — Aktuelles Produktionsmodell, Benchmark-Vergleich
- [[SARIMAX]] — Statistische Baseline
- [[PatchTST]] — Teilt den Patching-Ansatz mit TimesFM

## Offene Fragen / Nächste Schritte

- **Evaluation starten**: Chronos-2 und TimesFM 2.5 Zero-Shot auf unseren Daten testen
- Vergleich mit aktuellem [[N-HiTS]]-Benchmark (MAE auf gleichen Backtest-Perioden)
- Klären: Fine-Tuning sinnvoll oder Zero-Shot ausreichend?
- GPU-Infrastruktur: reicht die lokale Hardware oder Cloud nötig?
- Können Foundation Models das [[Feature Engineering für Zeitreihen|Feature Engineering]] komplett ersetzen oder nur ergänzen?

## Benchmark-Überblick (Zero-Shot, fev-bench 2025)

| Modell | Skill Score | Covariate-Support | Multivariat |
|--------|-------------|-------------------|-------------|
| **Chronos-2** | **47.3%** | ✓ full | ✓ |
| TiRex | 42.6% | ✗ | ✗ |
| TimesFM-2.5 | 42.3% | teilweise | ✓ |
| Chronos-Bolt | 38.9% | ✗ | ✗ |

Chronos-2 gewinnt auf allen 3 grossen Benchmarks (fev-bench, GIFT-Eval, Chronos Benchmark II).

## Quellen

- [[chronos-2-paper]] — Ansari et al. (Amazon, 2025): Chronos-2 Architektur, Training, Benchmarks
