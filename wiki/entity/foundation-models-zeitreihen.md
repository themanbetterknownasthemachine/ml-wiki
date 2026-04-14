---
title: "Foundation Models für Zeitreihen"
type: entity
tags: [forecasting, foundation-model, chronos, timesfm, deep-learning]
status: draft
erstellt: 2026-04-14
aktualisiert: 2026-04-14
quellen: 0
---

# Foundation Models für Zeitreihen

## Überblick

Foundation Models für Zeitreihen sind vortrainierte Modelle, die auf grossen Mengen diverser Zeitreihen trainiert wurden und ohne oder mit minimalem Fine-Tuning auf neue, ungesehene Zeitreihen angewendet werden können (Zero-Shot oder Few-Shot Forecasting). Sie übertragen die Idee von GPT/BERT auf die Zeitreihen-Domäne.

## Kandidaten (Stand April 2026)

### Chronos-2 (Amazon)
- Nachfolger von Chronos (2024), basiert auf T5-Architektur
- Tokenisiert Zeitreihenwerte ähnlich wie Text-Tokens
- Vortrainiert auf Millionen von Zeitreihen (synthetisch + real)
- Verschiedene Grössen verfügbar (Mini bis Large)
- Open Source (Apache 2.0)

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

## Quellen
