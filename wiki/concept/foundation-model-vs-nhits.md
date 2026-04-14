---
title: "Foundation Model vs. N-HiTS: Entscheidungsrahmen"
type: concept
tags: [forecasting, foundation-model, n-hits, modellwahl, vergleich]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---

# Foundation Model vs. N-HiTS: Entscheidungsrahmen

Wann lohnt sich der Wechsel von unserem etablierten [[N-HiTS]]-Ansatz zu einem Foundation Model wie [[Chronos-2 Paper|Chronos-2]]? Diese Seite dokumentiert die Abwägung.

## Architektur-Philosophie

**N-HiTS** ist *aufgabenspezifisch*: lernt ausschliesslich auf eigenen Daten. Die hierarchischen Stacks (grob → mittel → fein) passen sich der konkreten Frequenzstruktur der Zeitreihe an.

**Chronos-2** ist ein *Foundation Model*: vortrainiert auf ~700K diversen Zeitreihen + synthetischen Strukturen. Kein Wissen über spezifische Domänen — aber breites Wissen über allgemeine Zeitreihenmuster. Kern-Mechanismus ist **Group Attention** für In-Context Learning über verwandte Serien.

## Direkter Vergleich

| Dimension | N-HiTS | Chronos-2 |
|-----------|--------|-----------|
| **Training** | Auf eigenen Daten (mind. ~2 Jahre täglich) | Zero-Shot — kein Training nötig |
| **Feature Engineering** | Nötig (Lags, Kalender, Rolling) | Oft nicht nötig |
| **Multivariate** | Begrenzt (Multi-Series Training) | Vollständig (Group Attention) |
| **Covariates** | Exogene Features möglich | Past-Only + Known + kategorisch |
| **Probabilistisch** | Nur mit Konfiguration | Out-of-the-box (21 Quantile) |
| **Cold Start** | Schwach (braucht Historien) | Stark (wenig Daten reichen) |
| **Interpretierbarkeit** | Gering (SHAP nötig) | Noch geringer |
| **Infrastruktur** | CPU reicht | GPU empfohlen (~300 Serien/s auf A10G) |
| **Production-Reife** | Erprobt | Jünger, weniger Erfahrung |

## Entscheidungsbaum

```
Haben wir genug Trainingsdaten (>2 Jahre täglich)?
├── Nein → Chronos-2 bevorzugen (Cold Start Stärke)
└── Ja → Sind domänen-spezifische Muster entscheidend?
          ├── Ja → N-HiTS bevorzugen
          └── Nein → Benchmark-Vergleich nötig

Sind strukturierte Covariates vorhanden (Promotionen, Feiertage, etc.)?
└── Ja → Chronos-2 Known Covariates testen (+13–15 Skill-Score lt. Paper)

Wird Interpretierbarkeit gefordert?
└── Ja → N-HiTS + SHAP bevorzugen
```

## Wann N-HiTS gewinnt

- Genug Trainingsdaten vorhanden (N-HiTS profitiert mehr von eigenen Daten)
- Domain-spezifische Muster, die in keinem öffentlichen Datensatz vorkommen
- Interpretierbarkeit nötig — SHAP auf N-HiTS ist etabliert
- Ressourcen-begrenzt — N-HiTS läuft stabil auf CPU

## Wann Chronos-2 gewinnt

- **Cold Start / neue Produkte** ohne ausreichende Historien
- **Covariates nutzbar** (Promotionen, Feiertage) — direkt als Known Covariates → laut Paper grösster Vorteil aller getesteten Szenarien
- **Schnelles Prototyping** — zero-shot ohne Training-Pipeline
- **Viele kurze Zeitreihen** — Group Attention lernt quer über Serien

## Empfohlenes Vorgehen

Keine Entweder-oder-Entscheidung, sondern komplementärer Einsatz:

1. **Chronos-2 Small (28M) Zero-Shot** als neuen Benchmark auf bestehenden [[Backtesting|Backtest-Perioden]] laufen lassen — geringer Aufwand
2. **Cold-Start-Serien** (neue Produkte, wenig Historien): Chronos-2 einsetzen wo N-HiTS instabil ist
3. **Covariates strukturiert vorhanden**: Chronos-2 mit Known Covariates testen — stärkster Bereich laut [[chronos-2-paper]]
4. **Ensemble-Option**: N-HiTS + Chronos-2 Quantile kombinieren für robustere Vorhersagen

## Offene Fragen

- Wie gross ist der Unterschied auf unseren konkreten Daten? (Backtest-Vergleich steht aus)
- Lohnt sich Fine-Tuning von Chronos-2 auf eigenen Daten oder reicht Zero-Shot?
- Welche Covariates stehen strukturiert zur Verfügung?
- Reicht lokale Hardware für Chronos-2 Small (28M) im Batch-Betrieb?

## Verwandte Seiten

- [[N-HiTS]] — Aktuelles Produktionsmodell
- [[Foundation Models für Zeitreihen]] — Überblick Foundation Models
- [[chronos-2-paper]] — Technische Details Chronos-2
- [[Backtesting]] — Evaluierungs-Methodik für den Vergleich
- [[Feature Engineering für Zeitreihen]] — Was Foundation Models potenziell ersetzen
