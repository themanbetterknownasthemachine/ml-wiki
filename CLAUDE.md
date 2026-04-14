# ML & Forecasting Wiki — Schema

> Dieses Wiki dokumentiert das akkumulierte Wissen des BI & ML Teams im Bereich
> Machine Learning, Time Series Forecasting und Data Science. Es wird von einem
> LLM-Agenten (Claude Code) gepflegt und in Obsidian gelesen.

## Grundregeln

- **Sprache**: Deutsch. Technische Fachbegriffe (MAE, SHAP, Hyperparameter, etc.) auf Englisch belassen.
- **Keine sensiblen Geschäftsdaten**: Keine Kundennamen, Umsatzzahlen, konkreten Tonnagen, Organisationsstrukturen oder interne Stakeholder-Namen. Das Wiki enthält technisches Wissen, keine Business-KPIs.
- **Obsidian-kompatibel**: Alle Seiten sind Markdown mit `[[wiki-links]]` für Querverweise. YAML-Frontmatter auf jeder Seite.
- **Eigentum**: Das LLM schreibt und pflegt alle Wiki-Seiten. Raw Sources werden nie verändert.

## Verzeichnisstruktur

```
ml-wiki/
├── CLAUDE.md              ← Dieses Schema (LLM-Konfiguration)
├── raw/                   ← Immutable Quellen (Papers, Artikel, Logs)
├── wiki/
│   ├── index.md           ← Inhaltsverzeichnis aller Wiki-Seiten
│   ├── log.md             ← Chronologisches Aktivitätsprotokoll
│   ├── entity/            ← Systeme, Modelle, Pipelines, Tools
│   ├── concept/           ← Wiederverwendbare Patterns und Konzepte
│   ├── decision/          ← Architekturentscheidungen (ADRs)
│   ├── source/            ← Zusammenfassungen von Quellen
│   └── runbook/           ← Operative Schritt-für-Schritt Anleitungen
└── templates/             ← Vorlagen für neue Seiten
```

## Seitentypen und Frontmatter

### Entity (`wiki/entity/`)
Ein konkretes System, Modell, Tool oder Pipeline-Komponente.

```yaml
---
title: "Name des Systems/Modells"
type: entity
tags: [forecasting, modell, pipeline]  # relevante Tags
status: aktuell                        # draft | aktuell | veraltet
erstellt: 2026-04-14
aktualisiert: 2026-04-14
quellen: 0                            # Anzahl verlinkter Quellen
---
```

### Concept (`wiki/concept/`)
Ein wiederverwendbares Pattern, eine Technik oder ein Konzept.

```yaml
---
title: "Name des Konzepts"
type: concept
tags: [feature-engineering, zeitreihen]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---
```

### Decision (`wiki/decision/`)
Eine Architekturentscheidung mit Kontext, Alternativen und Begründung.

```yaml
---
title: "ADR-NNN: Kurztitel"
type: decision
tags: [modellwahl, architektur]
status: aktuell          # aktuell | überholt
entschieden: 2026-04-14
beteiligte: []           # Rollen, keine Namen
---
```

### Source (`wiki/source/`)
Zusammenfassung einer externen Quelle (Paper, Artikel, Vortrag).

```yaml
---
title: "Titel der Quelle"
type: source
tags: [paper, foundation-model]
quelle_typ: paper        # paper | artikel | vortrag | dokumentation
url: ""
autoren: []
datum: 2026-01-01
---
```

### Runbook (`wiki/runbook/`)
Operative Anleitung für wiederkehrende Aufgaben.

```yaml
---
title: "Aufgabe beschreiben"
type: runbook
tags: [monitoring, training]
status: aktuell
zuletzt_getestet: 2026-04-14
---
```

## Workflows

### Ingest (neue Quelle verarbeiten)

1. Quelle in `raw/` ablegen (nie verändern)
2. Quelle lesen, Key Takeaways mit dem User besprechen
3. `wiki/source/` Zusammenfassung erstellen
4. Betroffene Entity- und Concept-Pages identifizieren und updaten
5. Neue Querverweise (`[[links]]`) setzen wo sinnvoll
6. Widersprüche zu bestehenden Seiten **explizit markieren** mit `> ⚠️ Widerspruch: ...`
7. `wiki/index.md` aktualisieren
8. `wiki/log.md` Eintrag schreiben

### Query (Frage beantworten)

1. `wiki/index.md` lesen um relevante Seiten zu finden
2. Relevante Seiten lesen und Antwort synthetisieren
3. Wertvolle Antworten (Analysen, Vergleiche) als neue Wiki-Seite ablegen
4. Log-Eintrag schreiben

### Lint (Wiki-Gesundheitscheck)

Prüfe auf:
- Widersprüche zwischen Seiten
- Veraltete Informationen (status: veraltet ohne Nachfolger)
- Orphan-Seiten (keine eingehenden Links)
- Erwähnte Konzepte ohne eigene Seite
- Fehlende Querverweise
- Wissenslücken die mit einer Web-Suche gefüllt werden könnten

## Domänen-spezifische Regeln

### ML-Modelle
- Immer angeben: Modelltyp, Architektur-Kernidee, typische Hyperparameter
- Metriken generisch halten (MAE, MAPE, RMSE erklären, keine konkreten Produktionswerte)
- Stärken/Schwächen im Vergleich zu Alternativen dokumentieren

### Forecasting
- Zeitreihen-spezifische Aspekte immer berücksichtigen: Saisonalität, Trend, Stationarität
- Feature Engineering separat dokumentieren (welche Features, warum, wie berechnet)
- Train/Val/Test Split-Strategie und Backtesting erklären

### Code-Beispiele
- Python bevorzugt (pandas, numpy, scikit-learn, PyTorch, LightGBM, NeuralForecast)
- Code muss lauffähig und verständlich sein
- Immer kommentieren warum, nicht nur was

### Verlinkung
- Jede Seite sollte mindestens 2-3 ausgehende `[[links]]` haben
- Bei Modell-Entities: immer auf verwendete Concepts linken
- Bei Concepts: immer auf Entities linken die das Concept nutzen
