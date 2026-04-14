# ML & Forecasting Wiki

Persönliches Teamwiki für Machine Learning, Time Series Forecasting und Data Science.
Gepflegt von einem LLM-Agenten (Claude Code), gelesen in Obsidian.

## Quickstart

1. Repository klonen
2. In [Obsidian](https://obsidian.md/) öffnen: "Open folder as vault" → `wiki/` Ordner auswählen
3. Obsidian-Einstellungen:
   - Unter "Files and links" → "New link format" auf **Shortest path** setzen
   - Community Plugins aktivieren: **Dataview** (optional, für dynamische Tabellen)
4. Wiki erkunden: `index.md` als Einstiegspunkt nutzen oder Graph View öffnen

## Mit Claude Code arbeiten

```bash
cd ml-wiki
claude   # Claude Code starten
```

Beispiel-Befehle:
- "Ingest das Paper in raw/chronos-2-paper.pdf"
- "Erstelle eine Concept-Seite über Backtesting"
- "Lint das Wiki — was fehlt?"
- "Was wissen wir über Feature Engineering für tägliche Forecasts?"

Claude Code liest `CLAUDE.md` automatisch und arbeitet gemäss den definierten Workflows.

## Verzeichnisstruktur

```
ml-wiki/
├── CLAUDE.md          ← LLM-Schema und Workflows
├── README.md          ← Diese Datei
├── raw/               ← Quellen (immutable)
├── wiki/
│   ├── index.md       ← Inhaltsverzeichnis
│   ├── log.md         ← Aktivitätsprotokoll
│   ├── entity/        ← Modelle, Tools, Pipelines
│   ├── concept/       ← Patterns und Konzepte
│   ├── decision/      ← Architekturentscheidungen
│   ├── source/        ← Quellen-Zusammenfassungen
│   └── runbook/       ← Operative Anleitungen
└── templates/         ← Vorlagen für neue Seiten
```

## Datenschutz

Dieses Wiki enthält **kein** sensibles Geschäftswissen, keine Kundendaten,
Umsatzzahlen, konkreten Mengen oder interne Organisationsdetails.
Es dokumentiert technisches Handwerkswissen: ML-Konzepte, Patterns,
Architekturentscheidungen und Troubleshooting-Guides.
