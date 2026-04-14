# ML Wiki — Setup-Anleitung Schritt für Schritt

## Schritt 0: Privacy-Einstellungen prüfen (WICHTIG — vor allem anderen)

### 0.1 Training Opt-out auf claude.ai

Das ist die wichtigste Einstellung. Sie verhindert, dass eure Chats und
Claude-Code-Sessions für Modelltraining verwendet werden.

1. Öffne https://claude.ai und melde dich an
2. Klicke auf dein **Profilbild** (oben rechts) → **Settings**
3. Gehe zu **Privacy** → **Privacy Settings**
   (Direktlink: https://claude.ai/settings/data-privacy-controls)
4. Suche die Option **"Help improve Claude"** (oder ähnlich benannt)
5. Stelle sicher, dass der Toggle **AUS** ist

**Was das bewirkt:**

- AUS: Deine Daten werden NICHT für Modelltraining verwendet.
  Retention: 30 Tage, dann gelöscht.
- AN: Deine Daten können für Training verwendet werden.
  Retention: bis zu 5 Jahre.

**Achtung:** Diese Einstellung gilt sowohl für claude.ai Chats als auch
für Claude Code Sessions mit demselben Account.

### 0.2 Zusätzliche Telemetrie in Claude Code reduzieren (optional)

Wenn du Claude Code installiert hast, kannst du zusätzliche
Telemetrie-Daten (Latenz-Metriken, Error-Logs) deaktivieren.
Diese sind getrennt vom Training-Opt-in, aber reduzieren generell,
was deine Maschine verlässt.

Füge diese Zeilen zu deiner Shell-Konfiguration hinzu:

**PowerShell** (Windows, in `$PROFILE`):

```powershell
$env:DISABLE_TELEMETRY = "1"
$env:DISABLE_ERROR_REPORTING = "1"
```

**Bash/Zsh** (in `~/.bashrc` oder `~/.zshrc`):

```bash
export DISABLE_TELEMETRY=1
export DISABLE_ERROR_REPORTING=1
```

### 0.3 Was trotzdem zu Anthropic geht

Auch mit deaktiviertem Training:

- Jede Datei, die Claude Code explizit liest, wird an die Anthropic API
  gesendet (verschlüsselt über TLS), dort verarbeitet, und die Antwort
  kommt zurück.
- Backend-Logs werden 30 Tage aufbewahrt, dann gelöscht.
- Bei Safety-Verstössen können Daten bis zu 2 Jahre aufbewahrt werden.
- Wenn du /feedback oder den Thumbs-down-Button nutzt, wird die gesamte
  zugehörige Konversation bis zu 10 Jahre gespeichert.

**Faustregel für das Wiki:** Keine Kundennamen, keine Umsatzzahlen,
keine konkreten Tonnagen, keine internen Organisationsdetails.
Technisches Wissen (Patterns, Konzepte, Architekturentscheidungen) ist OK.

---

## Schritt 1: Voraussetzungen installieren

### 1.1 Git installieren (falls noch nicht vorhanden)

Git brauchst du für Versionierung des Wikis UND für Claude Code auf Windows.

**Prüfen ob Git installiert ist:**

```bash
git --version
```

Falls nicht installiert:
- Gehe zu https://git-scm.com/install/win
- Installer herunterladen und ausführen
- Bei allen Auswahlbildschirmen die Defaults belassen und "Next" klicken
- Nach der Installation: Terminal neu öffnen und `git --version` erneut prüfen

### 1.2 Claude Code installieren

Claude Code braucht einen **Claude Pro, Max, Team oder Enterprise Account**.
Der kostenlose Plan reicht nicht.

**Native Installer (empfohlen, kein Node.js nötig):**

Öffne **PowerShell** (Start → "Terminal" suchen → öffnen) und führe aus:

```powershell
irm https://code.claude.com/install.ps1 | iex
```

**Oder via npm** (falls Node.js 18+ bereits installiert):

```bash
npm install -g @anthropic-ai/claude-code
```

**Installation prüfen:**

```bash
claude --version
```

Du solltest eine Versionsnummer sehen.

### 1.3 Beim ersten Start authentifizieren

```bash
claude
```

Claude Code öffnet automatisch den Browser zur Anmeldung.
Folge den Anweisungen um deinen Anthropic-Account zu autorisieren.
Danach bist du authentifiziert und kannst Claude Code verwenden.

### 1.4 Obsidian installieren

1. Gehe zu https://obsidian.md/download
2. Lade die Windows-Version herunter und installiere sie
3. Beim ersten Start: "Open folder as vault" wählen (NICHT "Create new vault")
   → Aber noch keinen Ordner auswählen, das kommt in Schritt 3

---

## Schritt 2: Wiki entpacken und Git einrichten

### 2.1 Archiv entpacken

Das `ml-wiki.tar.gz` Archiv enthält die komplette Wiki-Struktur.

Wähle einen Ort, an dem das Wiki dauerhaft leben soll.
Empfehlung: ein eigener Ordner in deinem Projekte-Verzeichnis.

**In PowerShell oder Git Bash:**

```bash
# In dein Projekte-Verzeichnis wechseln (Beispiel)
cd C:\Users\DEIN_USERNAME\Projekte

# Archiv entpacken
tar -xzf C:\Users\DEIN_USERNAME\Downloads\ml-wiki.tar.gz
```

Danach hast du einen Ordner `ml-wiki/` mit dieser Struktur:

```
ml-wiki/
├── CLAUDE.md
├── README.md
├── .gitignore
├── raw/
│   └── README.md
├── templates/
│   ├── entity.md
│   ├── concept.md
│   ├── decision.md
│   ├── source.md
│   └── runbook.md
└── wiki/
    ├── index.md
    ├── log.md
    ├── entity/
    │   ├── n-hits.md
    │   ├── sarimax.md
    │   └── foundation-models-zeitreihen.md
    ├── concept/
    │   ├── hyperparameter-tuning-optuna.md
    │   ├── shap-explainability.md
    │   └── feature-engineering-zeitreihen.md
    └── decision/
        └── adr-001-nhits-statt-prophet.md
```

### 2.2 Git Repository initialisieren

```bash
cd ml-wiki

# Git-Repo initialisieren
git init

# Deinen Namen und Email setzen (für Commit-Historie)
git config user.name "Dein Name"
git config user.email "deine.email@pistor.ch"

# Alle Dateien stagen
git add .

# Ersten Commit erstellen
git commit -m "Wiki-Grundstruktur: Schema, Templates, 7 initiale Seiten"
```

**Warum Git?** Jede Wiki-Änderung ist versioniert. Wenn Claude Code
eine Seite falsch updatet, kannst du mit `git diff` sehen was sich
geändert hat und mit `git checkout -- wiki/entity/n-hits.md` zurücksetzen.
Ausserdem kannst du das Repo später auf euren internen Git-Server
(GitHub/GitLab) pushen, damit das Team Zugriff hat.

---

## Schritt 3: In Obsidian als Vault öffnen

### 3.1 Vault erstellen

1. Öffne Obsidian
2. Klicke "Open folder as vault"
3. Navigiere zu deinem `ml-wiki/wiki/` Ordner und wähle ihn aus

**Wichtig:** Öffne den `wiki/` Unterordner als Vault, NICHT den
`ml-wiki/` Root-Ordner. Der Grund: Obsidian zeigt dann nur die
Wiki-Seiten (entity/, concept/, decision/, etc.) und nicht die
Templates oder Raw-Quellen. Das hält die Navigation sauber.

### 3.2 Obsidian-Einstellungen anpassen

Nach dem Öffnen:

1. **Settings** (Zahnrad unten links) → **Files and links**
   - "New link format" → **Shortest path when possible**
     (damit `[[n-hits]]` statt `[[entity/n-hits]]` reicht)
   - "Default location for new attachments" → **In subfolder under current folder**

2. **Settings** → **Core plugins**
   - Stelle sicher, dass **Graph view** aktiviert ist

3. Optional: **Settings** → **Community plugins** → **Browse**
   - Suche nach **Dataview** und installiere es
     (damit kannst du später dynamische Tabellen über Frontmatter-Felder bauen,
      z.B. "alle Entities mit status: draft")

### 3.3 Wiki erkunden

- Öffne `index.md` — das ist dein Einstiegspunkt
- Klicke auf einen `[[link]]` um zur verlinkten Seite zu springen
- Öffne den **Graph View** (linke Seitenleiste oder Ctrl+G):
  Du siehst sofort das Netzwerk der verlinkten Seiten

---

## Schritt 4: Erster Ingest mit Claude Code testen

### 4.1 Beispiel-Quelle vorbereiten

Für den ersten Test brauchst du eine Quelle. Am einfachsten:
ein Paper oder Artikel als PDF oder Markdown.

**Option A — Chronos-2 Paper (empfohlen für den PoC):**

1. Gehe zu https://arxiv.org und suche nach "Chronos 2 time series"
2. Lade das PDF herunter
3. Kopiere es nach `ml-wiki/raw/`:

```bash
cp ~/Downloads/chronos-2-paper.pdf ml-wiki/raw/
```

**Option B — Artikel als Markdown:**

Falls du den Obsidian Web Clipper installiert hast, kannst du einen
Blog-Artikel (z.B. von Nixtla oder Hugging Face über TimesFM)
direkt als Markdown clippen und nach `raw/` legen.

**Option C — Eigene Notizen:**

Du kannst auch eigene Notizen als `.md` in `raw/` legen, z.B.
Experiment-Logs oder Meeting-Notes zum Forecasting-Thema.

### 4.2 Claude Code starten und CLAUDE.md prüfen

```bash
# In den Wiki-Ordner wechseln
cd ml-wiki

# Claude Code starten
claude
```

Claude Code liest automatisch die `CLAUDE.md` im Root-Verzeichnis
und kennt damit die Wiki-Konventionen, Seitentypen und Workflows.

**Tipp:** Beim ersten Start kannst du kurz prüfen:

```
> Lies die CLAUDE.md und bestätige, dass du die Wiki-Struktur verstehst.
```

Claude sollte die Seitentypen (entity, concept, decision, source, runbook),
die Frontmatter-Konventionen und den Ingest-Workflow wiedergeben können.

### 4.3 Ingest durchführen

Jetzt der eigentliche Test. Sage Claude Code:

```
> Ingest raw/chronos-2-paper.pdf
```

Claude Code wird dann gemäss dem Workflow in CLAUDE.md:

1. Das Paper lesen und die Key Takeaways zusammenfassen
2. Eine Source-Seite erstellen: `wiki/source/chronos-2-paper.md`
3. Die Entity-Seite `wiki/entity/foundation-models-zeitreihen.md` updaten
   (mit konkreten Infos aus dem Paper)
4. Eventuell die Concept-Seite `wiki/concept/feature-engineering-zeitreihen.md`
   updaten (wenn das Paper Feature-Engineering-Aspekte behandelt)
5. Neue `[[links]]` setzen wo sinnvoll
6. Den `wiki/index.md` aktualisieren
7. Einen Eintrag in `wiki/log.md` schreiben

**Beobachte in Obsidian:** Du siehst in Echtzeit, wie neue Dateien
erscheinen und bestehende sich ändern. Im Graph View tauchen neue
Knoten und Verbindungen auf.

### 4.4 Ergebnis prüfen und committen

Nach dem Ingest:

1. **In Obsidian:** Prüfe die erstellte Source-Seite und die Updates
2. **Im Terminal:** Prüfe was sich geändert hat:

```bash
git diff                    # Zeigt Änderungen in bestehenden Dateien
git status                  # Zeigt neue Dateien
```

3. Wenn du zufrieden bist, committen:

```bash
git add .
git commit -m "Ingest: Chronos-2 Paper — Source + Entity-Updates"
```

### 4.5 Erste Query testen

Nach dem Ingest kannst du das Wiki befragen:

```
> Was wissen wir über Foundation Models für Zeitreihen?
  Vergleiche Chronos-2 mit unserem aktuellen N-HiTS Ansatz.
```

Claude Code liest den Index, findet die relevanten Seiten, und
synthetisiert eine Antwort. Wenn die Antwort wertvoll ist:

```
> Speichere diese Analyse als neue Wiki-Seite
  wiki/concept/foundation-models-vs-nhits.md
```

So wächst das Wiki mit jeder Frage.

---

## Schritt 5: Weiter ausbauen

### Nächste sinnvolle Ingests

- TimesFM 2.5 Paper/Dokumentation
- Eigene Experiment-Logs (z.B. N-HiTS Tuning-Ergebnisse, anonymisiert)
- NeuralForecast Dokumentation
- Optuna Best Practices Artikel

### Nächste sinnvolle Seiten (manuell oder per Query)

Die 15 fehlenden Seiten aus dem Index sind gute Kandidaten:

```
> Erstelle eine Concept-Seite über Backtesting für Zeitreihen.
  Nutze dein Wissen und verlinke auf bestehende Wiki-Seiten.
```

### Erster Lint

Nach 3-4 Ingests:

```
> Lint das Wiki. Prüfe auf fehlende Querverweise,
  Widersprüche und verwaiste Seiten.
```

---

## Troubleshooting

### Claude Code findet CLAUDE.md nicht
Stelle sicher, dass du Claude Code im `ml-wiki/` Root-Ordner startest,
nicht in einem Unterordner.

### Obsidian zeigt [[links]] nicht als klickbar
Prüfe, dass du den `wiki/` Ordner als Vault geöffnet hast und dass
"New link format" auf "Shortest path" steht.

### Claude Code ändert Dateien die es nicht soll
Nutze `git diff` um Änderungen zu prüfen. Mit `git checkout -- datei.md`
kannst du einzelne Dateien zurücksetzen.

### Ingest dauert sehr lange
PDFs mit vielen Seiten brauchen mehr Zeit. Für den Anfang:
arbeite mit kürzeren Quellen (Blog-Artikel, einzelne Paper-Kapitel).
