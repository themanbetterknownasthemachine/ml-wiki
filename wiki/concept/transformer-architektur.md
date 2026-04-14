---
title: "Transformer Architektur"
type: concept
tags: [deep-learning, transformer, attention, architektur, foundation-model]
status: aktuell
erstellt: 2026-04-14
aktualisiert: 2026-04-14
---

# Transformer Architektur

Der Transformer ist die Basis der meisten modernen Deep-Learning-Modelle — von Sprachmodellen (GPT, BERT) bis zu Zeitreihen-Foundation-Models wie [[Foundation Models für Zeitreihen|Chronos-2]]. Das Kernprinzip: **Attention statt Rekursion**.

## Das Problem vor dem Transformer

Ältere Sequenzmodelle (RNN, LSTM) verarbeiten Zeitreihen Schritt für Schritt. Das hat zwei Nachteile:
- **Kein Paralleltraining**: Jeder Schritt hängt vom vorherigen ab
- **Vergessen**: Lange Abhängigkeiten gehen im Hidden State verloren

Der Transformer löst beide Probleme durch **globale, parallele Attention**.

## Self-Attention: Q, K, V

Das Herzstück des Transformers. Jedes Element einer Sequenz fragt: *Welche anderen Elemente sind für mich relevant?*

```
Eingabe: Sequenz X ∈ R^(n × d)

Q = X · W_Q    (Query:  "Was suche ich?")
K = X · W_K    (Key:    "Was biete ich an?")
V = X · W_V    (Value:  "Was gebe ich weiter?")

Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

- **Query** (Q): Repräsentiert den aktuellen Token/Patch — "Was ist mein Kontext-Bedarf?"
- **Key** (K): Repräsentiert jeden anderen Token — "Wie gut passe ich zu diesem Query?"
- **Value** (V): Der eigentliche Inhalt, der weitergegeben wird wenn Attention hoch ist
- **Skalierung `√d_k`**: Verhindert zu grosse Dot-Products bei hoher Dimension (würden Softmax sättigen)

Das Ergebnis: jede Position bekommt eine **gewichtete Summe aller anderen Positionen**, wobei die Gewichte durch Query-Key-Ähnlichkeit bestimmt werden.

### Multi-Head Attention

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
wobei head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

Mehrere Attention-Köpfe lernen **verschiedene Aspekte** der Abhängigkeiten gleichzeitig — z.B. kurzfristige Muster in einem Kopf, saisonale Abhängigkeiten in einem anderen.

## Positional Encoding: Wie kommt Zeit ins Modell?

Attention selbst ist **positionsunabhängig** — die Reihenfolge der Tokens spielt keine Rolle. Zeitreihen brauchen aber temporale Ordnung. Lösung: Positions-Information explizit hinzufügen.

### Sinusoidales Positional Encoding (original)
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Absolute Position als festes Signal addiert auf Embeddings.

### Rotary Position Embeddings (RoPE)
Moderner Ansatz — verwendet von [[Foundation Models für Zeitreihen|Chronos-2]] und vielen aktuellen LLMs.

**Idee**: Statt absolute Positionen zu addieren, wird die **relative Position** direkt in die Query-Key-Interaktion eingebaut:

```
Attention(q_m, k_n) hängt nur von (q_m, k_n, m-n) ab
```

Vorteile gegenüber sinusoidalem PE:
- **Bessere Extrapolation**: Funktioniert auch für längere Sequenzen als im Training gesehen
- **Relative Abstände**: Das Modell lernt "2 Schritte entfernt" statt "an Position 42"
- Für Zeitreihen besonders wertvoll: *Wie weit ist dieser Patch vom aktuellen entfernt?* ist relevanter als *Welcher absolute Zeitindex ist das?*

## Encoder vs. Decoder

| Typ | Beschreibung | Beispiele |
|-----|-------------|---------|
| **Encoder-only** | Liest gesamte Sequenz, bidirektionale Attention | BERT, Chronos-2 |
| **Decoder-only** | Autoregressive Generierung, nur vergangene Tokens sichtbar | GPT, TimesFM |
| **Encoder-Decoder** | Encoder komprimiert Input, Decoder generiert Output | T5, Chronos-Bolt |

Für Zeitreihen-Forecasting:
- **Encoder-only** (Chronos-2): Verarbeitet historischen Kontext vollständig → gibt Quantile direkt aus
- **Decoder-only** (TimesFM): Generiert Forecast autoregressiv, ein Token nach dem anderen

## Patching: Zeitreihen als Transformer-Input

Rohe Zeitreihenwerte direkt als Tokens zu verwenden ist ineffizient (lange Sequenzen, viel Rauschen). Stattdessen: **Patches**.

```
Zeitreihe: [t1, t2, t3, t4, t5, t6, t7, t8, ...]
            |___Patch 1___|  |___Patch 2___|

Patch-Embedding → Transformer verarbeitet Patch-Sequenz
```

Vorteile:
- Kürzere Sequenz → quadratische Attention-Kosten sinken (O(n²) → O((n/P)²))
- Jeder Patch fasst lokale Muster zusammen (implizite Glättung)
- Ermöglicht lange Kontextfenster (Chronos-2: bis 8192 Token = sehr viele Patches)

Verwendet von: [[Foundation Models für Zeitreihen|Chronos-2]], TimesFM, [[PatchTST]]

## Group Attention (Chronos-2 Erweiterung)

Klassische Self-Attention läuft *innerhalb* einer Zeitreihe. Chronos-2 ergänzt:

- **Time Attention**: über Patches einer einzelnen Serie (temporal)
- **Group Attention**: über Serien innerhalb einer Gruppe (cross-series)

Das ermöglicht In-Context Learning: das Modell lernt beim Inference-Zeitpunkt aus verwandten Serien — ohne Neutraining. Siehe [[chronos-2-paper]].

## Warum Transformer für Zeitreihen?

| Vorteil | Erklärung |
|---------|-----------|
| **Globale Abhängigkeiten** | Attention verbindet direkt t=1 mit t=500 — keine Degradation wie bei RNNs |
| **Saisonalität** | Attention lernt automatisch periodische Muster (Jahrestag ↔ Jahrestag) |
| **Paralleltraining** | Gesamte Sequenz auf einmal — skaliert gut auf grosse Datasets |
| **Flexible Kontextlänge** | Mit RoPE gut generalisierbar auf ungesehene Längen |

**Schwäche**: Quadratische Attention-Komplexität O(n²) — bei sehr langen Zeitreihen teuer. Patching und effiziente Attention-Varianten mildern das.

## Verwandte Seiten

- [[Foundation Models für Zeitreihen]] — nutzt Transformer als Basis (Chronos-2, TimesFM)
- [[chronos-2-paper]] — Encoder-only + Group Attention + RoPE im Detail
- [[N-HiTS]] — bewusst *kein* Transformer (MLP-basiert, effizienter bei tabellarischen Daten)
- [[PatchTST]] — reiner Transformer-Ansatz für Zeitreihen ohne Foundation-Model-Pretraining
- [[backtesting]] — Evaluation von Transformer-basierten Modellen
