# Outlier Detection – Ergebnisse für die Präsentation

Erzeugt von `notebooks/02_outlier_detection.ipynb` (Hilfsfunktionen: `src/outliers.py`).
Datengrundlage: 3.839 Briefe; 13 handgebaute numerische Features (keine Embeddings, kein TF-IDF, keine Dimensionsreduktion).
Alle Abbildungen liegen als PNG (300 dpi) und SVG in `figures/outliers/`.

## Kernaussagen für die Folien

1. **Die Rohfeatures sind stark rechtsschief.** Die Skewness von `n_words` sinkt durch `log1p` von 3.05 auf -0.27.
   Die Anteils-Features (`share_*`) bleiben dagegen schief (z. B. `share_upper` 7.68 → 5.82).
   Parametrische Verfahren gelten hier also nur eingeschränkt.
2. **Der IQR-Faktor ist subjektiv:** Mit 1.5·IQR fallen 1.143 Briefe (30 %)
   in mindestens einem der 9 stetigen Features auf, mit 3·IQR nur noch 352.
3. **Korrelationen und Masking zählen:** euklidisch 149, Mahalanobis klassisch 128,
   robust (MCD) 533 Briefe über dem 99,5-%-Quantil von χ²_8.
4. **Konsens statt Einzelmethode:** 135 Briefe werden von ≥ 3 der 8 Methoden markiert. Davon sind
   47 % vorläufig Datenprobleme, 33 % mögliche echte Sonderfälle und 19 % unklar.
5. **Für die Pipeline werden nur 27 Briefe (0.7 %) ausgeschlossen** (Fragmente < 20 Wörter, Archivverweise).
   122 Sprachangaben sind zu prüfen, Sonderfälle und kurze Telegramme bleiben markiert im Datensatz.

## Markierte Briefe pro Methode

| Methode | Quelle | markiert |
|---|---|---|
| `iqr_any` | Vorlesung | 1143 |
| `zscore_any` | Vorlesung | 45 |
| `maha_classic` | Vorlesung | 128 |
| `maha_robust` | Erweiterung | 533 |
| `hull` | Vorlesung | 37 |
| `abod` | Vorlesung | 39 |
| `iforest` | Erweiterung | 39 |
| `lof` | Erweiterung | 39 |

## Abbildungen

| Datei | Erklärung |
|---|---|
| `boxplots_features` | Boxplots aller 13 Features; rote Punkte liegen außerhalb der 1.5·IQR-Whisker. Starke Rechtsschiefe; bei `n_cases`/`n_attachments` ist IQR = 0, dort ist jede Abweichung vom Normalfall ein „Ausreißer“. |
| `boxplot_length_by_language` | Textlänge (log) nach Sprache und nach doc_style. Die Längenverteilungen unterscheiden sich zwischen Gruppen, ein globaler Schwellwert benachteiligt daher einzelne Sprachen/Formate. |
| `scatter_length_vs_nonalpha` | Visuelle 2D-Prüfung: sehr kurze Texte mit hohem Nicht-Buchstaben-Anteil sind Archivverweise statt Briefe. |
| `qq_before_after_n_words` | QQ-Plot von `n_words` vor/nach `log1p`: Aus der gekrümmten Kurve wird nahezu eine Gerade, die Voraussetzung für z-Score und Mahalanobis. |
| `qq_all_before`, `qq_all_after` | QQ-Plots aller stetigen Features vor/nach Transformation; die `share_*`-Features bleiben nicht normal. |
| `chi2_euclid_classic_robust` | χ²-Plots für euklidische, klassische und robuste (MCD) Mahalanobis-Distanz. Der rechte Rand weicht deutlich von der Steigung-1-Linie ab; MCD deckt maskierte Ausreißer auf. |
| `chi2_euclid_vs_classic` | Nur euklidisch vs. klassisch (Schritt 6). |
| `hull_peeling` | Convex Hull Peeling in 2D (log n_words vs. share_non_alpha bzw. n_persons); die äußeren 2 Layer gelten als Ausreißer. |
| `score_histograms` | Score-Verteilungen von ABOD (log ABOF, niedrig = Ausreißer), Isolation Forest und LOF mit 1-%-Grenze. |
| `method_overlap_heatmap` | Jaccard-Überlappung der markierten Mengen; zeigt, welche Methoden ähnliche Briefe finden. |
| `flags_per_method` | Anzahl markierter Briefe je Methode (blau = Vorlesung, türkis = Erweiterung). |
| `correlation_heatmap` | Korrelationsmatrix der Features; `n_words`/`n_chars` sind redundant (r ≈ 0,997), deshalb ist `n_chars` nicht in der Mahalanobis-Distanz. |

## Tabellen (`results/outliers/`)

- `features.csv` – Feature-Tabelle pro Brief
- `languages.csv` – main_language, lang_detected, language_filled pro Brief
- `iqr_flags_per_feature.csv`, `skewness_before_after.csv`, `method_jaccard.csv`
- `outlier_overview.csv` – eine Zeile pro Brief, Bool-Spalte je Methode, `n_flags_lecture`, `n_flags_all`, Scores, Kategorie
- `top_outliers.csv` – Top-20 mit Text-Anfang, GPT-4o-Summary und leerer Spalte `manual_label`
- `outlier_flags_for_pipeline.csv` – `id`, `exclude`, `reason` für die weitere Pipeline
- `outlier_stats.json` – alle Kennzahlen
