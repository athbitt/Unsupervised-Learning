# Dimensionality Reduction – Ergebnisse für die Präsentation

Erzeugt von `notebooks/03_dimensionality_reduction.ipynb` (Hilfsfunktionen: `src/dimred.py`).
Datengrundlage: 3.812 Briefe (27 laut Outlier-Notebook ausgeschlossen: Fragmente/Archivverweise).
Alle Abbildungen liegen als PNG (300 dpi) und SVG in `figures/dimred/`, die 2D-Projektionen als `.npy` in `results/dimred/projections/`.

## Kernaussagen für die Folien

1. **Features: wenige, gut interpretierbare Achsen.** Die Korrelations-PCA der 9 Text-/Metadaten-Features erklärt mit PC1–PC3
   36 % / 17 % / 12 % der Varianz.
   Empfohlene Komponentenzahl: Kaiser 3, 80-%-Regel 5, Parallel Analysis 3.
2. **Ohne Standardisierung dominiert die Skala:** In der Kovarianz-PCA bestimmen die Längen-Features PC1, die Anteils-Features gehen unter (Gewicht ≈ 0).
3. **TF-IDF ist hochdimensional und dünn:** 12.546 Terme, 99.3 % Nullen.
   2 SVD-Komponenten erklären nur 2.4 % der Varianz, 100 Komponenten 18.5 %.
4. **TF-IDF trennt vor allem nach Sprache:** In allen Verfahren bilden Italienisch, Deutsch und Französisch getrennte Inseln.
   Inhaltliche Struktur (Anliegen) wird erst in der Variante „nur Italienisch“ sichtbar.
5. **Lokal vs. global (Vorlesung bestätigt):** UMAP erhält 20 % der 10 nächsten Nachbarn,
   t-SNE 24 %, PCA/LSA nur 2 %.
   Globale Distanzen (Spearman ρ): PCA/LSA 0.69, UMAP 0.70,
   t-SNE 0.56.
6. **Outlier verschwinden in 2D:** Im TF-IDF-Raum liegen die Outlier deutlich isolierter (Median-Perzentil der 10-NN-Distanz
   73 vs. 50), in der UMAP-Projektion nicht
   (42 vs. 50). Outlier also im Originalraum suchen.
7. **Linearer Autoencoder vs. PCA:** Eine Richtung trifft PC2 (Winkel 3°), die andere nicht
   (66°), weil λ₂ ≈ λ₃ (4.36 vs. 4.33).
   Der 2D-Unterraum ist damit kaum eindeutig bestimmt.

## Abbildungen (`figures/dimred/`)

| Datei | Erklärung |
|---|---|
| `curse_of_dimensionality` | Simulation Unif([0,1]^d): Der relative Distanzkontrast fällt von d = 2 bis d = 1000 um Größenordnungen, die Distanzverteilung wird schmal. |
| `curse_of_dimensionality_tfidf` | Echte Daten: Die Kosinus-Distanzen der TF-IDF-Vektoren ballen sich nahe 1 (fast alle Briefe gleich unähnlich), anders als im 9-Feature-Raum. |
| `pca_cov_vs_corr` | PCA auf Kovarianz vs. Korrelation: Ohne Standardisierung dominieren die großvarianten Längen-Features PC1. |
| `scree_kaiser_parallel` | Scree-Plot mit Kaiser-Linie (λ = 1) und Parallel Analysis (95-%-Quantil von 100 Permutationen). |
| `cumulative_variance` | Kumulierte erklärte Varianz der Feature-PCA mit 80-%-Schwelle. |
| `loadings_heatmap` | Loadings der ersten 4 PCs, Grundlage für die inhaltliche Benennung der Achsen. |
| `biplot_pc1_pc2`, `biplot_pc1_pc3` | Biplots (Scores + Loading-Pfeile), gefärbt nach Sprache bzw. Outlier-Flag. |
| `svd_explained_variance` | Erklärte Varianz der TruncatedSVD (alle Sprachen / nur Italienisch). |
| `svd_top_terms` | Terme mit den stärksten positiven/negativen Ladungen der ersten 5 SVD-Komponenten. |
| `projection_pca_all`, `projection_tsne_all`, `projection_umap_all`, `projection_pacmap_all` | Je eine Projektion, gefärbt nach allen 6 Färbe-Variablen. |
| `tsne_perplexity_grid` | t-SNE mit Perplexity 5/30/50/100: Kleine Werte zersplittern, große glätten. |
| `tsne_init_comparison`, `tsne_seeds` | Initialisierung pca vs. random und drei Seeds: Lage und Form der Inseln ändern sich, die Nachbarschaften bleiben. |
| `umap_param_grid` | UMAP-Grid n_neighbors × min_dist: n_neighbors steuert lokal/global, min_dist die Kompaktheit. |
| `umap_init_comparison` | UMAP spectral vs. random (Kobak & Linderman 2021). |
| `umap_transform` | UMAP transform(): 20 % neue Briefe werden ohne Neuberechnung eingebettet. |
| `autoencoder_loss` | Lernkurven (Training/Validierung) mit Early Stopping. |
| `autoencoder_linear_vs_pca` | PCA vs. linearer AE vs. ReLU-AE. Der lineare AE trifft PC2, die andere Richtung ist wegen λ₂ ≈ λ₃ unbestimmt. |
| `method_comparison_language` | **Zentrale Folie:** PCA, t-SNE, UMAP, PaCMAP, Autoencoder nebeneinander, gefärbt nach Sprache. |
| `method_comparison_request` | Dieselben Projektionen, gefärbt nach Anliegen-Kategorie. |
| `projection_pca_italian`, `tsne_perplexity_grid_italian`, `umap_param_grid_italian` | Variante nur italienische Briefe. |
| `method_comparison_italian` | Fünf Verfahren nur auf italienischen Briefen, gefärbt nach Anliegen und Archiv. |
| `quality_metrics` | Trustworthiness, kNN-Erhalt, Spearman ρ und Laufzeit je Verfahren (alle Sprachen / Italienisch). |
| `umap_regions` | Beste UMAP-Projektion mit manuell gewählten Regionen A–… (Inhalte: `results/dimred/umap_regions.csv`). |
| `outliers_in_projection` | Outlier aus dem Outlier-Notebook in der UMAP-Projektion und ihre lokale Dichte. |

## Tabellen (`results/dimred/`)

- `dimred_quality.csv` – Qualitätsmetriken + Laufzeit je Verfahren und Datensatz
- `umap_grid_quality.csv` – Qualitätsmetriken aller UMAP-Einstellungen
- `features_pca_loadings.csv`, `features_pca_n_components.csv` – Loadings und Komponentenwahl der Feature-PCA
- `svd_top_terms.csv`, `umap_regions.csv`
- `dimred_stats.json` – alle Kennzahlen
- `projections/*.npy` – alle 2D-Projektionen (Reihenfolge der Briefe = Reihenfolge nach Ausschluss; IDs in `projections/ids_*.npy`)
