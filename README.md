# Habemus Data – Unsupervised Learning (WS 2026/27)

Projekt zu 3.839 transkribierten Bittbriefen an Papst Pius XII. aus den Vatikanischen Archiven (1938–1945).
Stand: Zwischenpräsentation (Outlier Detection, Dimensionality Reduction). Clustering folgt in Teil 2.

## Setup

```bash
python -m venv ul
ul\Scripts\activate            # Windows
pip install -r requirements.txt
python -m spacy download it_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
```

Die Daten liegen als `data/Habemus_Data_-20261108.zip` im Repo. Die Notebooks entpacken sie automatisch nach `data/raw/`.
Die Spaltenbeschreibung steht in [data/README.md](data/README.md).

**Notebooks in der Reihenfolge 02 → 03 ausführen.** Notebook 03 liest die Ergebnisse aus `results/outliers/`.

## Struktur

```
data/
  Habemus_Data_-20261108.zip       Rohdaten (3 CSVs: rawdata, enriched, summaries)
  README.md                        Spaltenbeschreibung
notebooks/
  02_outlier_detection.ipynb       IQR, QQ-Plots, z-Score, Mahalanobis (+ MCD), Hull Peeling, ABOD, Isolation Forest, LOF
  03_dimensionality_reduction.ipynb Curse of Dimensionality, PCA, TF-IDF/SVD, t-SNE, UMAP, PaCMAP, Autoencoder, Qualitätsmaße
src/
  outliers.py                      Laden, Features, Outlier-Hilfsfunktionen, Plot-Stil
  dimred.py                        Hilfsfunktionen für die Dimensionsreduktion
figures/outliers/, figures/dimred/ alle Abbildungen (PNG + SVG)
results/outliers/                  Tabellen + Kennzahlen, Übersicht: OUTLIER_README.md
results/dimred/                    Projektionen, Kennzahlen, Übersicht: DIMRED_README.md
```

Welche Abbildung was zeigt, steht in [results/outliers/OUTLIER_README.md](results/outliers/OUTLIER_README.md)
und [results/dimred/DIMRED_README.md](results/dimred/DIMRED_README.md).
