"""Hilfsfunktionen für die Dimensionsreduktion (Notebook 03_dimensionality_reduction.ipynb).

Stil (Farben, Schrift, Speichern) wird aus src/outliers.py übernommen, damit alle
Abbildungen des Projekts einheitlich aussehen.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .outliers import PALETTE, RANDOM_STATE, save_fig, set_style  # noqa: F401

# Kategoriale Palette (max. 8 Klassen + Grau für "Andere/unbekannt")
CAT_COLORS = ["#2E5A87", "#D1495B", "#EDAE49", "#00798C", "#66A182",
              "#8E6C8A", "#F08A4B", "#5B5F97"]
OTHER_COLOR = "#B8BEC4"
OTHER_LABELS = {"Andere", "unbekannt", "Sonstiges", "(leer)", "nein"}


# --------------------------------------------------------------------------
# Färbe-Variablen
# --------------------------------------------------------------------------
def year_group(y) -> str:
    if pd.isna(y):
        return "unbekannt"
    y = int(y)
    if y <= 1939:
        return "≤1939"
    if y >= 1943:
        return "≥1943"
    return str(y)


YEAR_ORDER = ["≤1939", "1940", "1941", "1942", "≥1943", "unbekannt"]

REQUEST_RULES = [
    ("Freilassung/Internierung", r"releas|liberat|intern|camp|prison|detain|detention|arrest|deport|"
                                 r"confinement|captiv|jail|freedom|set free|pardon"),
    ("Geld/Unterstützung", r"money|financ|fund|subsid|alms|donation|loan|food|cloth|material|payment|"
                           r"contribution|monetary|charit|pension|lira|lire|dollar|debt|job|employ|work|position"),
    ("Visum/Emigration", r"visa|emigra|immigra|entry|passport|permit|travel|passage|transit|ship|"
                         r"residen|citizenship|affidavit|to leave|leave the country|admission|ticket"),
    ("Auskunft/Suche", r"inform|news|whereabout|search|locat|find|trace|contact|inquir|enquir|fate|"
                       r"address|forward|transmit|message|tidings|word of|status of"),
    ("Fürsprache/Empfehlung", r"interce|interven|recommend|protect|support|favou?r|audience|influence|"
                              r"good offices|advocat|step|plead|consideration"),
]
REQUEST_ORDER = [r[0] for r in REQUEST_RULES] + ["Sonstiges"]


def request_category(text) -> str:
    """Bündelt what_is_requested per Keyword-Regel (Reihenfolge = Priorität)."""
    if not isinstance(text, str) or not text.strip():
        return "Sonstiges"
    t = text.lower()
    for name, pattern in REQUEST_RULES:
        if re.search(pattern, t):
            return name
    return "Sonstiges"


def archive_short(s) -> str:
    if not isinstance(s, str):
        return "unbekannt"
    m = re.search(r"\(([A-Z]+)\)\s*$", s.strip())
    return m.group(1) if m else s[:12]


def top_k(series: pd.Series, k: int, other: str = "Andere") -> pd.Series:
    keep = series.value_counts().index[:k]
    return series.where(series.isin(keep), other)


# --------------------------------------------------------------------------
# Plot einer 2D-Projektion, je Färbe-Variable ein Subplot
# --------------------------------------------------------------------------
def _ordered_levels(labels: pd.Series, order: list | None) -> list:
    present = list(pd.unique(labels))
    if order:
        lv = [o for o in order if o in present] + [p for p in present if p not in order]
    else:
        lv = list(labels.value_counts().index)
    # Restkategorien ans Ende
    return [l for l in lv if l not in OTHER_LABELS] + [l for l in lv if l in OTHER_LABELS]


def scatter_by(ax, Y: np.ndarray, labels: pd.Series, order: list | None = None,
               s: float = 5, alpha: float = 0.65, legend: bool = True, legend_loc="best",
               highlight: str | None = None):
    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    levels = _ordered_levels(labels, order)
    colors, ci = {}, 0
    for lv in levels:
        if lv in OTHER_LABELS:
            colors[lv] = OTHER_COLOR
        else:
            colors[lv] = CAT_COLORS[ci % len(CAT_COLORS)]
            ci += 1
    # "Andere" zuerst zeichnen (Hintergrund), Hervorhebung zuletzt
    draw = sorted(levels, key=lambda l: (l not in OTHER_LABELS, l == highlight))
    for lv in draw:
        m = (labels == lv).to_numpy()
        big = highlight is not None and lv == highlight
        ax.scatter(Y[m, 0], Y[m, 1], s=s * (4 if big else 1), alpha=1.0 if big else alpha,
                   color=colors[lv], label=f"{lv} ({m.sum()})", linewidths=0,
                   zorder=3 if big else 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if legend:
        leg = ax.legend(loc=legend_loc, fontsize=10, markerscale=2.2, frameon=True,
                        framealpha=0.85, handletextpad=0.2, borderpad=0.3)
        leg.set_zorder(5)
    return colors


def plot_projection(Y: np.ndarray, color_frame: pd.DataFrame, title: str,
                    cols: list | None = None, orders: dict | None = None,
                    ncols: int = 3, size: float = 5.2, s: float = 5):
    """Eine 2D-Projektion, je Färbe-Variable ein Subplot."""
    cols = cols or list(color_frame.columns)
    orders = orders or {}
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        scatter_by(ax, Y, color_frame[c], order=orders.get(c), s=s,
                   highlight="ja" if c == "Outlier" else None)
        ax.set_title(c)
    for ax in axes[len(cols):]:
        ax.remove()
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Curse of Dimensionality
# --------------------------------------------------------------------------
def distance_contrast(X: np.ndarray, metric: str = "euclidean") -> float:
    from scipy.spatial.distance import pdist
    d = pdist(X, metric=metric)
    d = d[d > 0]
    return (d.max() - d.min()) / d.min()


# --------------------------------------------------------------------------
# Text-Pre-Processing
# --------------------------------------------------------------------------
SPACY_MODELS = {"Italienisch": "it_core_news_sm", "Deutsch": "de_core_news_sm",
                "Französisch": "fr_core_news_sm"}
BLANK_CODES = {"Englisch": "en", "Spanisch": "es", "Portugiesisch": "pt", "Polnisch": "pl",
               "Hebräisch": "he", "Rumänisch": "ro", "Ungarisch": "hu", "Latein": "la",
               "Katalanisch": "ca", "Kroatisch": "hr", "Jiddisch": "xx"}
TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def global_stopwords() -> set:
    """Vereinigung der Stopwords der fünf häufigsten Sprachen.

    Viele Briefe sind gemischtsprachig (z. B. deutscher Text mit italienischem Archivvermerk)
    oder falsch etikettiert; sprachspezifische Listen allein lassen dann Funktionswörter
    der anderen Sprache durch, die die TF-IDF-Struktur verfälschen.
    """
    import spacy
    stop = set()
    for code in ("it", "de", "fr", "en", "es"):
        stop |= {w.lower() for w in spacy.blank(code).Defaults.stop_words}
    return stop


def preprocess_texts(texts: pd.Series, langs: pd.Series, batch_size: int = 64) -> pd.Series:
    """Kleinschreibung, Zahlen/Satzzeichen entfernen, Stopwords, Lemmatisierung (IT/DE/FR).

    `langs` bestimmt, welches spaCy-Modell lemmatisiert. Stopwords: Vereinigung über
    IT/DE/FR/EN/ES (siehe global_stopwords). Übrige Sprachen: nur Tokenisierung + Stopwords.
    """
    import spacy
    out = pd.Series([[] for _ in range(len(texts))], index=texts.index, dtype=object)
    gstop = global_stopwords()
    for lang, model in SPACY_MODELS.items():
        idx = langs.index[langs == lang]
        if len(idx) == 0:
            continue
        nlp = spacy.load(model, disable=["parser", "ner"])
        nlp.max_length = 2_000_000
        stop = nlp.Defaults.stop_words | gstop
        docs = nlp.pipe((str(t) for t in texts.loc[idx]), batch_size=batch_size)
        for i, doc in zip(idx, docs):
            toks = []
            for tok in doc:
                if not tok.is_alpha or tok.is_stop:
                    continue
                lemma = tok.lemma_.lower()
                if len(lemma) < 3 or lemma in stop or not lemma.isalpha():
                    continue
                toks.append(lemma)
            out.at[i] = toks
    rest = langs.index[~langs.isin(SPACY_MODELS)]
    for lang in langs.loc[rest].unique():
        code = BLANK_CODES.get(lang, "xx")
        try:
            stop = spacy.blank(code).Defaults.stop_words | gstop
        except Exception:  # Sprache in spaCy nicht vorhanden
            stop = gstop
        for i in rest[langs.loc[rest] == lang]:
            toks = [t for t in TOKEN_RE.findall(str(texts.at[i]).lower()) if t not in stop]
            out.at[i] = toks
    return out


def identity(x):
    return x


# --------------------------------------------------------------------------
# Caching von Projektionen (Array + Laufzeit)
# --------------------------------------------------------------------------
def cached_projection(path: Path, fn, force: bool = False):
    """Lädt path (.npy) samt Laufzeit aus path.json oder berechnet fn() und speichert beides."""
    path = Path(path)
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists() and not force:
        return np.load(path), json.loads(meta.read_text())["runtime_s"]
    t0 = time.perf_counter()
    Y = np.asarray(fn())
    rt = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, Y)
    meta.write_text(json.dumps({"runtime_s": rt}))
    return Y, rt


# --------------------------------------------------------------------------
# Qualitätsmetriken
# --------------------------------------------------------------------------
def knn_sets(X, k: int, metric: str):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric,
                          algorithm="brute" if metric == "cosine" else "auto").fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    return idx[:, 1:]


def knn_preservation(knn_orig: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    knn_low = knn_sets(Y, k, "euclidean")
    overlap = [len(set(a) & set(b)) / k for a, b in zip(knn_orig, knn_low)]
    return float(np.mean(overlap))


def global_spearman(X, Y: np.ndarray, n_pairs: int = 2000, seed: int = RANDOM_STATE) -> float:
    from sklearn.metrics.pairwise import paired_cosine_distances
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_orig = paired_cosine_distances(X[i], X[j])
    d_low = np.linalg.norm(Y[i] - Y[j], axis=1)
    return float(stats.spearmanr(d_orig, d_low).statistic)


def quality_metrics(X, Y: np.ndarray, knn_orig: np.ndarray, k: int = 10) -> dict:
    from sklearn.manifold import trustworthiness
    return {
        "trustworthiness": float(trustworthiness(X, Y, n_neighbors=k, metric="cosine")),
        "knn_preservation": knn_preservation(knn_orig, Y, k),
        "spearman_global": global_spearman(X, Y),
    }


# --------------------------------------------------------------------------
# Autoencoder (PyTorch)
# --------------------------------------------------------------------------
def build_autoencoder(p: int, dims=(64, 16), latent: int = 2, linear: bool = False):
    import torch.nn as nn

    def block(a, b, act):
        return [nn.Linear(a, b)] + ([nn.ReLU()] if act else [])

    enc, dec = [], []
    sizes = [p, *dims]
    for a, b in zip(sizes[:-1], sizes[1:]):
        enc += block(a, b, not linear)
    enc += [nn.Linear(sizes[-1], latent)]                    # linearer Bottleneck
    rsizes = [latent, *dims[::-1]]
    for a, b in zip(rsizes[:-1], rsizes[1:]):
        dec += block(a, b, not linear)
    dec += [nn.Linear(rsizes[-1], p)]
    return nn.Sequential(*enc), nn.Sequential(*dec)


def train_autoencoder(X: np.ndarray, linear: bool = False, dims=(64, 16), max_epochs: int = 300,
                      patience: int = 15, lr: float = 1e-3, batch_size: int = 128,
                      val_frac: float = 0.1, seed: int = RANDOM_STATE):
    """MSE-Autoencoder mit Adam und Early Stopping auf 10 % Validierungsdaten."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    enc, dec = build_autoencoder(X.shape[1], dims=dims, linear=linear)
    model = nn.Sequential(enc, dec)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    hist = {"train": [], "val": []}
    best, best_state, wait = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(tr_idx)
        tot = 0.0
        for b in range(0, len(order), batch_size):
            xb = Xt[order[b:b + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xt[val_idx]), Xt[val_idx]).item()
        hist["train"].append(tot / len(tr_idx))
        hist["val"].append(vl)
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Z = enc(Xt).numpy()
    return Z, hist, enc, dec


def linear_decoder_matrix(dec) -> np.ndarray:
    """Produkt der Gewichtsmatrizen eines linearen Decoders (p × latent)."""
    import torch.nn as nn
    M = None
    for layer in dec:
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().numpy()
            M = W if M is None else W @ M
    return M
