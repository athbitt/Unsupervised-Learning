"""Hilfsfunktionen für die Outlier Detection (Notebook 02_outlier_detection.ipynb).

Alle Funktionen arbeiten ausschließlich auf numerischen, handgebauten Features
(keine Embeddings, kein TF-IDF, keine Dimensionsreduktion).
"""
from __future__ import annotations

import ast
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull, QhullError

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Einheitlicher Plot-Stil
# --------------------------------------------------------------------------
PALETTE = {
    "main": "#2E5A87",      # Standardfarbe (unauffällige Briefe, Balken)
    "outlier": "#D1495B",   # Ausreißer / Grenzwerte
    "accent": "#EDAE49",    # zweite Hervorhebung (z. B. Faktor 3.0)
    "teal": "#00798C",      # dritte Farbe (z. B. robuste Schätzung)
    "grey": "#8C96A0",      # Hilfslinien, Hintergrundpunkte
}
SEQ_CMAP = "Blues"          # sequentielle Farbskala (Heatmaps)
DIV_CMAP = "RdBu_r"         # divergierende Skala (Korrelationen)


def set_style() -> None:
    """Setzt einheitliche Matplotlib-Defaults (Schrift >= 12)."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": plt.cycler(color=[PALETTE[k] for k in
                                             ["main", "outlier", "accent", "teal", "grey"]]),
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
    })


def save_fig(fig, name: str, figdir: Path) -> None:
    """Speichert eine Abbildung als PNG (300 dpi) und SVG."""
    import time
    figdir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        for attempt in range(5):  # Windows: Datei kurz von Vorschau/Virenscanner gesperrt
            try:
                fig.savefig(figdir / f"{name}.{ext}", **kw)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(1.0)


# --------------------------------------------------------------------------
# Laden & Parsen
# --------------------------------------------------------------------------
LIST_COLS = ["person_ids", "place_ids", "organisation_ids", "languages",
             "attachment_ids", "cases", "doc_style"]


def parse_list(value) -> list:
    """Parst Listen-Strings wie "['P1', 'P2']"; '[]' oder NaN -> []."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if s in ("", "[]"):
        return []
    try:
        out = ast.literal_eval(s)
        return list(out) if isinstance(out, (list, tuple)) else [out]
    except (ValueError, SyntaxError):
        return []


def load_data(zip_path: Path, extract_dir: Path) -> pd.DataFrame:
    """Entpackt das Zip, liest die drei CSVs und joint sie 1:1 über 'id'."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    raw = pd.read_csv(extract_dir / "documents_rawdata.csv", encoding="utf-8")
    enr = pd.read_csv(extract_dir / "documents_enriched.csv", encoding="utf-8")
    summ = pd.read_csv(extract_dir / "documents_summaries.csv", encoding="utf-8")
    for d, name in [(raw, "rawdata"), (enr, "enriched"), (summ, "summaries")]:
        assert d["id"].is_unique, f"id in {name} nicht eindeutig"
    df = (raw.merge(enr, on="id", how="inner", validate="1:1")
             .merge(summ, on="id", how="inner", validate="1:1"))
    for c in LIST_COLS:
        df[c] = df[c].map(parse_list)
    return df


YEAR_RE = re.compile(r"(19[3-4]\d)")


def extract_year(value) -> float:
    """Erstes Jahr 1930–1949 aus dem Datumsstring, sonst NaN."""
    if not isinstance(value, str):
        return np.nan
    m = YEAR_RE.search(value)
    return float(m.group(1)) if m else np.nan


# langdetect liefert ISO-Codes; main_language ist deutsch beschriftet.
LANG_CODE_TO_DE = {
    "it": "Italienisch", "de": "Deutsch", "fr": "Französisch", "es": "Spanisch",
    "en": "Englisch", "hu": "Ungarisch", "pt": "Portugiesisch", "hr": "Kroatisch",
    "he": "Hebräisch", "pl": "Polnisch", "nl": "Niederländisch", "ro": "Rumänisch",
    "cs": "Tschechisch", "sk": "Slowakisch", "sl": "Slowenisch", "ca": "Katalanisch",
}
# Sprachen in main_language, die langdetect gar nicht kennt -> kein Mismatch-Vergleich
LANGDETECT_UNSUPPORTED = {"Latein", "Jiddisch"}


def detect_language(text: str, max_chars: int = 3000) -> str | None:
    """Deterministische Spracherkennung (langdetect, Seed fest) -> deutscher Name."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        code = detect(text[:max_chars])
    except LangDetectException:
        return None
    return LANG_CODE_TO_DE.get(code, code)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def text_features(text: str) -> dict:
    text = text if isinstance(text, str) else ""
    words = text.split()
    n_chars = len(text)
    letters = sum(ch.isalpha() for ch in text)
    non_alpha = sum((not ch.isalpha()) and (not ch.isspace()) for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    upper = sum(ch.isupper() for ch in text)
    return {
        "n_words": len(words),
        "n_chars": n_chars,
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "share_non_alpha": non_alpha / n_chars if n_chars else 0.0,
        "share_digits": digits / n_chars if n_chars else 0.0,
        # Anteil Großbuchstaben an allen Buchstaben
        "share_upper": upper / letters if letters else 0.0,
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    tf = pd.DataFrame([text_features(t) for t in df["raw_text"]], index=df.index)
    feat = pd.DataFrame({"id": df["id"]})
    feat = pd.concat([feat, tf], axis=1)
    feat["n_persons"] = df["person_ids"].map(len)
    feat["n_places"] = df["place_ids"].map(len)
    feat["n_organisations"] = df["organisation_ids"].map(len)
    feat["n_languages"] = df["languages"].map(len)
    feat["n_attachments"] = df["attachment_ids"].map(len)
    feat["n_cases"] = df["cases"].map(len)
    feat["year"] = df["year"]
    both = (df["main_language"].notna() & df["lang_detected"].notna()
            & ~df["main_language"].isin(LANGDETECT_UNSUPPORTED))
    feat["lang_mismatch"] = both & (df["main_language"] != df["lang_detected"])
    return feat


# --------------------------------------------------------------------------
# Univariate Methoden
# --------------------------------------------------------------------------
def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_flags(feat: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """Bool-Matrix: True, wenn Wert außerhalb [Q1 - k*IQR, Q3 + k*IQR]."""
    out = {}
    for c in cols:
        s = feat[c]
        lo, hi = iqr_bounds(s.dropna(), k)
        out[c] = ((s < lo) | (s > hi)).fillna(False)
    return pd.DataFrame(out, index=feat.index)


def skew_table(feat: pd.DataFrame, cols: list[str]) -> pd.Series:
    return feat[cols].apply(lambda s: stats.skew(s.dropna()))


# --------------------------------------------------------------------------
# Multivariate Distanzen
# --------------------------------------------------------------------------
def mahalanobis_sq(X: np.ndarray, center: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """d² = (x - c)^T Σ^{-1} (x - c) für jede Zeile von X."""
    diff = X - center
    inv = np.linalg.inv(cov)
    return np.einsum("ij,jk,ik->i", diff, inv, diff)


def chi2_quantiles(n: int, p: int) -> np.ndarray:
    """χ²_p-Quantile bei (j - 0.5)/n, j = 1..n."""
    probs = (np.arange(1, n + 1) - 0.5) / n
    return stats.chi2.ppf(probs, df=p)


def chi2_plot(ax, d2: np.ndarray, p: int, title: str, q: float = 0.995,
              color: str | None = None, label_top: pd.Series | None = None):
    """χ²-QQ-Plot: geordnete d² gegen χ²_p-Quantile, Steigung-1-Linie, Grenze."""
    d2_sorted = np.sort(d2)
    theo = chi2_quantiles(len(d2), p)
    cutoff = stats.chi2.ppf(q, df=p)
    is_out = d2_sorted > cutoff
    ax.scatter(theo[~is_out], d2_sorted[~is_out], s=10, color=color or PALETTE["main"],
               alpha=0.6, label="unauffällig")
    ax.scatter(theo[is_out], d2_sorted[is_out], s=14, color=PALETTE["outlier"],
               label=f"> {q*100:.1f}-%-Quantil (n={is_out.sum()})")
    lim = max(theo.max(), 1)
    ax.plot([0, lim], [0, lim], color="black", lw=1.2, label="Steigung 1")
    ax.axhline(cutoff, color=PALETTE["outlier"], ls="--", lw=1.2)
    ax.set_xlabel(f"$\\chi^2_{{{p}}}$-Quantil")
    ax.set_ylabel("geordnete $d^2$")
    ax.set_title(title)
    ax.set_yscale("symlog", linthresh=50)
    ax.legend(loc="upper left", fontsize=12)
    return cutoff


# --------------------------------------------------------------------------
# Depth-based: Convex Hull Peeling
# --------------------------------------------------------------------------
def convex_hull_peeling(X: np.ndarray) -> np.ndarray:
    """Iteratives Schälen konvexer Hüllen (2D). Rückgabe: Layer-Nummer je Punkt (1 = außen).

    Läuft, bis < 3 Punkte übrig sind oder die Restpunkte degeneriert (kollinear)
    sind; diese erhalten den letzten Layer + 1.
    """
    n = len(X)
    layers = np.zeros(n, dtype=int)
    remaining = np.arange(n)
    layer = 1
    while len(remaining) >= 3:
        try:
            hull = ConvexHull(X[remaining])
        except QhullError:
            break  # degeneriert (z. B. alle Restpunkte kollinear)
        on_hull = remaining[hull.vertices]
        # Duplikate eines Eckpunkts gehören zur selben Schicht
        vert_coords = {tuple(v) for v in X[on_hull]}
        dup = np.array([tuple(X[i]) in vert_coords for i in remaining])
        layers[remaining[dup]] = layer
        remaining = remaining[~dup]
        layer += 1
    layers[remaining] = layer
    return layers


# --------------------------------------------------------------------------
# Vergleich
# --------------------------------------------------------------------------
def jaccard_matrix(flags: pd.DataFrame) -> pd.DataFrame:
    cols = flags.columns
    m = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            inter = (flags[a] & flags[b]).sum()
            union = (flags[a] | flags[b]).sum()
            m.loc[a, b] = inter / union if union else np.nan
    return m
