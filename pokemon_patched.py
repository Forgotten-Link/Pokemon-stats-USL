#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project 2 — Pokémon Role Clustering (FULL PIPELINE) — PATCHED
- EDA (data quality, histograms, correlation heatmap)
- KMeans sweep k=3..10 (elbow + silhouette), pick best by silhouette
- Fit best k, save labeled CSV + profiles + PCA plot
- Comparisons: GMM, Agglomerative (Ward), DBSCAN, K-Medoids (if installed)
- Optional dendrogram (if scipy installed)
This version fixes the aggregation error by NOT averaging string columns (Archetype).
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Optional libs
try:
    from sklearn_extra.cluster import KMedoids
    HAVE_SKLEARN_EXTRA = True
except Exception:
    HAVE_SKLEARN_EXTRA = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Config ----------------
INPUT_CSV = "PokemonData.csv"  # hardcoded for convenience
NAME_COL = "Name"
TOTAL_COL = "Stat Total"

RANDOM_STATE = 42
K_RANGE = list(range(3, 11))
OUTDIR = "./outputs"
os.makedirs(OUTDIR, exist_ok=True)

SP_ATK_ALIASES = ["Sp.Attack", "Sp Atk", "Sp. Atk", "SpAtk", "Sp_Attack", "Sp Attack"]
SP_DEF_ALIASES = ["Sp.Defense", "Sp Def", "Sp. Def", "SpDef", "Sp_Defense", "Sp Defense"]

# ---------------- Helpers ----------------
def find_first(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidates present: {candidates}")

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str, str]:
    df = df.copy()
    sp_atk = find_first(df, SP_ATK_ALIASES)
    sp_def = find_first(df, SP_DEF_ALIASES)
    if TOTAL_COL not in df.columns:
        raise KeyError(f"Total column '{TOTAL_COL}' not found in CSV.")
    # Derived
    df["Attack_Bias"] = df["Attack"] - df[sp_atk]
    df["Defense_Bias"] = df["Defense"] - df[sp_def]
    features = ["HP","Attack","Defense",sp_atk,sp_def,"Speed",TOTAL_COL,"Attack_Bias","Defense_Bias"]
    return df, features, sp_atk, sp_def

def internal_metrics(X, labels) -> Dict[str, float]:
    uniq = np.unique(labels)
    out = {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    if len(uniq) < 2:
        return out
    try: out["silhouette"] = float(silhouette_score(X, labels))
    except Exception: pass
    try: out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception: pass
    try: out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception: pass
    return out

def label_archetypes(means: pd.DataFrame, sp_atk: str, sp_def: str) -> Dict[int, str]:
    names = {}
    for cl, row in means.iterrows():
        atk = row["Attack"]; satk = row[sp_atk]; spe = row["Speed"]
        de = row["Defense"]; sde = row[sp_def]; hp = row["HP"]
        if spe > 100 and max(atk, satk) > 110:
            names[cl] = "Fast Sweepers"
        elif hp > 100 and min(de, sde) > 100:
            names[cl] = "Bulky Tanks"
        elif atk - satk > 20 and atk > 110:
            names[cl] = "Physical Bruisers"
        elif satk - atk > 20 and satk > 110:
            names[cl] = "Special Cannons"
        elif abs(atk - satk) < 15 and abs(de - sde) < 15:
            names[cl] = "Balanced/Mixed"
        else:
            names[cl] = "Hybrids/Utility"
    return names

def save_elbow_silhouette(X, k_range, path_png, metrics_csv):
    inertias, sils = [], []
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inertias.append(km.inertia_)
        met = internal_metrics(X, km.labels_)
        sils.append(met["silhouette"])
        rows.append({"method":"kmeans","k":k, **met, "inertia": km.inertia_})
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].plot(k_range, inertias, marker="o")
    ax[0].set_title("Elbow (Inertia vs k)"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
    ax[1].plot(k_range, sils, marker="o")
    ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette")
    plt.tight_layout(); plt.savefig(path_png, dpi=160); plt.close()

def pca_scatter(X, labels, title, path_png):
    p = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = p.fit_transform(X)
    plt.figure(figsize=(7,6))
    plt.scatter(Z[:,0], Z[:,1], c=labels, s=20)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, linestyle=":")
    plt.tight_layout(); plt.savefig(path_png, dpi=160); plt.close()

def run_eda(df: pd.DataFrame, outdir: str, sp_atk: str, sp_def: str):
    # Data quality
    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes.values],
        "non_null": df.notnull().sum().values,
        "nulls": df.isnull().sum().values,
        "unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    quality.to_csv(os.path.join(outdir, "eda_data_quality.csv"), index=False)

    with open(os.path.join(outdir, "eda_duplicates.txt"), "w", encoding="utf-8") as f:
        f.write(f"Duplicate rows (exact): {int(df.duplicated().sum())}\n")

    # Histograms for key numerical columns
    key_cols = ["HP","Attack","Defense",sp_atk,sp_def,"Speed",TOTAL_COL]
    if "Attack_Bias" in df.columns: key_cols.append("Attack_Bias")
    if "Defense_Bias" in df.columns: key_cols.append("Defense_Bias")
    key_cols = [c for c in key_cols if c in df.columns]

    for col in key_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        plt.figure(figsize=(6,4))
        plt.hist(df[col].dropna().values, bins=30)
        plt.title(f"Histogram — {col}")
        plt.xlabel(col); plt.ylabel("Frequency"); plt.grid(True, linestyle=":")
        p = os.path.join(outdir, f"hist_{col.replace(' ', '_').replace('.', '')}.png")
        plt.tight_layout(); plt.savefig(p, dpi=140); plt.close()

    # Correlation heatmap
    use_cols = [c for c in key_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(use_cols) >= 2:
        corr = df[use_cols].corr(numeric_only=True)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        im = ax.imshow(corr.values, interpolation='nearest')
        ax.set_xticks(range(len(use_cols))); ax.set_yticks(range(len(use_cols)))
        ax.set_xticklabels(use_cols, rotation=90); ax.set_yticklabels(use_cols)
        fig.colorbar(im); plt.title("Correlation Heatmap (key stats)")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "correlation_heatmap.png"), dpi=150); plt.close()

def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if NAME_COL not in df.columns:
        df[NAME_COL] = [f"Pokemon_{i}" for i in range(len(df))]

    # Feature engineering
    df2, feat_cols, sp_atk, sp_def = engineer_features(df)

    # EDA
    run_eda(df2, OUTDIR, sp_atk, sp_def)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df2[feat_cols].values)

    # KMeans sweep + plots + metrics
    elbow_sil_png = os.path.join(OUTDIR, "kmeans_elbow_silhouette.png")
    kmeans_metrics_csv = os.path.join(OUTDIR, "kmeans_metrics.csv")
    save_elbow_silhouette(X, K_RANGE, elbow_sil_png, kmeans_metrics_csv)
    mets = pd.read_csv(kmeans_metrics_csv)
    best_row = mets.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    print(f"Best k by silhouette: {best_k} (score={best_row['silhouette']:.3f})")

    # Fit best and label
    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X)
    labels = km_best.labels_
    df_labeled = df.copy()
    df_labeled["Cluster_KMeans"] = labels

    means = df_labeled.groupby("Cluster_KMeans")[["HP","Attack","Defense",sp_atk,sp_def,"Speed"]].mean()
    arch = label_archetypes(means, sp_atk, sp_def)
    df_labeled["Archetype"] = df_labeled["Cluster_KMeans"].map(arch)
    df_labeled.to_csv(os.path.join(OUTDIR, f"pokemon_with_kmeans_k{best_k}.csv"), index=False)

    # --------- PATCHED profiling (no mean/std on strings) ---------
    numeric_cols = ["HP","Attack","Defense",sp_atk,sp_def,"Speed",TOTAL_COL]
    prof_numeric = (
        df_labeled
        .groupby("Cluster_KMeans")[numeric_cols]
        .agg(["mean","std"])
        .round(2)
    )
    # Mode (most frequent) archetype per cluster
    arch_mode = (
        df_labeled
        .groupby("Cluster_KMeans")["Archetype"]
        .agg(lambda s: s.value_counts().index[0])
        .rename(("Archetype","mode"))
    )
    prof = prof_numeric.copy()
    prof[("Archetype","mode")] = arch_mode
    prof.to_csv(os.path.join(OUTDIR, f"kmeans_profiles_k{best_k}.csv"))

    pca_scatter(X, labels, f"KMeans (k={best_k}) — PCA", os.path.join(OUTDIR, f"kmeans_pca_k{best_k}.png"))

    # Comparisons
    comp_rows = []
    # GMM
    for k in K_RANGE:
        g = GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit(X)
        lab = g.predict(X)
        m = internal_metrics(X, lab)
        comp_rows.append({"method":"gmm","k":k, **m})
        if k == best_k:
            pca_scatter(X, lab, f"GMM (k={k}) — PCA", os.path.join(OUTDIR, f"gmm_pca_k{k}.png"))
    # Agglomerative
    for k in K_RANGE:
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        lab = ag.labels_
        m = internal_metrics(X, lab)
        comp_rows.append({"method":"hierarchical_ward","k":k, **m})
        if k == best_k:
            pca_scatter(X, lab, f"Agglo Ward (k={k}) — PCA", os.path.join(OUTDIR, f"hierarchical_ward_pca_k{k}.png"))
    # DBSCAN coarse grid
    for eps in [0.8, 1.0, 1.2, 1.5]:
        for ms in [5, 10, 20]:
            db = DBSCAN(eps=eps, min_samples=ms).fit(X)
            lab = db.labels_
            m = internal_metrics(X, lab)
            nclust = len(set(lab)) - (1 if -1 in lab else 0)
            comp_rows.append({"method":"dbscan","k":nclust,"eps":eps,"min_samples":ms, **m})
    # K-Medoids (optional)
    if HAVE_SKLEARN_EXTRA:
        for k in K_RANGE:
            kmdo = KMedoids(n_clusters=k, random_state=RANDOM_STATE).fit(X)
            lab = kmdo.labels_
            m = internal_metrics(X, lab)
            comp_rows.append({"method":"kmedoids","k":k, **m})
            if k == best_k:
                pca_scatter(X, lab, f"KMedoids (k={k}) — PCA", os.path.join(OUTDIR, f"kmedoids_pca_k{k}.png"))

    comps = pd.DataFrame(comp_rows)
    comps.to_csv(os.path.join(OUTDIR, "comparisons_metrics.csv"), index=False)

    # Optional dendrogram
    if HAVE_SCIPY:
        Z = linkage(X, method="ward")
        plt.figure(figsize=(9,4))
        dendrogram(Z, no_labels=True)
        plt.title("Hierarchical Dendrogram (Ward)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "hierarchical_dendrogram.png"), dpi=160)
        plt.close()

    print("All done. See ./outputs for EDA, metrics, plots, and labeled CSVs.")

if __name__ == "__main__":
    main()
