#!/usr/bin/env python3
# fast_resp.py – speedy stand-alone demo of the RESP heuristic
# ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans                   #   needs scikit-learn

# ── GLOBALS ──────────────────────────────────────────────────
MU_WEIGHT    = 0.5        # distance vs. workload weight in site-selection
ES_CAPACITY  = 5_000      # minutes
BS_MAX_DELAY = 50_000.0   # proxy units (same as distance matrix)
K_INIT       = 10         # K-means restarts
L_NEAR       = 15         # # nearest ES candidates examined per BS
# ─────────────────────────────────────────────────────────────

def load_bs(xlsx_path: str) -> pd.DataFrame:
    """Read Shanghai data  ➜  one row per BS with (lat,lon,workload)."""
    df = pd.read_excel(xlsx_path)
    df["workload"] = (pd.to_datetime(df["end time"])
                      - pd.to_datetime(df["start time"])
                      ).dt.total_seconds() / 60.0
    df = (df.groupby(["latitude", "longitude"])["workload"]
            .sum().reset_index()
            .rename(columns={"latitude": "lat", "longitude": "lon"}))
    return df.reset_index(drop=True)

# ── fast “k-medoids”: k-means → snap each centroid to nearest BS ────────────
def kmedoids_fast(latlon_arr: np.ndarray, k: int, rng) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=K_INIT, random_state=rng).fit(latlon_arr)
    centres = km.cluster_centers_
    # pick the BS index with minimal Euclidean distance to each centroid
    idx = [np.argmin(((latlon_arr - c) ** 2).sum(1)) for c in centres]
    return np.unique(idx)[:k]                        # de-dup defensive

# ── greedy BS→ES assignment with L-nearest shortcut ────────────────────────
def dual_assign(bs: pd.DataFrame, es: pd.DataFrame) -> pd.DataFrame:
    dist = cdist(bs[["lat", "lon"]], es[["lat", "lon"]])
    nearest_L = np.argpartition(dist, L_NEAR, axis=1)[:, :L_NEAR]
    load = np.zeros(len(es))
    prim = -np.ones(len(bs), int)
    back = -np.ones(len(bs), int)

    for i, wl in enumerate(bs["workload"]):
        # primary
        for es_id in nearest_L[i][dist[i, nearest_L[i]].argsort()]:
            if load[es_id] + wl <= ES_CAPACITY:
                prim[i] = es_id
                load[es_id] += wl
                break
        # backup
        for es_id in nearest_L[i][dist[i, nearest_L[i]].argsort()]:
            if es_id != prim[i] and dist[i, es_id] <= BS_MAX_DELAY \
                               and load[es_id] + wl <= ES_CAPACITY:
                back[i] = es_id
                load[es_id] += wl
                break
        if back[i] == -1:                       # fall-back if no spare found
            back[i] = prim[i]
    return pd.DataFrame({"primary": prim, "backup": back})

# ── headline metrics ───────────────────────────────────────────────────────
def compute_metrics(bs, es, assign, fail_ratio=0.20):
    dist  = cdist(bs[["lat", "lon"]], es[["lat", "lon"]])
    delay = dist[np.arange(len(assign)), assign["primary"]].mean()
    wb    = bs.groupby(assign["primary"])["workload"].sum().std()

    rng = np.random.default_rng(0)
    failed = rng.choice(len(es), max(1, int(fail_ratio * len(es))), replace=False)
    alive  = (~((assign["primary"].isin(failed)) &
                (assign["backup"].isin(failed)))).mean()
    fsr = alive
    rc  = (assign["primary"] != assign["backup"]).mean()
    return delay, wb, fsr, rc

# ── driver ─────────────────────────────────────────────────────────────────
def run_pipeline(xlsx_path: str, k_es: int = 50, rng_seed: int = 42):
    rng = np.random.default_rng(rng_seed)
    print("➤ Loading base-station table …")
    bs = load_bs(xlsx_path)

    print("➤ Selecting ES sites via fast k-medoids …")
    medoid_idx = kmedoids_fast(bs[["lat", "lon"]].to_numpy(), k_es, rng)
    es_df      = bs.iloc[medoid_idx].reset_index(drop=True)

    print("➤ Assigning primary + backup ES …")
    assign = dual_assign(bs, es_df)

    print("➤ Computing metrics …")
    delay, wb, fsr, rc = compute_metrics(bs, es_df, assign)

    # ── quick plots for visual sanity-check ────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=140)

    # (a) map of BSs and ESs
    size = 15 * np.sqrt(bs["workload"] / bs["workload"].max())
    ax[0].scatter(bs["lon"], bs["lat"], s=size, c="tab:blue", alpha=.3, label="BS")
    ax[0].scatter(es_df["lon"], es_df["lat"], marker="X", s=60,
                  c="crimson", label="ES")
    ax[0].set_title(f"RESP-selected ES sites (K={k_es})")
    ax[0].axis("equal"); ax[0].set_xlabel("lon"); ax[0].set_ylabel("lat")
    ax[0].legend(loc="lower right")

    # (b) bar chart of metrics
    vals = [delay, wb, fsr, rc]
    bars = ax[1].bar(["Delay", "WB σ", "FSR", "RC"], vals,
                     color=["tab:gray"]*4)
    for b in bars:
        ax[1].text(b.get_x() + b.get_width()/2, b.get_height(),
                   f"{b.get_height():.2f}", ha="center", va="bottom")
    ax[1].set_ylim(0, max(vals)*1.10)
    ax[1].set_title("Headline metrics")

    plt.tight_layout()
    plt.show()

    # console summary ------------------------------------------------------
    print(f"\nAverage delay (proxy-units): {delay:.3f}")
    print(f"Work-load st.dev.:          {wb:.1f}")
    print(f"Fail-over success rate:     {fsr:.3f}")
    print(f"Redundancy coverage:        {rc:.3f}")

# ── entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline("/content/resp_data.xlsx", k_es=50)
