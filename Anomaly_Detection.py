"""
Anomaly_Detection.py
Unsupervised multivariate anomaly detection using:
PCA, k-Means, Hierarchical Clustering, DBSCAN, GMM (EM), One-Class SVM.
Final anomaly decision is based on a combined anomaly score and a robust MAD threshold.
"""

# ------------------------------
# CONFIGURATION
# ------------------------------
CSV_PATH = "CC GENERAL.csv"
OUTPUT_CSV = "Anomaly_Percentage.csv"
PLOT_IMAGE = "Anomaly_Scatter.png"

RANDOM_STATE = 42
K = 3
DB_NN = 5
DB_EPS_PERCENTILE = 90
OCSVM_NU = 0.01
PCA_VARIANCE_KEEP = 0.95
MAD_THRESHOLD = 3.5

SAVE_PLOT = True
PRINT_SUMMARY = True

# ------------------------------
# IMPORTS
# ------------------------------
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import time

# ------------------------------
# HELPERS
# ------------------------------
def find_custid_column(df):
    for c in df.columns:
        if c.lower() in ("cust_id", "customerid", "custid", "customer_id"):
            return c
    for c in df.columns:
        if c.lower().startswith("cust"):
            return c
    return None

def preprocess_numeric(df, cust_col):
    drop_cols = [c for c in df.columns if c != cust_col and "id" in c.lower()]
    df_num = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])

    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df_num),
                         columns=df_num.columns)

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imp),
                            columns=df_num.columns)

    return X_scaled, X_imp

def minmax(x):
    x = np.array(x, float)
    return np.zeros_like(x) if x.max() == x.min() else (x - x.min()) / (x.max() - x.min())

def robust_z_score(values):
    values = np.array(values)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return np.zeros_like(values) if mad == 0 else 0.6745 * (values - med) / mad

# ------------------------------
# MODEL SCORERS
# ------------------------------
def pca_recon_error(X):
    p = PCA(n_components=PCA_VARIANCE_KEEP, svd_solver='full', random_state=RANDOM_STATE)
    proj = p.fit_transform(X)
    rec = p.inverse_transform(proj)
    return ((X.values - rec) ** 2).mean(axis=1)

def kmeans_score(X):
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE).fit(X)
    _, dists = pairwise_distances_argmin_min(X.values, km.cluster_centers_)
    return dists

def hierarchical_score(X):
    h = AgglomerativeClustering(n_clusters=K).fit(X)
    labels = h.labels_
    centroids = np.vstack([X.values[labels == lab].mean(axis=0) for lab in np.unique(labels)])
    return np.array([np.linalg.norm(x - centroids[l]) for x, l in zip(X.values, labels)])

def dbscan_score(X):
    nn = NearestNeighbors(n_neighbors=DB_NN).fit(X)
    d, _ = nn.kneighbors(X)
    kdist = d[:, -1]
    eps = np.percentile(kdist, DB_EPS_PERCENTILE)
    db = DBSCAN(eps=eps, min_samples=DB_NN).fit(X)
    return kdist, (db.labels_ == -1).astype(int), db.labels_, eps

def gmm_score(X):
    g = GaussianMixture(n_components=K, random_state=RANDOM_STATE).fit(X)
    return -g.score_samples(X)

def ocsvm_score(X):
    oc = OneClassSVM(nu=OCSVM_NU, gamma="scale").fit(X)
    score = -oc.score_samples(X)
    return score, (oc.predict(X) == -1).astype(int)

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def run_pipeline():
    start_time = time.time()

    df = pd.read_csv(CSV_PATH)
    cust_col = find_custid_column(df)
    X_scaled, _ = preprocess_numeric(df, cust_col)

    pca_err = pca_recon_error(X_scaled)
    km = kmeans_score(X_scaled)
    hr = hierarchical_score(X_scaled)
    db_kd, db_flag, db_labels, db_eps = dbscan_score(X_scaled)
    gmm = gmm_score(X_scaled)
    oc, oc_flag = ocsvm_score(X_scaled)

    scores = pd.DataFrame({
        "pca": pca_err, "kmeans": km, "hier": hr,
        "db_kdist": db_kd, "gmm": gmm, "ocsvm": oc,
        "db_flag": db_flag, "oc_flag": oc_flag
    })

    for col in ["pca", "kmeans", "hier", "db_kdist", "gmm", "ocsvm"]:
        scores[col + "_n"] = minmax(scores[col])

    scores["anomaly_percent"] = (
        scores[["pca_n", "kmeans_n", "hier_n", "db_kdist_n", "gmm_n", "ocsvm_n"]]
        .mean(axis=1) * 100
    )

    combined_z = robust_z_score(scores["anomaly_percent"])
    scores["is_anomaly"] = (combined_z > MAD_THRESHOLD).astype(int)

    z_pca = robust_z_score(scores["pca"])
    z_km = robust_z_score(scores["kmeans"])
    z_hr = robust_z_score(scores["hier"])
    z_db = robust_z_score(scores["db_kdist"])
    z_gmm = robust_z_score(scores["gmm"])
    z_oc = robust_z_score(scores["ocsvm"])

    out = pd.DataFrame({
        "CUST_ID": df[cust_col] if cust_col else df.index,
        "anomaly_percent": scores["anomaly_percent"],
        "is_anomaly": scores["is_anomaly"]
    })
    out.to_csv(OUTPUT_CSV, index=False)

    if SAVE_PLOT:
        pca_vis = PCA(n_components=2).fit(X_scaled)
        vis = pca_vis.transform(X_scaled)
        var1, var2 = pca_vis.explained_variance_ratio_ * 100

        plt.figure(figsize=(9,7))
        sc = plt.scatter(vis[:,0], vis[:,1], c=scores["anomaly_percent"], cmap="viridis", s=25)

        an_idx = scores.index[scores["is_anomaly"] == 1]
        plt.scatter(vis[an_idx,0], vis[an_idx,1],
                    facecolors='none', edgecolors='k', s=80, label="Anomalies")

        plt.xlabel(f"PCA Component 1 ({var1:.2f}% variance)")
        plt.ylabel(f"PCA Component 2 ({var2:.2f}% variance)")
        plt.title("2D PCA Projection – Anomaly Detection")
        plt.colorbar(sc).set_label("Anomaly % Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_IMAGE, dpi=200)

    end_time = time.time()
    total_time = end_time - start_time

    if PRINT_SUMMARY:
        print("ANOMALY DETECTION RESULT SUMMARY")
        print("--------------------------------")
        print(f"Input: {CSV_PATH}")
        print(f"Total samples: {len(scores)}\n")
        print(f"Detected anomalies (MAD > {MAD_THRESHOLD}): {scores['is_anomaly'].sum()}\n")

        print("Counts flagged per model:")
        print(f" - PCA reconstruction error: { (z_pca > MAD_THRESHOLD).sum() }")
        print(f" - k-Means distance: { (z_km > MAD_THRESHOLD).sum() }")
        print(f" - Hierarchical distance: { (z_hr > MAD_THRESHOLD).sum() }")
        print(f" - DBSCAN k-distance: { (z_db > MAD_THRESHOLD).sum() }")
        print(f" - GMM neg-log-likelihood: { (z_gmm > MAD_THRESHOLD).sum() }")
        print(f" - One-Class SVM score: { (z_oc > MAD_THRESHOLD).sum() }")
        print(f" - DBSCAN explicit labels: {scores['db_flag'].sum()}")
        print(f" - One-Class SVM explicit labels: {scores['oc_flag'].sum()}\n")

        print(f"Total execution time: {total_time:.2f} seconds\n")
        print(f"CSV written to: {OUTPUT_CSV}")
        print(f"Plot saved to: {PLOT_IMAGE}")

    return out


if __name__ == "__main__":
    run_pipeline()