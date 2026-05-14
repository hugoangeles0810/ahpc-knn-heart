#!/usr/bin/env python3
"""Unified KNN runner: sklearn (brute/kd_tree/ball_tree) + manual brute.

OMP_NUM_THREADS is read from the environment (set by the bash caller before
invoking Python). The script does NOT modify BLAS/OMP env vars — that's the
caller's responsibility. The observed value is logged into the CSV.
"""
import os

# Read OMP threads from env BEFORE importing numpy/sklearn.
OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", "0") or "0")

import argparse
import csv
import time
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset", "dataset.csv",
)

SKLEARN_ALGOS = {"brute", "kd_tree", "ball_tree"}
IMPL_CHOICES = [f"sklearn-{a}" for a in SKLEARN_ALGOS] + ["manual-brute"]


# ---------- Data expansion helpers (copied from knn_heart_*_scale.py) ----------
def expand_samples(X, y, factor: int, jitter: float = 0.0, seed: int = 42):
    if factor <= 1:
        return X, y
    rng = np.random.RandomState(seed)
    X_big = np.repeat(X, factor, axis=0)
    y_big = np.repeat(y, factor, axis=0)
    if jitter > 0.0:
        X_big = X_big + rng.normal(0.0, jitter, size=X_big.shape)
    return X_big, y_big


def expand_features(X, feature_mult: int, mode: str = "repeat", seed: int = 42):
    if feature_mult <= 1:
        return X
    n, d = X.shape
    if mode == "repeat":
        return np.tile(X, (1, feature_mult))
    if mode == "mix":
        rng = np.random.RandomState(seed)
        W = rng.normal(0, 1, size=(d, (feature_mult - 1) * d))
        X_new = X @ W
        return np.concatenate([X, X_new], axis=1)
    raise ValueError("feat-mode must be 'repeat' or 'mix'")


def estimate_flops(n_train: int, n_test: int, d: int, k: int):
    flops_dist = n_test * n_train * (3.0 * d)
    flops_select = n_test * (5.0 * n_train)
    return flops_dist, flops_select, flops_dist + flops_select


def load_cleveland(path: str):
    df = pd.read_csv(path)
    y = df["target"].astype(np.int32).to_numpy()
    X = df.drop(columns=["target", "num"]).astype(np.float64).to_numpy()
    return X, y


# ---------- Manual brute KNN primitives (from knn_heart_joblib_scale.py) ----------
def euclidean_distance_batch(X_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((X_train - x) ** 2, axis=1))


def predict_one(x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k: int) -> int:
    dists = euclidean_distance_batch(X_train, x)
    k_indices = np.argpartition(dists, k)[:k]
    k_labels = y_train[k_indices]
    cnt = Counter(k_labels)
    return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# ---------- Runner classes ----------
class KNNRunner:
    impl_tag: str
    algorithm: str  # value stored in the 'algorithm' CSV column (compat)

    def fit_predict(self, X_train, y_train, X_test):
        """Return (fit_time_s, pred_time_s, y_pred)."""
        raise NotImplementedError


class SklearnRunner(KNNRunner):
    def __init__(self, k: int, algorithm: str, n_jobs: int, backend: str):
        if algorithm not in SKLEARN_ALGOS:
            raise ValueError(f"sklearn algorithm must be one of {SKLEARN_ALGOS}")
        self.impl_tag = f"sklearn-{algorithm}"
        self.algorithm = algorithm
        self.k = k
        self.n_jobs = n_jobs
        self.backend = backend

    def fit_predict(self, X_tr, y_tr, X_te):
        clf = KNeighborsClassifier(
            n_neighbors=self.k,
            algorithm=self.algorithm,
            metric="euclidean",
            n_jobs=self.n_jobs,
        )
        t0 = time.perf_counter()
        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            clf.fit(X_tr, y_tr)
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            y_pred = clf.predict(X_te)
        pred_t = time.perf_counter() - t1
        return fit_t, pred_t, y_pred


class ManualRunner(KNNRunner):
    impl_tag = "manual-brute"
    algorithm = "brute_manual"

    def __init__(self, k: int, n_jobs: int, backend: str):
        self.k = k
        self.n_jobs = n_jobs
        self.backend = backend

    def fit_predict(self, X_tr, y_tr, X_te):
        # fit is lazy (no index, just stash references)
        fit_t = 0.0
        t1 = time.perf_counter()
        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            preds = Parallel()(
                delayed(predict_one)(x, X_tr, y_tr, self.k) for x in X_te
            )
        pred_t = time.perf_counter() - t1
        return fit_t, pred_t, np.array(preds, dtype=y_tr.dtype)


def build_runner(args) -> KNNRunner:
    if args.impl.startswith("sklearn-"):
        algo = args.impl[len("sklearn-"):]
        return SklearnRunner(k=args.k, algorithm=algo,
                             n_jobs=args.jobs, backend=args.backend)
    if args.impl == "manual-brute":
        return ManualRunner(k=args.k, n_jobs=args.jobs, backend=args.backend)
    raise ValueError(f"unknown impl: {args.impl}")


# ---------- CLI / main ----------
CSV_HEADER = [
    "implementation", "omp_num_threads", "n_jobs", "backend", "algorithm",
    "n_train", "n_test", "n_features", "k",
    "feat_mult", "feat_mode", "mult_train", "mult_test", "jitter", "reps",
    "fit_time_s_avg", "pred_time_s_avg", "accuracy_avg",
    "flops_distance", "flops_select", "flops_total",
    "slurm_job_id", "slurm_array_task_id",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Unified KNN runner (sklearn + manual brute) for the heart-disease scalability sweep.",
    )
    ap.add_argument("--impl", required=True, choices=IMPL_CHOICES)
    ap.add_argument("--backend", default="threading", choices=["threading", "loky"])
    ap.add_argument("--jobs", type=int, default=1, help="n_jobs for joblib / sklearn")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat-mult", type=int, default=1)
    ap.add_argument("--feat-mode", type=str, default="mix", choices=["repeat", "mix"])
    ap.add_argument("--mult-train", type=int, default=1)
    ap.add_argument("--mult-test", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--no-scale", action="store_true",
                    help="Skip StandardScaler.")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output", type=str, default="results_knn.csv")
    return ap.parse_args()


def main():
    args = parse_args()

    X, y = load_cleveland(args.csv)
    X = expand_features(X, feature_mult=args.feat_mult, mode=args.feat_mode, seed=args.seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    X_train, y_train = expand_samples(X_train, y_train, factor=args.mult_train,
                                      jitter=args.jitter, seed=args.seed)
    X_test,  y_test  = expand_samples(X_test,  y_test,  factor=args.mult_test,
                                      jitter=args.jitter, seed=args.seed)

    if not args.no_scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    n_train, d = X_train.shape
    n_test = X_test.shape[0]

    runner = build_runner(args)

    # Warm-up: first run pays for import/init overheads (matters for tiny tasks).
    _ = runner.fit_predict(X_train, y_train, X_test[:min(8, n_test)])

    fit_times, pred_times, accs = [], [], []
    for _ in range(args.reps):
        fit_t, pred_t, y_pred = runner.fit_predict(X_train, y_train, X_test)
        fit_times.append(fit_t)
        pred_times.append(pred_t)
        accs.append(accuracy_score(y_test, y_pred))

    fit_avg = float(np.mean(fit_times))
    pred_avg = float(np.mean(pred_times))
    acc_avg = float(np.mean(accs))
    fdist, fsel, ftotal = estimate_flops(n_train, n_test, d, args.k)

    row = {
        "implementation": runner.impl_tag,
        "omp_num_threads": OMP_THREADS,
        "n_jobs": args.jobs,
        "backend": args.backend,
        "algorithm": runner.algorithm,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": d,
        "k": args.k,
        "feat_mult": args.feat_mult,
        "feat_mode": args.feat_mode,
        "mult_train": args.mult_train,
        "mult_test": args.mult_test,
        "jitter": args.jitter,
        "reps": args.reps,
        "fit_time_s_avg": round(fit_avg, 4),
        "pred_time_s_avg": round(pred_avg, 4),
        "accuracy_avg": round(acc_avg, 4),
        "flops_distance": f"{fdist:.2e}",
        "flops_select": f"{fsel:.2e}",
        "flops_total": f"{ftotal:.2e}",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"[OK] impl={runner.impl_tag} omp={OMP_THREADS} jobs={args.jobs} "
          f"backend={args.backend} n_train={n_train} n_test={n_test} d={d} "
          f"fit_avg={fit_avg:.4f}s pred_avg={pred_avg:.4f}s acc={acc_avg:.4f} "
          f"-> {args.output}")


if __name__ == "__main__":
    main()
