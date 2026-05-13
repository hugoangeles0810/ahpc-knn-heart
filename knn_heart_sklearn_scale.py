#!/usr/bin/env python3
# knn_heart_sklearn_scale.py
import os

# Evitar sobre-suscripción BLAS (debe ir antes de importar numpy/sklearn)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import sys
import time

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset", "dataset.csv",
)

# Data expansion utilities (same spirit as knn_digits_joblib_scale.py)
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
    elif mode == "mix":
        rng = np.random.RandomState(seed)
        W = rng.normal(0, 1, size=(d, (feature_mult - 1) * d))
        X_new = X @ W
        return np.concatenate([X, X_new], axis=1)
    else:
        raise ValueError("feat-mode must be 'repeat' or 'mix'")

def estimate_flops(n_train: int, n_test: int, d: int, k: int):
    flops_dist = n_test * n_train * (3.0 * d)
    flops_select = n_test * (5.0 * n_train)
    return flops_dist, flops_select, flops_dist + flops_select

def load_cleveland(path: str):
    df = pd.read_csv(path)
    # 'num' is the original (non-binarized) disease severity; dropping it avoids target leakage.
    y = df["target"].astype(np.int32).to_numpy()
    X = df.drop(columns=["target", "num"]).astype(np.float64).to_numpy()
    return X, y

def parse_args():
    ap = argparse.ArgumentParser(
        description="Sklearn KNN on Cleveland heart disease with joblib-style n_jobs parallelism."
    )
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--jobs", type=int, default=-1, help="-1: all cores")
    ap.add_argument("--backend", type=str, default="loky", choices=["loky", "threading"])
    ap.add_argument("--algorithm", type=str, default="brute",
                    choices=["brute", "kd_tree", "ball_tree", "auto"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat-mult", type=int, default=1)
    ap.add_argument("--feat-mode", type=str, default="repeat", choices=["repeat", "mix"])
    ap.add_argument("--mult-train", type=int, default=1)
    ap.add_argument("--mult-test", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--no-scale", action="store_true",
                    help="Skip StandardScaler (KNN is distance-based, scaling is normally recommended).")
    ap.add_argument("--reps", type=int, default=3, help="repeticiones para promediar tiempos")
    ap.add_argument("--output", type=str, default="results_knn_heart.csv")
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

    fit_times, pred_times, accs = [], [], []

    for r in range(args.reps):
        clf = KNeighborsClassifier(
            n_neighbors=args.k,
            algorithm=args.algorithm,
            metric="euclidean",
            n_jobs=args.jobs,
        )

        # Entrenamiento (fit en KNN es lazy: solo guarda referencias / construye índice si aplica)
        t0 = time.perf_counter()
        with parallel_backend(backend=args.backend, n_jobs=args.jobs):
            clf.fit(X_train, y_train)
        fit_t = time.perf_counter() - t0

        # Warm-up SOLO en la primera repetición para sacar overheads de import/BLAS del cronómetro
        if r == 0:
            _ = clf.predict(X_test[:min(8, n_test)])

        # Predicción (aquí es donde n_jobs realmente trabaja en KNN)
        t1 = time.perf_counter()
        with parallel_backend(backend=args.backend, n_jobs=args.jobs):
            y_pred = clf.predict(X_test)
        pred_t = time.perf_counter() - t1

        acc = accuracy_score(y_test, y_pred)
        fit_times.append(fit_t)
        pred_times.append(pred_t)
        accs.append(acc)

    fit_avg = float(np.mean(fit_times))
    pred_avg = float(np.mean(pred_times))
    acc_avg = float(np.mean(accs))
    fdist, fsel, ftotal = estimate_flops(n_train, n_test, d, args.k)

    # ===== Guardar resultados =====
    header = [
        "n_jobs", "backend", "algorithm",
        "n_train", "n_test", "n_features", "k",
        "feat_mult", "feat_mode", "mult_train", "mult_test", "jitter",
        "reps", "fit_time_s_avg", "pred_time_s_avg", "accuracy_avg",
        "flops_distance", "flops_select", "flops_total",
        "slurm_job_id", "slurm_array_task_id",
    ]
    row = {
        "n_jobs": args.jobs,
        "backend": args.backend,
        "algorithm": args.algorithm,
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
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"[OK] n_jobs={args.jobs} n_train={n_train} n_test={n_test} d={d} k={args.k} "
          f"fit_avg={fit_avg:.3f}s pred_avg={pred_avg:.3f}s acc_avg={acc_avg:.4f} "
          f"-> {args.output}")

if __name__ == "__main__":
    main()
