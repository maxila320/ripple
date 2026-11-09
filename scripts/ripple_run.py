#!/usr/bin/env python3
"""Run RIPPLE: threshold calibration and recall/QPS measurements.

Sub-commands
  probe  quick check that c and the pruning quantile rho change the search (2-3 minutes)
  calib  compute theta_rho (rho-quantile of the navigation scores of invalid objects, tuning split)
  run    full timed grid over the radius factor c for one dataset + workload; resumable;
         validity check at the end

Data layout:
  <ds-dir>/base.fbin, base_labels.indptr.npy, base_labels.indices.npy, manifest.json
  <ds-dir>/workloads/<workload>/<split>/queries.fbin, query_labels.indptr.npy, query_labels.indices.npy
Ground truth: --gt FILE (.ibin/.npy, n_queries x >=k ids); otherwise the script looks for
*gt*/*groundtruth*/*neighbors* in the split folder; otherwise it computes exact filtered GT
and caches it in --out. When a GT file is found, the first 20 queries are re-checked exactly.

Nothing outside --out is written.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

K = 10
RHOS = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90]   # pruning quantiles; rho = 0 disables pruning
C_GRID = [round(0.90 + 0.02 * i, 2) for i in range(16)] + [1.25, 1.30, 1.40]   # radius factor c
EF = 2048                 # capacity of the internal result list
MAX_SCORED = 200000       # safety budget on distance computations per query


def rho_name(rho):
    return f"rho{float(rho):.2f}"
LOAD_LIMIT = 0.7 * 240


# ---------------------------------------------------------------- data
def load_bin(path):
    hdr = np.fromfile(path, dtype=np.uint32, count=2)
    n, d = int(hdr[0]), int(hdr[1])
    dt = np.float32 if str(path).endswith(".fbin") else np.int32
    return np.memmap(path, dtype=dt, mode="r", offset=8, shape=(n, d))


class Data:
    def __init__(self, ds_dir, workload, split):
        D = Path(ds_dir)
        self.D, self.W = D, D / "workloads" / workload / split
        self.X = load_bin(D / "base.fbin")
        self.bip = np.load(D / "base_labels.indptr.npy")
        self.bix = np.load(D / "base_labels.indices.npy")
        mf = D / "manifest.json"
        nl = json.loads(mf.read_text()).get("n_labels") if mf.exists() else None
        self.n_labels = int(nl or int(self.bix.max()) + 1)
        self.Q = np.ascontiguousarray(load_bin(self.W / "queries.fbin"), dtype=np.float32)
        self.qip = np.load(self.W / "query_labels.indptr.npy")
        self.qix = np.load(self.W / "query_labels.indices.npy")
        rows = np.repeat(np.arange(len(self.bip) - 1), np.diff(self.bip))
        order = np.argsort(self.bix, kind="stable")
        self.post_rows = rows[order]
        self.post_ptr = np.searchsorted(self.bix[order], np.arange(self.n_labels + 1))
        self.QA = np.zeros((len(self.Q), self.n_labels), dtype=np.int32)
        for i in range(len(self.Q)):
            self.QA[i, self.labels(i)] = 1

    def labels(self, i):
        return self.qix[self.qip[i]:self.qip[i + 1]].astype(np.int64)

    def valid(self, i):
        m = None
        for l in self.labels(i):
            mk = np.zeros(len(self.X), bool)
            mk[self.post_rows[self.post_ptr[l]:self.post_ptr[l + 1]]] = True
            m = mk if m is None else m & mk
        return np.ones(len(self.X), bool) if m is None else m

    def exact(self, i, k=K):
        ids = np.flatnonzero(self.valid(i))
        if len(ids) == 0:
            return np.array([], dtype=np.int64)
        V = np.asarray(self.X[ids], dtype=np.float32)
        d = ((V - self.Q[i]) ** 2).sum(1)
        o = np.argsort(d, kind="stable")[:k]
        return ids[o]


def load_gt(data, gt_arg, out):
    cand = [gt_arg] if gt_arg else sorted(
        p for pat in ("*gt*", "*groundtruth*", "*neighbors*") for p in glob.glob(str(data.W / pat))
        if p.endswith((".ibin", ".npy")))
    if cand:
        p = cand[0]
        G = np.load(p) if p.endswith(".npy") else np.asarray(load_bin(p))
        G = np.asarray(G)[:, :K].astype(np.int64)
        bad = sum(len(set(G[i]) ^ set(data.exact(i))) > 0 for i in range(min(20, len(G))))
        print(f"GT from {p}; exact re-check of 20 queries: {20 - bad}/20 identical", flush=True)
        if bad > 2:
            sys.exit("GT file disagrees with exact filtered search; pass --gt or delete the guess")
        return [g[g >= 0] for g in G]
    cache = Path(out) / "gt_exact.npy"
    if cache.exists():
        G = np.load(cache, allow_pickle=True)
        return list(G)
    print("computing exact filtered GT ...", flush=True)
    G = [data.exact(i) for i in range(len(data.Q))]
    np.save(cache, np.array(G, dtype=object), allow_pickle=True)
    return G


def recall(labels, G):
    r = []
    for row, g in zip(labels, G):
        if len(g):
            r.append(len(np.intersect1d(row, g)) / len(g))
    return float(np.mean(r))


# ---------------------------------------------------------------- index
def load_index(tree, index_path, dim, n_labels):
    sys.path.insert(0, os.path.abspath(tree))
    import hnswlib
    if os.path.dirname(os.path.abspath(hnswlib.__file__)) != os.path.abspath(tree):
        sys.exit(f"wrong hnswlib loaded: {hnswlib.__file__}")
    idx = hnswlib.Index(space="l2", dim=dim, dim_attr=n_labels)
    idx.load_index(index_path)
    idx.set_num_threads(1)
    so = hashlib.sha256(open(hnswlib.__file__, "rb").read()).hexdigest()
    return idx, so


def load_field(path, n):
    z = np.fromfile(path, dtype=np.float32)
    if z.size % n:
        sys.exit(f"field size {z.size} not a multiple of n={n}")
    return z.reshape(n, -1)


def configure(idx, cfg, c, theta):
    """cfg = rho_name(rho); theta maps rho names to the calibrated thresholds."""
    idx.set_ef(EF)
    idx.set_search_params(k_target=K, c=float(c), max_scored=MAX_SCORED)
    idx.set_prune_threshold(float(theta[cfg]))

def timed(idx, Q, QA, warm):
    idx.knn_query(Q[:warm], QA[:warm], k=K, num_threads=1)   # warm-up (not timed)
    if hasattr(idx, "reset_scored_count"):
        idx.reset_scored_count()
    t = time.perf_counter()
    lab, _ = idx.knn_query(Q, QA, k=K, num_threads=1)
    el = time.perf_counter() - t
    sc = idx.get_scored_count() / len(Q) if hasattr(idx, "get_scored_count") else np.nan
    return lab, len(Q) / el, sc


def wait_load():
    t0, flag = time.time(), 0
    while os.getloadavg()[0] > LOAD_LIMIT:
        if time.time() - t0 > 1800:
            flag = 1
            break
        time.sleep(60)
    return " ".join(f"{x:.2f}" for x in os.getloadavg()), flag


# ---------------------------------------------------------------- theta
def calibrate(a):
    tune = Data(a.ds_dir, a.workload, "tuning")
    z = load_field(a.field, len(tune.X))
    rng = np.random.default_rng(0)
    sample = rng.choice(len(tune.X), size=min(200000, len(tune.X)), replace=False)
    zs = z[sample]
    pool = []
    for i in range(min(200, len(tune.Q))):
        L = tune.labels(i)
        inv = ~tune.valid(i)[sample]
        pool.append(zs[inv][:, L].mean(1))
    pool = np.concatenate(pool)
    if len(pool) == 0:
        sys.exit("no invalid nodes in the sample for the tuning queries; cannot calibrate theta")
    theta = {rho_name(r): (-3.0e38 if r == 0 else float(np.quantile(pool, r))) for r in RHOS}
    frac0 = float((pool == 0).mean())
    out = Path(a.out) / "theta.json"
    json.dump(dict(theta=theta, n_scores=int(len(pool)), frac_exact_zero=frac0,
                   field=a.field), open(out, "w"), indent=1)
    print("theta:", theta, f"| share of scores exactly 0: {frac0:.3f}")
    if len({v for r, v in zip(RHOS, theta.values()) if r > 0}) < sum(r > 0 for r in RHOS):
        print("WARNING: different rho give the same theta (many tied scores)")
    return theta


def get_theta(a):
    p = Path(a.out) / "theta.json"
    need = {rho_name(r) for r in RHOS}
    if p.exists():
        th = json.load(open(p))["theta"]
        if need <= set(th):
            return th
    return calibrate(a)


# ---------------------------------------------------------------- commands
def common(a):
    os.makedirs(a.out, exist_ok=True)
    try:
        os.sched_setaffinity(0, {a.core})
    except Exception as e:
        print("could not pin core:", e)
    data = Data(a.ds_dir, a.workload, "final_test")
    idx, so = load_index(a.tree, a.index, data.X.shape[1], data.n_labels)
    idx.set_spectral_prune_table(load_field(a.field, len(data.X)))
    return data, idx, so


def cmd_probe(a):
    data, idx, _ = common(a)
    theta = get_theta(a)
    G = load_gt(data, a.gt, a.out)
    Q, QA, n = data.Q[:200], data.QA[:200], 200
    print("config        c       recall   qps      scored/q")
    for cfg in [rho_name(r) for r in a.rho]:
        for v in [0.95, 1.05, 1.20]:
            configure(idx, cfg, v, theta)
            lab, qps, sc = timed(idx, Q, QA, 20)
            print(f"{cfg:14s}{v:7g}  {recall(lab, G[:n]):.4f}  {qps:8.0f}  {sc:9.0f}")
    print("each rho must show clearly different recall / scored across the values of c")



def cmd_run(a):
    data, idx, so = common(a)
    out = Path(a.out)
    if (out / "DONE").exists():
        print("DONE exists, skipping")
        return
    theta = get_theta(a)
    G = load_gt(data, a.gt, a.out)
    Q, QA = data.Q, data.QA
    csv = out / "raw_runs.csv"
    done = set()
    if csv.exists():
        import pandas as pd
        old = pd.read_csv(csv)
        done = set(zip(old.config, old.repeat_id, old.c))
    cols = ["dataset", "workload", "config", "rho", "theta", "c", "repeat_id",
            "query_count", "warmup_queries", "recall", "qps", "latency_mean_ms",
            "scored_per_query", "loadavg_start", "load_flag", "cpu_core",
            "query_threads", "so_sha256"]
    if not csv.exists():
        csv.write_text(",".join(cols) + "\n")
    cfgs = [rho_name(r) for r in a.rho]
    for rep in range(a.repeats):
        jobs = [(c, v) for c in cfgs for v in C_GRID]
        jobs = [jobs[i] for i in np.random.default_rng(rep).permutation(len(jobs))]  # interleave configs
        for cfg, v in jobs:
            if (cfg, rep, float(v)) in done:
                continue
            la, flag = wait_load()
            configure(idx, cfg, v, theta)
            lab, qps, sc = timed(idx, Q, QA, a.warmup)
            row = [a.dataset, a.workload, cfg, cfg[3:], theta[cfg], v, rep,
                   len(Q), a.warmup, f"{recall(lab, G):.6f}", f"{qps:.4f}", f"{1000 / qps:.6f}",
                   f"{sc:.4f}", la, flag, a.core, 1, so]
            with open(csv, "a") as f:
                f.write(",".join(map(str, row)) + "\n")
            print(f"rep{rep} {cfg:16s} {v:<6g} recall={float(row[9]):.4f} qps={qps:.0f}", flush=True)
    # validity check
    import pandas as pd
    r = pd.read_csv(csv)
    bad = []
    for cfg, d in r[r.repeat_id == 0].groupby("config"):
        d = d.sort_values("c")
        distinct = d[["recall", "qps"]].round(6).drop_duplicates().shape[0]
        mono = (d.recall.diff().dropna() >= -0.002).all()
        if distinct < 5 or not mono:
            bad.append(f"{cfg}: distinct={distinct}, monotone={mono}")
    if bad:
        (out / "CHECK_FAILED").write_text("\n".join(bad) + "\n")
        print("CHECK FAILED:", bad)
    else:
        (out / "DONE").write_text(time.strftime("%F %T") + "\n")
        print("check passed, DONE written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "calib", "run"])
    ap.add_argument("--tree", required=True, help="folder that holds the compiled hnswlib .so (src)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--field", required=True, help="attribute projection z_*.f32 (n x n_labels float32)")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--rho", type=float, nargs="+", default=RHOS, help="pruning quantiles to run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--core", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"probe": cmd_probe, "run": cmd_run,
     "calib": lambda a: calibrate(a)}[a.cmd](a)


if __name__ == "__main__":
    main()