#!/usr/bin/env python3
"""Parallel version of build_spectral_agg.py.

Every node is computed by node_spectral() of build_spectral_agg.py; only the loop over nodes is
split across processes. Two random-number modes:

  --rng-mode stream   (default) bit-identical to the sequential build_spectral_agg.py.
                      build_spectral_agg.py uses ONE generator for all nodes in order, so a
                      sequential pre-pass replays only the random draws of the walks (no
                      eigendecomposition) and records the generator state at the start of every
                      chunk; each worker restores that state and continues exactly as the
                      sequential loop would. The pre-pass runs in a small C kernel
                      (fast_prepass.py, compiled with cc on first use); --prepass python uses the
                      pure-Python replay instead (same states, much slower).
  --rng-mode pernode  generator seeded by (seed, node): no pre-pass, result independent of the
                      number of processes, but NOT bit-identical to tables built sequentially.

Same arguments and output file names as build_spectral_agg.py, plus --procs, --chunk,
--rng-mode, --verify-against (compare random rows with an existing table).

    python3 scripts/build_spectral_agg_par.py --procs 30 \
        --dataset sift1m_rwalks --ds-dir ... --index ... --tree src --out-root ... \
        --W 32 --m-max 64 --rank 4 --variant skip_const --seed 100 --rng-mode stream
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker (eigh), no oversubscription

import argparse
import copy
import importlib
import json
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_prepass  # noqa: E402  (same folder)
G = {}                                       # shared with forked workers (copy-on-write)


def walk_draws(node, sip, six, walks, m_max, rng):
    """Consume exactly the random numbers node_spectral() consumes for this node."""
    chosen = 1
    seen = {int(node)}
    for _ in range(int(walks)):
        current = int(node)
        for _step in range(3):
            lo, hi = int(sip[current]), int(sip[current + 1])
            if hi <= lo:
                break
            current = int(six[lo + int(rng.integers(hi - lo))])
            if current not in seen:
                seen.add(current)
                chosen += 1
                if chosen >= m_max:
                    break
        if chosen >= m_max:
            break


def work(task):
    a, b, state = task
    g = G
    mod = g["mod"]
    z = np.memmap(g["raw"], mode="r+", dtype=np.float32, shape=(g["limit"], g["n_labels"]))
    if g["rng_mode"] == "stream":
        rng = np.random.default_rng()
        rng.bit_generator.state = state
    for node in range(a, b):
        if g["rng_mode"] == "pernode":
            rng = np.random.default_rng([g["seed"], node])
        z[node], _ = mod.node_spectral(node, g["X"], g["sip"], g["six"], g["labels"], g["lab_ip"],
                                       g["lab_ix"], g["n_labels"], g["W"], g["m_max"], g["rank"],
                                       g["variant"], rng)
    z.flush()
    del z
    return b - a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out-root", default="results/spectral_agg")
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--variant", choices=("skip_const", "keep_const"), default="skip_const")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-nodes", type=int, default=0)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=0, help="nodes per task (default: n / (8*procs))")
    ap.add_argument("--rng-mode", choices=("stream", "pernode"), default="stream")
    ap.add_argument("--verify-against", default=None)
    ap.add_argument("--verify-rows", type=int, default=10000)
    ap.add_argument("--prepass", choices=("fast", "python"), default="fast",
                    help="how to compute the per-chunk generator states for --rng-mode stream (same result)")
    ap.add_argument("--check-prepass", type=int, default=0,
                    help="compare the first K chunk states with the Python replay (not timed; 0 = off)")
    args = ap.parse_args()

    mod = importlib.import_module("build_spectral_agg")
    t0 = time.time()
    ds = Path(args.ds_dir)
    out = Path(args.out_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    X = mod.read_fbin(ds / "base.fbin")
    lab_ip, lab_ix, n_labels = mod.load_labels(ds)
    idx, ip, ix, labels = mod.load_graph(Path(args.tree), Path(args.index), X.shape[1], n_labels)
    del idx                                   # free the index before forking
    if len(X) != len(labels):
        raise RuntimeError("base and graph counts differ")
    tg = time.time()
    sip, six = mod.symmetrize(ip, ix)
    graph_sec = time.time() - tg
    n_dir_edges = len(ix)
    del ip, ix
    n = len(labels)
    limit = n if args.max_nodes <= 0 else min(n, args.max_nodes)
    name = f"z_r{args.rank}_{args.variant}_W{args.W}_m{args.m_max}"
    raw = out / f"{name}.f32"
    meta_path = out / f"{name}.json"
    if raw.exists() or meta_path.exists():
        raise SystemExit(f"output exists: {raw}; choose a fresh out-root")
    z = np.memmap(raw, mode="w+", dtype=np.float32, shape=(limit, n_labels))
    del z

    chunk = args.chunk or max(1000, limit // (8 * args.procs))
    bounds = list(range(0, limit, chunk)) + [limit]
    tasks = []
    tp = time.time()
    check = None
    if args.rng_mode == "stream" and args.prepass == "fast":
        # generator state per chunk start, computed by a C kernel (fast_prepass.py); identical to the
        # sequential Python replay below
        states = fast_prepass.chunk_states(sip, six, bounds, args.W, args.m_max, args.seed)
        tasks = [(bounds[k], bounds[k + 1], states[k]) for k in range(len(bounds) - 1)]
        prepass_sec = time.time() - tp
        if args.check_prepass > 0:             # optional: compare with the Python replay on a prefix
            kchk = min(args.check_prepass, len(states))
            ref = fast_prepass.python_states(sip, six, bounds, args.W, args.m_max, args.seed, walk_draws,
                                             n_chunks=kchk)
            check = dict(chunks=kchk, identical=bool(all(fast_prepass.same_state(a, b)
                                                         for a, b in zip(ref, states[:kchk]))))
            print(json.dumps({"check_prepass": check}), flush=True)
            if not check["identical"]:
                raise SystemExit("fast pre-pass differs from the Python pre-pass; rerun with --prepass python")
    elif args.rng_mode == "stream":           # sequential Python pre-pass: generator state per chunk start
        rng = np.random.default_rng(args.seed)
        for k in range(len(bounds) - 1):
            tasks.append((bounds[k], bounds[k + 1], copy.deepcopy(rng.bit_generator.state)))
            for node in range(bounds[k], bounds[k + 1]):
                walk_draws(node, sip, six, args.W, args.m_max, rng)
        prepass_sec = time.time() - tp
    else:
        tasks = [(bounds[k], bounds[k + 1], None) for k in range(len(bounds) - 1)]
        prepass_sec = time.time() - tp
    print(json.dumps({"prepass_seconds": prepass_sec, "prepass": args.prepass, "tasks": len(tasks),
                      "chunk": chunk}), flush=True)

    G.update(mod=mod, X=X, sip=sip, six=six, labels=labels, lab_ip=lab_ip, lab_ix=lab_ix,
             n_labels=n_labels, W=args.W, m_max=args.m_max, rank=args.rank, variant=args.variant,
             seed=args.seed, rng_mode=args.rng_mode, raw=str(raw), limit=limit)
    tl = time.time()
    done = 0
    with mp.get_context("fork").Pool(args.procs) as pool:
        for k in pool.imap_unordered(work, tasks):
            done += k
            if done == limit or done // max(1, limit // 20) != (done - k) // max(1, limit // 20):
                print(json.dumps({"node": done, "total": limit, "elapsed": time.time() - tl}), flush=True)
    loop_sec = time.time() - tl

    verify = None
    if args.verify_against:
        old = np.memmap(args.verify_against, dtype=np.float32, mode="r")
        old = old.reshape(-1, n_labels)[:limit]
        new = np.memmap(raw, dtype=np.float32, mode="r", shape=(limit, n_labels))
        rows = np.random.default_rng(0).choice(limit, size=min(args.verify_rows, limit), replace=False)
        d = np.abs(np.asarray(new[rows]) - np.asarray(old[rows]))
        verify = dict(against=args.verify_against, rows=int(len(rows)),
                      identical_rows=int((d.max(1) == 0).sum()), max_abs_diff=float(d.max()))
        print(json.dumps({"verify": verify}), flush=True)

    meta = {
        "dataset": args.dataset, "n": int(n), "rows_written": int(limit), "partial": bool(limit != n),
        "dim": int(X.shape[1]), "n_labels": int(n_labels), "W": int(args.W), "walk_length": 3,
        "m_max": int(args.m_max), "rank": int(args.rank), "variant": args.variant, "seed": int(args.seed),
        "graph": "exact symmetrization of level-0 directed CSR",
        "graph_edges_directed": int(n_dir_edges), "graph_edges_undirected": int(len(six)),
        "parallel": {"procs": int(args.procs), "chunk": int(chunk), "rng_mode": args.rng_mode,
                     "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "symmetrize_seconds": float(graph_sec), "prepass_seconds": float(prepass_sec),
        "prepass": args.prepass, "check_prepass": check,
        "loop_seconds": float(loop_sec), "wall_seconds": float(time.time() - t0),
        "seconds_per_1e5_rows": float(loop_sec / max(1, limit) * 1e5),
        "peak_rss_bytes_parent": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "verify": verify, "implementation": mod.__file__,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"raw": str(raw), "meta": str(meta_path), "wall_seconds": meta["wall_seconds"],
                      "loop_seconds": loop_sec, "prepass_seconds": prepass_sec, "symmetrize_seconds": graph_sec,
                      "procs": args.procs}))


if __name__ == "__main__":
    main()