#!/usr/bin/env python3
"""Build the attribute projection Z used by RIPPLE (single process).

For every node v, sample a connected neighborhood of at most m_max nodes with W random walks of
length 3 on the symmetrized level-0 graph, build its Laplacian, and project the attribute encodings
onto the r eigenvectors with the smallest positive eigenvalues (variant skip_const). The output is
a raw float32 matrix with one row per HNSW *internal* id and one column per attribute-value pair.
The index itself is not modified. build_spectral_agg_par.py runs the same computation in parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


def read_fbin(path: Path):
    h = np.fromfile(path, dtype=np.uint32, count=2)
    if h.size != 2:
        raise ValueError(f"invalid fbin header: {path}")
    n, d = map(int, h)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))


def load_labels(ds: Path):
    ip = np.load(ds / "base_labels.indptr.npy", mmap_mode="r")
    ix = np.load(ds / "base_labels.indices.npy", mmap_mode="r")
    mf = ds / "manifest.json"
    if mf.exists():
        n_labels = int(json.loads(mf.read_text()).get("n_labels", int(ix.max()) + 1))
    else:
        n_labels = int(ix.max()) + 1
    return ip, ix, n_labels


def load_graph(tree: Path, index: Path, dim: int, n_labels: int):
    sys.path.insert(0, str(tree.resolve()))
    import hnswlib  # type: ignore
    mod = Path(hnswlib.__file__).resolve()
    if mod.parent != tree.resolve():
        raise RuntimeError(f"wrong hnswlib module: {mod}; expected {tree}")
    idx = hnswlib.Index(space="l2", dim=int(dim), dim_attr=int(n_labels))
    idx.load_index(str(index))
    idx.set_num_threads(1)
    g = idx.get_level0_graph()
    if not isinstance(g, (tuple, list)) or len(g) != 3:
        raise RuntimeError("get_level0_graph() must return (indptr, indices, labels)")
    ip, ix, labels = (np.asarray(x) for x in g)
    ip = ip.astype(np.int64, copy=False)
    ix = ix.astype(np.int32, copy=False)
    labels = labels.astype(np.int64, copy=False)
    n = len(labels)
    if ip.shape != (n + 1,) or int(ip[-1]) != len(ix):
        raise RuntimeError("inconsistent level-0 graph CSR")
    if not np.array_equal(np.sort(labels), np.arange(n, dtype=np.int64)):
        raise RuntimeError("graph labels are not an internal/external permutation")
    return idx, ip, ix, labels


def symmetrize(ip: np.ndarray, ix: np.ndarray):
    """Build an exact undirected CSR from the directed level-0 CSR.

    The temporary arrays are int32 and are deleted before the spectral loop;
    their size is recorded in metadata so the memory cost is visible.
    """
    n = len(ip) - 1
    deg = np.diff(ip).astype(np.int64, copy=False)
    src = np.repeat(np.arange(n, dtype=np.int32), deg)
    dst = ix.astype(np.int32, copy=True)
    a = np.concatenate((src, dst))
    b = np.concatenate((dst, src))
    order = np.lexsort((b, a))
    a = a[order]
    b = b[order]
    keep = np.ones(len(a), dtype=bool)
    if len(keep) > 1:
        keep[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    a = a[keep]
    b = b[keep]
    out_deg = np.bincount(a.astype(np.int64), minlength=n)
    out_ip = np.empty(n + 1, dtype=np.int64)
    out_ip[0] = 0
    np.cumsum(out_deg, out=out_ip[1:])
    out_ix = b.astype(np.int32, copy=False)
    del src, dst, order, keep, a, b, out_deg
    return out_ip, out_ix


def row_labels(internal_ids, labels, lab_ip, lab_ix, n_labels):
    y = np.zeros((len(internal_ids), n_labels), dtype=np.float32)
    for j, ii in enumerate(np.asarray(internal_ids, dtype=np.int64)):
        ext = int(labels[int(ii)])
        y[j, lab_ix[lab_ip[ext]:lab_ip[ext + 1]].astype(np.int64, copy=False)] = 1.0
    return y


def node_spectral(node, X, sip, six, labels, lab_ip, lab_ix, n_labels,
                  walks, m_max, rank, variant, rng):
    """Return z(v,:) and the sampled internal-id neighborhood."""
    chosen = [int(node)]
    seen = {int(node)}
    current = int(node)
    for _ in range(int(walks)):
        current = int(node)
        # Each walk has length three; walk starts are independent at v.
        for _step in range(3):
            lo, hi = int(sip[current]), int(sip[current + 1])
            if hi <= lo:
                break
            current = int(six[lo + int(rng.integers(hi - lo))])
            if current not in seen:
                seen.add(current)
                chosen.append(current)
                if len(chosen) >= m_max:
                    break
        if len(chosen) >= m_max:
            break
    ids = np.asarray(chosen[:m_max], dtype=np.int32)
    m = len(ids)
    pos = {int(v): i for i, v in enumerate(ids.tolist())}
    A = np.zeros((m, m), dtype=np.float64)
    for i, u in enumerate(ids.tolist()):
        for w in six[int(sip[u]):int(sip[u + 1])].tolist():
            j = pos.get(int(w))
            if j is not None:
                A[i, j] = 1.0
                A[j, i] = 1.0
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    ew, U = np.linalg.eigh(L)
    if variant == "skip_const":
        cols = np.arange(1, min(m, rank + 1), dtype=np.int64)
    elif variant == "keep_const":
        cols = np.arange(0, min(m, rank), dtype=np.int64)
    else:
        raise ValueError(f"unknown variant {variant}")
    if len(cols) == 0:
        return np.zeros(n_labels, dtype=np.float32), ids
    Y = row_labels(ids, labels, lab_ip, lab_ix, n_labels).astype(np.float64, copy=False)
    weights = U[0, cols] @ U[:, cols].T
    z = weights @ Y
    return z.astype(np.float32), ids


def sanity_full_basis(X, sip, six, labels, lab_ip, lab_ix, n_labels, seed, count):
    rng = np.random.default_rng(seed)
    n = len(labels)
    errs = []
    for node in rng.choice(n, size=min(count, n), replace=False):
        z, ids = node_spectral(int(node), X, sip, six, labels, lab_ip, lab_ix,
                                n_labels, walks=32, m_max=max(1, len(ids) if False else 64),
                                rank=64, variant="keep_const", rng=rng)
        # The full-basis check is only valid when the sampled neighborhood is
        # smaller than the requested rank.  Recompute with its exact size.
        z, ids = node_spectral(int(node), X, sip, six, labels, lab_ip, lab_ix,
                                n_labels, walks=32, m_max=64, rank=64,
                                variant="keep_const", rng=rng)
        own = row_labels([int(node)], labels, lab_ip, lab_ix, n_labels)[0]
        errs.append(float(np.max(np.abs(z - own))))
    return max(errs) if errs else 0.0


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
    ap.add_argument("--sanity-count", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    ds = Path(args.ds_dir); out = Path(args.out_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    X = read_fbin(ds / "base.fbin")
    lab_ip, lab_ix, n_labels = load_labels(ds)
    idx, ip, ix, labels = load_graph(Path(args.tree), Path(args.index), X.shape[1], n_labels)
    if len(X) != len(labels):
        raise RuntimeError("base and graph counts differ")
    graph_t0 = time.time(); sip, six = symmetrize(ip, ix); graph_sec = time.time() - graph_t0
    n = len(labels); limit = n if args.max_nodes <= 0 else min(n, args.max_nodes)
    name = f"z_r{args.rank}_{args.variant}_W{args.W}_m{args.m_max}"
    raw = out / f"{name}.f32"
    meta_path = out / f"{name}.json"
    if raw.exists() or meta_path.exists():
        raise SystemExit(f"output exists: {raw}; choose a fresh out-root")
    zmap = np.memmap(raw, mode="w+", dtype=np.float32, shape=(limit, n_labels))
    rng = np.random.default_rng(args.seed)
    t_loop = time.time()
    for node in range(limit):
        zmap[node], _ = node_spectral(node, X, sip, six, labels, lab_ip, lab_ix,
                                      n_labels, args.W, args.m_max, args.rank,
                                      args.variant, rng)
        if (node + 1) % 10000 == 0 or node + 1 == limit:
            print(json.dumps({"node": node + 1, "total": limit,
                              "elapsed": time.time() - t_loop}), flush=True)
    zmap.flush(); del zmap
    sanity = None
    if args.sanity_count:
        sanity = sanity_full_basis(X, sip, six, labels, lab_ip, lab_ix,
                                   n_labels, args.seed + 17, args.sanity_count)
    files = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    meta = {
        "dataset": args.dataset, "n": int(n), "rows_written": int(limit),
        "dim": int(X.shape[1]), "n_labels": int(n_labels), "W": int(args.W),
        "walk_length": 3, "m_max": int(args.m_max), "rank": int(args.rank),
        "variant": args.variant, "seed": int(args.seed),
        "graph": "exact symmetrization of level-0 directed CSR",
        "graph_edges_directed": int(len(ix)), "graph_edges_undirected": int(len(six)),
        "symmetrize_seconds": float(graph_sec), "wall_seconds": float(time.time() - t0),
        "seconds_per_1e5_rows": float((time.time() - t_loop) / max(1, limit) * 1e5),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "rows_written": int(limit), "partial": bool(limit != n),
        "sanity_full_basis_max_abs_err": sanity, "files": files,
        "module": str(Path(idx.__class__.__module__) if False else "hnswlib"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"raw": str(raw), "meta": str(meta_path),
                      "wall_seconds": meta["wall_seconds"],
                      "seconds_per_1e5_rows": meta["seconds_per_1e5_rows"]}))


if __name__ == "__main__":
    main()
