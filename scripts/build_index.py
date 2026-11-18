#!/usr/bin/env python3
"""Build the graph index searched by RIPPLE: HNSW (M = 32, efConstruction = 200) whose nodes store
the attribute encodings and the attribute aggregates (random walks inside add_items).

    python3 scripts/build_index.py --tree src --ds-dir <ds-dir> --out <dir>/index.bin --threads 30

Writes <dir>/index.bin, <dir>/index_attr.bin (attribute encodings and aggregates) and
<dir>/build_info.json. The attribute projection (build_spectral_agg_par.py) must be built from
the same index, because its rows follow the internal ids of this graph.
"""
import argparse, json, os, resource, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ripple_run import load_bin   # same reader as the search scripts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="folder that holds the compiled hnswlib .so (src)")
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--out", required=True, help="path of index.bin (index_attr.bin is written next to it)")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--threads", type=int, default=30)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, os.path.abspath(a.tree))
    import hnswlib
    if os.path.dirname(os.path.abspath(hnswlib.__file__)) != os.path.abspath(a.tree):
        sys.exit(f"wrong hnswlib loaded: {hnswlib.__file__}")

    D = Path(a.ds_dir)
    X = np.ascontiguousarray(load_bin(D / "base.fbin"), dtype=np.float32)
    bip = np.load(D / "base_labels.indptr.npy")
    bix = np.load(D / "base_labels.indices.npy")
    mf = D / "manifest.json"
    nl = json.loads(mf.read_text()).get("n_labels") if mf.exists() else None
    n_labels = int(nl or int(bix.max()) + 1)
    n, dim = X.shape

    # multi-hot attribute encodings y_i in {0,1}^T, one row per object (row id = external label)
    A = np.zeros((n, n_labels), dtype=np.int32)
    A[np.repeat(np.arange(n), np.diff(bip)), bix] = 1

    idx = hnswlib.Index(space="l2", dim=dim, dim_attr=n_labels)
    idx.init_index(max_elements=n, M=a.M, ef_construction=a.ef_construction, random_seed=a.seed)
    t = time.time()
    idx.add_items(X, A, num_threads=a.threads)   # one call: insertion, then RWalks aggregation over all nodes
    secs = time.time() - t
    idx.save_index(str(out))

    attr = out.with_name(out.name.replace(".bin", "_attr.bin"))
    info = {"build_seconds": secs, "n": n, "dim": dim, "n_labels": n_labels,
            "index_bytes": out.stat().st_size + (attr.stat().st_size if attr.exists() else 0),
            "build_params": {"M": a.M, "ef_construction": a.ef_construction, "random_seed": a.seed,
                             "build_threads": a.threads,
                             "walks": "getAttrAggregate(node, 5, 10, 1.0)"},
            "build_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    (out.parent / "build_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
