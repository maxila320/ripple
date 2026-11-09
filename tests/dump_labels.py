#!/usr/bin/env python3
"""Run RIPPLE for one pruning quantile rho and save the top-10 ids for several radius factors c,
to check that two builds of the library return identical results. Writes only into --out.

    taskset -c 51 python3 tests/dump_labels.py --tree src --index <index.bin> --ds-dir <dir> \
        --workload single_0.1-0.1 --field <z.f32> --theta-json <theta.json> --out <dir>

--theta-json is the theta.json written by `ripple_run.py calib`. The threshold for --rho is taken
from the entry whose name ends with that quantile (e.g. rho0.25).
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import ripple_run as R


def theta_for(table, rho):
    for key, val in table.items():
        m = re.search(r"(\d+(?:\.\d+)?)$", key)
        if m:
            q = float(m.group(1)) if "." in m.group(1) else int(m.group(1)) / 100.0
            if abs(q - rho) < 1e-9:
                return float(val)
    sys.exit(f"no threshold for rho={rho} in theta file")


ap = argparse.ArgumentParser()
for k in ("--tree", "--index", "--ds-dir", "--workload", "--field", "--theta-json", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--rho", type=float, default=0.25)
ap.add_argument("--c", type=float, nargs="+", default=[0.98, 1.06, 1.20])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
data = R.Data(a.ds_dir, a.workload, "final_test")
idx, so = R.load_index(a.tree, a.index, data.X.shape[1], data.n_labels)
idx.set_spectral_prune_table(R.load_field(a.field, len(data.X)))
cfg = R.rho_name(a.rho)
theta = {cfg: theta_for(json.load(open(a.theta_json))["theta"], a.rho)}
G = R.load_gt(data, None, a.out)
for c in a.c:
    R.configure(idx, cfg, c, theta)
    lab, qps, _ = R.timed(idx, data.Q, data.QA, warm=200)
    np.save(os.path.join(a.out, f"labels_c{c:.2f}.npy"), lab)
    print(f"c={c:.2f} recall={R.recall(lab, G):.4f} qps={qps:.0f} so_sha256={so[:12]}")
