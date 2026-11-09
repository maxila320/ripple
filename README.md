# RIPPLE

RIPPLE performs filtered approximate nearest neighbor search on an HNSW graph. It precomputes a low-frequency projection of node attributes using local graph Laplacians. During search, it uses navigation scores to prune neighbors that fail the filter and an adaptive radius to decide when to stop.

## Build

Requires Python 3.10, NumPy, pybind11, and a C++14 compiler with OpenMP. Run all commands from the repository root.

```bash
(cd src && python3 setup.py build_ext --inplace)
```

## Prepare the data

Set `DS_DIR` to the dataset directory and `DATASET` to its identifier. Set `INDEX_DIR`, `FIELD_DIR`, and `OUT_DIR` to your output paths, `WORKLOAD` to the workload name, and `CPU` to the CPU core ID.

The dataset directory contains `base.fbin`, `base_labels.indptr.npy`, and `base_labels.indices.npy`. Put query vectors, query labels, and exact neighbor IDs (`gt_ids.ibin`) in each of the `tuning` and `final_test` splits under `workloads/$WORKLOAD/`.

## 1. Build the index

```bash
python3 scripts/build_index.py --tree src --ds-dir "$DS_DIR" \
    --out "$INDEX_DIR/index.bin" --threads 30
```

This writes `index.bin` and `index_attr.bin`.

## 2. Project node attributes

```bash
python3 scripts/build_spectral_agg_par.py \
    --dataset "$DATASET" --ds-dir "$DS_DIR" \
    --index "$INDEX_DIR/index.bin" --tree src \
    --W 32 --m-max 64 --rank 4 --variant skip_const \
    --seed 100 --rng-mode stream --procs 30 --out-root "$FIELD_DIR"

FIELD="$FIELD_DIR/$DATASET/z_r4_skip_const_W32_m64.f32"
```

The projection table follows the index's internal node order. Regenerate it whenever you rebuild the index.

## 3. Calibrate and evaluate

Calibrate the pruning threshold on the tuning split, then evaluate on the final test split.

```bash
python3 scripts/ripple_run.py calib --tree src \
    --index "$INDEX_DIR/index.bin" --ds-dir "$DS_DIR" \
    --dataset "$DATASET" --workload "$WORKLOAD" \
    --field "$FIELD" --out "$OUT_DIR"

python3 scripts/ripple_run.py run --tree src \
    --index "$INDEX_DIR/index.bin" --ds-dir "$DS_DIR" \
    --dataset "$DATASET" --workload "$WORKLOAD" \
    --field "$FIELD" --rho 0.25 --repeats 1 --core "$CPU" --out "$OUT_DIR"
```

`--rho` sets the pruning quantile (`0` disables pruning). The run sweeps the radius factor and writes recall@10, QPS, and distance computations per query to `raw_runs.csv`.

## Acknowledgments

The graph index is an HNSW graph built with [hnswlib](https://github.com/nmslib/hnswlib), following

- Yu. A. Malkov and D. A. Yashunin. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 2020. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

The code that builds and stores the per-node attribute vectors is adapted from the RWalks implementation:

- A. Ait Aomar, K. Echihabi, M. Arnaboldi, I. Alagiannis, D. Hilloulin, and M. Cherkaoui. RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search. *Proceedings of the ACM on Management of Data*, 3(3), 2025. [doi:10.1145/3725349](https://doi.org/10.1145/3725349)

## License

This repository is released under the Apache License 2.0; see [LICENSE](LICENSE). It contains code derived from hnswlib (Apache License 2.0) and RWalks; see [NOTICE](NOTICE) for attribution.