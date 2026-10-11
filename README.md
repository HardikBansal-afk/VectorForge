# VectorForge — Phase 1: Exact Vector Search

C++20 brute-force cosine-similarity baseline with Python dataset generation, correctness verification, and benchmark tooling. Target: Apple Silicon macOS.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
make
```

## Smoke test

```bash
make test
```

## Larger test and benchmark

```bash
python python/generate_data.py --rows 10000 --dim 128 --queries 20 --seed 42
./build/vector_search data/vectors.csv data/queries.csv 10 results/topk.csv
python python/verify_results.py --vectors data/vectors.csv --queries data/queries.csv --results results/topk.csv --k 10
python python/benchmark.py --rows 10000 --dim 128 --queries 100 --k 10 --seed 42
```

## Algorithm

For every query, scan all corpus vectors, calculate cosine similarity, and keep the best k results in a bounded heap. Results are sorted by descending score, then ascending vector ID for deterministic tie-breaking. The Python verifier independently calculates exact cosine scores with NumPy and checks both IDs and scores.

## Limitations

This is a single-threaded exact baseline. CSV parsing and process startup are included in the benchmark's end-to-end timing. It is not an ANN index or a million-vector performance claim. Phase 2 will add HNSW and recall@k comparisons.
