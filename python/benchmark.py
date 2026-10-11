import argparse, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description='End-to-end baseline benchmark')
p.add_argument('--rows', type=int, default=10000); p.add_argument('--dim', type=int, default=128)
p.add_argument('--queries', type=int, default=100); p.add_argument('--k', type=int, default=10); p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
if a.k < 1: p.error('--k must be positive')
subprocess.run(['python3', str(ROOT/'python/generate_data.py'), '--rows', str(a.rows), '--dim', str(a.dim), '--queries', str(a.queries), '--seed', str(a.seed)], check=True)
binary = ROOT/'build/vector_search'
if not binary.exists(): raise SystemExit('Build missing; run make first.')
start = time.perf_counter()
proc = subprocess.run([str(binary), str(ROOT/'data/vectors.csv'), str(ROOT/'data/queries.csv'), str(a.k), str(ROOT/'results/topk.csv')], check=True, capture_output=True, text=True)
elapsed = time.perf_counter() - start
print(proc.stdout.strip())
print(f'Corpus: {a.rows:,} x {a.dim}; queries={a.queries:,}; k={a.k}')
print(f'End-to-end time (CSV loading + search + output): {elapsed:.4f} s')
print(f'Amortized latency: {elapsed * 1000 / a.queries:.3f} ms/query')
print(f'Throughput: {a.queries / elapsed:.2f} queries/s')
print('Note: this is process-level latency, not isolated per-query latency.')
