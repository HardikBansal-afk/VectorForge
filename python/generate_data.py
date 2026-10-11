import argparse, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def write_csv(path, ids, matrix):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id'] + [f'v{i}' for i in range(matrix.shape[1])])
        for idx, vec in zip(ids, matrix, strict=True):
            w.writerow([int(idx), *[f'{float(x):.9g}' for x in vec]])

def normalized(rng, rows, dim):
    x = rng.standard_normal((rows, dim)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True); norms[norms == 0] = 1
    return x / norms

p = argparse.ArgumentParser(description='Generate deterministic synthetic vectors')
p.add_argument('--rows', type=int, default=10000); p.add_argument('--dim', type=int, default=128)
p.add_argument('--queries', type=int, default=20); p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
if min(a.rows, a.dim, a.queries) < 1: p.error('--rows, --dim and --queries must be positive')
rng = np.random.default_rng(a.seed)
write_csv(ROOT/'data/vectors.csv', np.arange(a.rows), normalized(rng, a.rows, a.dim))
write_csv(ROOT/'data/queries.csv', np.arange(a.queries), normalized(rng, a.queries, a.dim))
print(f'Generated {a.rows:,} vectors and {a.queries:,} queries; dimension={a.dim}; seed={a.seed}.')
