import argparse, csv
from pathlib import Path
import numpy as np

def load(path):
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.reader(f); header = next(r)
        ids, vectors = [], []
        for row in r: ids.append(int(row[0])); vectors.append([float(x) for x in row[1:]])
    return np.asarray(ids, dtype=np.int64), np.asarray(vectors, dtype=np.float64)

p = argparse.ArgumentParser(description='Verify C++ top-k against NumPy exact cosine search')
p.add_argument('--vectors', type=Path, required=True); p.add_argument('--queries', type=Path, required=True)
p.add_argument('--results', type=Path, required=True); p.add_argument('--k', type=int, required=True)
a = p.parse_args()
ids, corpus = load(a.vectors); qids, queries = load(a.queries)
if corpus.shape[1] != queries.shape[1]: raise SystemExit('FAIL: dimensions differ')
norms = np.linalg.norm(corpus, axis=1); norms[norms == 0] = 1; corpus = corpus / norms[:, None]
found = {}
with a.results.open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f): found.setdefault(int(row['query_id']), []).append((int(row['neighbor_id']), float(row['cosine_similarity'])))
for qid, q in zip(qids, queries, strict=True):
    qnorm = np.linalg.norm(q); q = q / qnorm if qnorm else q
    scores = corpus @ q
    order = np.lexsort((ids, -scores))[:min(a.k, len(ids))]
    expected_ids = ids[order].tolist(); actual = found.get(int(qid), [])
    if [x[0] for x in actual] != expected_ids:
        raise SystemExit(f'FAIL query {qid}: expected {expected_ids}, got {[x[0] for x in actual]}')
    if not np.allclose([x[1] for x in actual], scores[order], atol=2e-6, rtol=2e-6):
        raise SystemExit(f'FAIL query {qid}: score mismatch')
print(f'PASS: exact top-{a.k} IDs and scores verified for {len(qids)} queries.')
