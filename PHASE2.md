# VectorForge Phase 2: HNSW foundation

Copy these files into the matching paths in your existing repository. Merge the target in `Makefile.phase2-snippet` into your existing Makefile; do not replace the Phase 1 targets.

Run `make hnsw-test`.

This is an educational HNSW foundation, not yet production-ready. Candidate selection uses a simple similarity ranking and does not implement the full diversity heuristic; insertion currently checks duplicate IDs in O(N). We will address correctness evaluation and optimization next.

API:
```cpp
vectorforge::HnswConfig config;
config.M = 16;
config.ef_construction = 200;
config.ef_search = 64;
vectorforge::HnswIndex index(config);
index.add(1001, {0.1F, 0.2F, 0.3F});
auto results = index.search({0.1F, 0.2F, 0.3F}, 1);
```
