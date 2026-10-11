#include "../cpp/hnsw/hnsw_index.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  vectorforge::HnswConfig cfg;
  cfg.M = 4;
  cfg.ef_construction = 32;
  cfg.ef_search = 16;
  cfg.seed = 7;
  vectorforge::HnswIndex index(cfg);
  index.add(10, {1.0F, 0.0F});
  index.add(20, {0.0F, 1.0F});
  index.add(30, {-1.0F, 0.0F});
  index.add(40, {0.9F, 0.1F});
  if (index.size() != 4 || index.dimension() != 2)
    throw std::runtime_error("wrong size or dimension");
  const auto found = index.search({1.0F, 0.0F}, 2, 8);
  if (found.size() != 2 || found[0].id != 10 || found[1].id != 40)
    throw std::runtime_error("unexpected nearest neighbors");
  if (std::abs(found[0].score - 1.0F) > 1e-5F)
    throw std::runtime_error("cosine score incorrect");
  bool duplicate_rejected = false;
  try {
    index.add(10, {0.0F, 0.0F});
  } catch (const std::invalid_argument &) {
    duplicate_rejected = true;
  }
  if (!duplicate_rejected)
    throw std::runtime_error("duplicate ID accepted");
  bool dimension_rejected = false;
  try {
    index.add(50, {1.0F, 2.0F, 3.0F});
  } catch (const std::invalid_argument &) {
    dimension_rejected = true;
  }
  if (!dimension_rejected)
    throw std::runtime_error("dimension mismatch accepted");
  std::cout << "PASS: HNSW smoke test; top-k, score, size and validation.\n";
}
