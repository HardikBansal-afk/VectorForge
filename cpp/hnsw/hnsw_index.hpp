#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

namespace vectorforge {
struct SearchResult {
  std::int64_t id{};
  float score{};
};
struct HnswConfig {
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  std::size_t ef_search = 64;
  std::uint32_t seed = 42;
};
class HnswIndex {
public:
  explicit HnswIndex(HnswConfig config = {});
  void add(std::int64_t id, const std::vector<float> &vector);
  [[nodiscard]] std::vector<SearchResult>
  search(const std::vector<float> &query, std::size_t k,
         std::size_t ef_search = 0) const;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t dimension() const noexcept;

private:
  struct Node {
    std::int64_t id{};
    std::vector<float> vector;
    double norm = 0.0;
    std::vector<std::vector<std::size_t>> neighbors;
    int level = 0;
  };
  struct ScoredCandidate {
    float score;
    std::size_t index;
  };
  HnswConfig config_;
  std::unordered_set<std::int64_t> ids_;
  std::vector<Node> nodes_;
  std::size_t dimension_ = 0;
  std::size_t entry_point_ = 0;
  int max_level_ = -1;
  std::mt19937 rng_;
  [[nodiscard]] float cosine(const std::vector<float> &a,
                             const std::vector<float> &b) const;
  [[nodiscard]] float cosine_with_norms(const std::vector<float> &a,
                                        double norm_a,
                                        const std::vector<float> &b,
                                        double norm_b) const;
  [[nodiscard]] int random_level();
  [[nodiscard]] std::size_t greedy(const std::vector<float> &query,
                                   std::size_t entry, int layer) const;
  [[nodiscard]] std::vector<ScoredCandidate>
  search_layer(const std::vector<float> &query,
               const std::vector<std::size_t> &entries, std::size_t ef,
               int layer) const;
  [[nodiscard]] std::vector<std::size_t>
  select_neighbors(const std::vector<float> &query,
                   const std::vector<std::size_t> &candidates,
                   std::size_t limit) const;
  void connect(std::size_t new_index,
               const std::vector<std::size_t> &candidates, int layer);
};
} // namespace vectorforge
