#include "hnsw_index.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace vectorforge {
namespace {
double vector_norm(const std::vector<float> &values) {
  double sum = 0.0;

  for (const float value : values) {
    const double v = static_cast<double>(value);
    sum += v * v;
  }

  return std::sqrt(sum);
}
struct Scored {
  float score;
  std::size_t index;
};
struct BestFirst {
  bool operator()(const Scored &a, const Scored &b) const {
    if (a.score != b.score)
      return a.score < b.score;
    return a.index > b.index;
  }
};
struct WorstFirst {
  bool operator()(const Scored &a, const Scored &b) const {
    if (a.score != b.score)
      return a.score > b.score;
    return a.index < b.index;
  }
};
} // namespace

HnswIndex::HnswIndex(HnswConfig config) : config_(config), rng_(config.seed) {
  if (config_.M < 2)
    throw std::invalid_argument("M must be >= 2");
  if (config_.ef_construction < config_.M)
    throw std::invalid_argument("ef_construction must be >= M");
  if (config_.ef_search == 0)
    throw std::invalid_argument("ef_search must be > 0");
}
std::size_t HnswIndex::size() const noexcept { return nodes_.size(); }
std::size_t HnswIndex::dimension() const noexcept { return dimension_; }

float HnswIndex::cosine_with_norms(const std::vector<float> &a, double norm_a,
                                   const std::vector<float> &b,
                                   double norm_b) const {
  if (a.size() != b.size()) {
    throw std::invalid_argument("cosine dimension mismatch");
  }

  if (norm_a == 0.0 || norm_b == 0.0) {
    return 0.0F;
  }

  double dot = 0.0;

  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }

  return static_cast<float>(dot / (norm_a * norm_b));
}

float HnswIndex::cosine(const std::vector<float> &a,
                        const std::vector<float> &b) const {
  const double norm_a = static_cast<double>(vector_norm(a));
  const double norm_b = static_cast<double>(vector_norm(b));

  return cosine_with_norms(a, norm_a, b, norm_b);
}
int HnswIndex::random_level() {
  const double u = std::max(std::generate_canonical<double, 53>(rng_),
                            std::numeric_limits<double>::min());

  return static_cast<int>(-std::log(u) /
                          std::log(static_cast<double>(config_.M)));
}

std::size_t HnswIndex::greedy(const std::vector<float> &query,
                              std::size_t entry, int layer) const {
  auto current = entry;
  auto score = cosine(query, nodes_[current].vector);
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto n : nodes_[current].neighbors[static_cast<std::size_t>(layer)]) {
      const auto candidate = cosine(query, nodes_[n].vector);
      if (candidate > score ||
          (candidate == score && nodes_[n].id < nodes_[current].id)) {
        current = n;
        score = candidate;
        changed = true;
      }
    }
  }
  return current;
}
std::vector<HnswIndex::ScoredCandidate>
HnswIndex::search_layer(const std::vector<float> &query,
                        const std::vector<std::size_t> &entries, std::size_t ef,
                        int layer) const {
  std::priority_queue<Scored, std::vector<Scored>, BestFirst> candidates;
  std::priority_queue<Scored, std::vector<Scored>, WorstFirst> best;

  std::vector<unsigned char> visited(nodes_.size(), 0);

  for (const auto entry : entries) {
    if (entry >= nodes_.size() || layer < 0 ||
        static_cast<std::size_t>(layer) >= nodes_[entry].neighbors.size()) {
      continue;
    }

    if (visited[entry]) {
      continue;
    }

    visited[entry] = 1;

    const float score = cosine(query, nodes_[entry].vector);

    candidates.push({score, entry});
    best.push({score, entry});
  }

  while (!candidates.empty()) {
    const auto current = candidates.top();
    candidates.pop();

    if (best.size() >= ef && current.score < best.top().score) {
      break;
    }

    for (const auto neighbor :
         nodes_[current.index].neighbors[static_cast<std::size_t>(layer)]) {
      if (visited[neighbor]) {
        continue;
      }

      visited[neighbor] = 1;

      const float score = cosine(query, nodes_[neighbor].vector);

      if (best.size() < ef || score > best.top().score ||
          (score == best.top().score &&
           nodes_[neighbor].id < nodes_[best.top().index].id)) {
        candidates.push({score, neighbor});
        best.push({score, neighbor});

        if (best.size() > ef) {
          best.pop();
        }
      }
    }
  }

  std::vector<Scored> scored_results;
  scored_results.reserve(best.size());

  while (!best.empty()) {
    scored_results.push_back(best.top());
    best.pop();
  }

  std::sort(scored_results.begin(), scored_results.end(),
            [this](const Scored &a, const Scored &b) {
              if (a.score != b.score) {
                return a.score > b.score;
              }

              return nodes_[a.index].id < nodes_[b.index].id;
            });

  std::vector<ScoredCandidate> results;
  results.reserve(scored_results.size());

  for (const auto &entry : scored_results) {
    results.push_back({entry.score, entry.index});
  }

  return results;
}
std::vector<std::size_t>
HnswIndex::select_neighbors(const std::vector<float> &query,
                            const std::vector<std::size_t> &candidates,
                            std::size_t limit) const {
  if (limit == 0 || candidates.empty()) {
    return {};
  }

  struct RankedCandidate {
    std::size_t index;
    float score;
  };

  std::vector<RankedCandidate> ranked;
  ranked.reserve(candidates.size());

  std::unordered_set<std::size_t> seen;
  seen.reserve(candidates.size());

  // Deduplicate candidates and compute each query similarity once.
  for (const auto candidate : candidates) {
    if (candidate >= nodes_.size()) {
      continue;
    }

    if (seen.insert(candidate).second) {
      ranked.push_back({candidate, cosine(query, nodes_[candidate].vector)});
    }
  }

  if (ranked.empty()) {
    return {};
  }

  // Sort using cached scores.
  std::sort(ranked.begin(), ranked.end(),
            [this](const RankedCandidate &a, const RankedCandidate &b) {
              if (a.score != b.score) {
                return a.score > b.score;
              }

              return nodes_[a.index].id < nodes_[b.index].id;
            });

  const std::size_t target = std::min(limit, ranked.size());

  std::vector<std::size_t> selected;
  selected.reserve(target);

  // Preserve the HNSW diversity heuristic.
  for (const auto &candidate : ranked) {
    bool diverse = true;

    for (const auto chosen : selected) {
      const float neighbor_similarity = cosine_with_norms(
          nodes_[candidate.index].vector, nodes_[candidate.index].norm,
          nodes_[chosen].vector, nodes_[chosen].norm);

      if (neighbor_similarity >= candidate.score) {
        diverse = false;
        break;
      }
    }

    if (diverse) {
      selected.push_back(candidate.index);

      if (selected.size() == target) {
        break;
      }
    }
  }

  // Preserve fallback behavior if diversity selection rejects candidates.
  if (selected.size() < target) {
    std::unordered_set<std::size_t> selected_set(selected.begin(),
                                                 selected.end());

    for (const auto &candidate : ranked) {
      if (selected.size() >= target) {
        break;
      }

      if (selected_set.insert(candidate.index).second) {
        selected.push_back(candidate.index);
      }
    }
  }

  return selected;
}
void HnswIndex::connect(std::size_t fresh,
                        const std::vector<std::size_t> &candidates, int layer) {
  const auto l = static_cast<std::size_t>(layer);
  const std::size_t cap = layer == 0 ? 2 * config_.M : config_.M;
  auto chosen = select_neighbors(nodes_[fresh].vector, candidates, cap);
  nodes_[fresh].neighbors[l] = chosen;
  for (auto n : chosen) {
    auto &reverse = nodes_[n].neighbors[l];
    reverse.push_back(fresh);
    if (reverse.size() > cap)
      reverse = select_neighbors(nodes_[n].vector, reverse, cap);
  }
}

void HnswIndex::add(std::int64_t id, const std::vector<float> &vector) {
  if (vector.empty())
    throw std::invalid_argument("vector must not be empty");
  for (float x : vector)
    if (!std::isfinite(x))
      throw std::invalid_argument("vector values must be finite");
  if (dimension_ != 0 && vector.size() != dimension_)
    throw std::invalid_argument("vector dimension mismatch");

  if (ids_.find(id) != ids_.end())
    throw std::invalid_argument("duplicate vector ID");

  Node fresh;
  fresh.id = id;
  fresh.vector = vector;
  fresh.norm = vector_norm(vector);
  fresh.level = random_level();
  fresh.neighbors.resize(static_cast<std::size_t>(fresh.level + 1));
  const auto index = nodes_.size();
  ids_.insert(id);
  if (dimension_ == 0)
    dimension_ = vector.size();
  if (nodes_.empty()) {
    nodes_.push_back(std::move(fresh));
    entry_point_ = 0;
    max_level_ = nodes_[0].level;
    return;
  }

  nodes_.push_back(std::move(fresh));
  auto entry = entry_point_;
  for (int layer = max_level_; layer > nodes_[index].level; --layer)
    entry = greedy(vector, entry, layer);

  const int stop = std::min(nodes_[index].level, max_level_);
  for (int layer = stop; layer >= 0; --layer) {
    auto scored_candidates =
        search_layer(vector, {entry}, config_.ef_construction, layer);

    std::vector<std::size_t> candidates;
    candidates.reserve(scored_candidates.size());

    for (const auto &candidate : scored_candidates) {
      candidates.push_back(candidate.index);
    }

    connect(index, candidates, layer);
    if (!nodes_[index].neighbors[static_cast<std::size_t>(layer)].empty())
      entry = nodes_[index].neighbors[static_cast<std::size_t>(layer)][0];
  }
  if (nodes_[index].level > max_level_) {
    entry_point_ = index;
    max_level_ = nodes_[index].level;
  }
}

std::vector<SearchResult> HnswIndex::search(const std::vector<float> &query,
                                            std::size_t k,
                                            std::size_t ef) const {
  if (k == 0 || nodes_.empty())
    return {};
  if (query.size() != dimension_)
    throw std::invalid_argument("query dimension mismatch");
  for (float x : query)
    if (!std::isfinite(x))
      throw std::invalid_argument("query values must be finite");
  if (ef == 0)
    ef = config_.ef_search;
  ef = std::max(ef, k);
  auto entry = entry_point_;
  for (int layer = max_level_; layer > 0; --layer)
    entry = greedy(query, entry, layer);
  const auto candidates = search_layer(query, {entry}, ef, 0);

  std::vector<SearchResult> result;
  result.reserve(std::min(k, candidates.size()));

  for (std::size_t i = 0; i < std::min(k, candidates.size()); ++i) {
    const auto &candidate = candidates[i];

    result.push_back({nodes_[candidate.index].id, candidate.score});
  }
  return result;
}
} // namespace vectorforge
