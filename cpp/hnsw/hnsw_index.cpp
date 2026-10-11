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

float HnswIndex::cosine(const std::vector<float> &a,
                        const std::vector<float> &b) const {
  float dot = 0.0F, na = 0.0F, nb = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na == 0.0F || nb == 0.0F)
    return 0.0F;
  return dot / (std::sqrt(na) * std::sqrt(nb));
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

std::vector<std::size_t>
HnswIndex::search_layer(const std::vector<float> &query,
                        const std::vector<std::size_t> &entries, std::size_t ef,
                        int layer) const {
  std::priority_queue<Scored, std::vector<Scored>, BestFirst> candidates;
  std::priority_queue<Scored, std::vector<Scored>, WorstFirst> best;
  std::vector<unsigned char> visited(nodes_.size(), 0);
  for (auto e : entries) {
    if (visited[e])
      continue;
    visited[e] = 1;
    const float s = cosine(query, nodes_[e].vector);
    candidates.push({s, e});
    best.push({s, e});
  }
  while (!candidates.empty()) {
    const auto current = candidates.top();
    candidates.pop();
    if (best.size() >= ef && current.score < best.top().score)
      break;
    for (auto n :
         nodes_[current.index].neighbors[static_cast<std::size_t>(layer)]) {
      if (visited[n])
        continue;
      visited[n] = 1;
      const float s = cosine(query, nodes_[n].vector);
      if (best.size() < ef || s > best.top().score ||
          (s == best.top().score &&
           nodes_[n].id < nodes_[best.top().index].id)) {
        candidates.push({s, n});
        best.push({s, n});
        if (best.size() > ef)
          best.pop();
      }
    }
  }
  std::vector<std::size_t> out;
  out.reserve(best.size());
  while (!best.empty()) {
    out.push_back(best.top().index);
    best.pop();
  }
  std::sort(out.begin(), out.end(), [this, &query](auto a, auto b) {
    const auto sa = cosine(query, nodes_[a].vector);
    const auto sb = cosine(query, nodes_[b].vector);
    if (sa != sb)
      return sa > sb;
    return nodes_[a].id < nodes_[b].id;
  });
  return out;
}

std::vector<std::size_t>
HnswIndex::select_neighbors(const std::vector<float> &query,
                            const std::vector<std::size_t> &candidates,
                            std::size_t limit) const {
  auto out = candidates;
  std::sort(out.begin(), out.end(), [this, &query](auto a, auto b) {
    const auto sa = cosine(query, nodes_[a].vector);
    const auto sb = cosine(query, nodes_[b].vector);
    if (sa != sb)
      return sa > sb;
    return nodes_[a].id < nodes_[b].id;
  });
  std::unordered_set<std::size_t> seen;
  out.erase(std::remove_if(out.begin(), out.end(),
                           [&seen](auto x) { return !seen.insert(x).second; }),
            out.end());
  if (out.size() > limit)
    out.resize(limit);
  return out;
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
    auto candidates =
        search_layer(vector, {entry}, config_.ef_construction, layer);
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
  for (std::size_t i = 0; i < std::min(k, candidates.size()); ++i) {
    const auto idx = candidates[i];
    result.push_back({nodes_[idx].id, cosine(query, nodes_[idx].vector)});
  }
  return result;
}
} // namespace vectorforge
