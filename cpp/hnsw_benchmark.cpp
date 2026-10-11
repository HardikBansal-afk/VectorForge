#include "hnsw/hnsw_index.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Record {
  std::int64_t id;
  std::vector<float> values;
};

struct Result {
  std::int64_t id;
  float score;
};

struct Candidate {
  std::int64_t id;
  float score;
};

struct SearchMeasurement {
  std::vector<Result> results;
  double latency_ms;
};

static std::vector<std::string> split_csv(const std::string &line) {
  std::stringstream stream(line);
  std::vector<std::string> fields;
  std::string field;

  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }

  if (!line.empty() && line.back() == ',') {
    fields.emplace_back();
  }

  return fields;
}

static std::vector<Record> load_csv(const std::string &path,
                                    std::size_t &dimension) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("Empty CSV: " + path);
  }
  const auto header = split_csv(line);
  if (header.size() < 2 || header[0] != "id") {
    throw std::runtime_error("Expected header: id,v0,v1,...");
  }
  dimension = header.size() - 1;
  std::vector<Record> records;
  std::size_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;

    if (line.empty()) {
      continue;
    }
    const auto fields = split_csv(line);
    if (fields.size() != dimension + 1) {
      throw std::runtime_error("Invalid column count at line " +
                               std::to_string(line_number));
    }
    Record record;
    try {
      record.id = std::stoll(fields[0]);
      record.values.reserve(dimension);
      for (std::size_t i = 1; i < fields.size(); ++i) {
        const float value = std::stof(fields[i]);
        if (!std::isfinite(value)) {
          throw std::runtime_error("Non-finite vector value");
        }
        record.values.push_back(value);
      }
    } catch (const std::exception &error) {
      throw std::runtime_error("Invalid CSV data at line " +
                               std::to_string(line_number) + ": " +
                               error.what());
    }
    records.push_back(std::move(record));
  }
  if (records.empty()) {
    throw std::runtime_error("Dataset contains no vectors");
  }
  return records;
}
static float cosine_similarity(const std::vector<float> &a,
                               const std::vector<float> &b) {
  float dot = 0.0F;
  float norm_a = 0.0F;
  float norm_b = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  if (norm_a == 0.0F || norm_b == 0.0F) {
    return 0.0F;
  }
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

static std::vector<Result> exact_search(const std::vector<float> &query,
                                        const std::vector<Record> &corpus,
                                        std::size_t k) {
  std::vector<Result> candidates;
  candidates.reserve(corpus.size());
  for (const auto &record : corpus) {
    candidates.push_back({record.id, cosine_similarity(query, record.values)});
  }
  const auto better = [](const Result &a, const Result &b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.id < b.id;
  };
  if (k < candidates.size()) {
    const auto middle = candidates.begin() +
                        static_cast<std::vector<Result>::difference_type>(k);
    std::partial_sort(candidates.begin(), middle, candidates.end(), better);
    candidates.resize(k);
  } else {
    std::sort(candidates.begin(), candidates.end(), better);
  }
  return candidates;
}

static double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  if (p < 0.0 || p > 1.0) {
    throw std::invalid_argument("Percentile must be between 0 and 1");
  }
  std::sort(values.begin(), values.end());
  const double sample_count = static_cast<double>(values.size());
  const std::size_t index =
      static_cast<std::size_t>(std::ceil(p * sample_count));
  return values[std::max<std::size_t>(1, index) - 1];
}

static double
recall_at_k(const std::vector<Result> &exact,
            const std::vector<vectorforge::SearchResult> &approximate,
            std::size_t k) {
  const std::size_t count = std::min(k, exact.size());

  if (count == 0) {
    return 1.0;
  }
  std::unordered_set<std::int64_t> expected;
  expected.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    expected.insert(exact[i].id);
  }
  std::size_t hits = 0;
  for (const auto &result : approximate) {
    if (expected.find(result.id) != expected.end()) {
      ++hits;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(count);
}

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0] << " <vectors.csv> <queries.csv> <k>"
              << " <M> <efConstruction> <efSearch>"
              << " <output.csv>\n";

    return 2;
  }

  try {
    const auto parse_positive = [](const char *value,
                                   const std::string &name) -> std::size_t {
      const long long parsed = std::stoll(value);

      if (parsed <= 0) {
        throw std::runtime_error(name + " must be positive");
      }

      return static_cast<std::size_t>(parsed);
    };

    const std::size_t k = parse_positive(argv[3], "k");

    vectorforge::HnswConfig config;

    config.M = parse_positive(argv[4], "M");

    config.ef_construction = parse_positive(argv[5], "efConstruction");

    config.ef_search = parse_positive(argv[6], "efSearch");

    config.seed = 42;

    if (config.ef_construction < config.M) {
      throw std::runtime_error("efConstruction must be >= M");
    }

    if (config.ef_search < k) {
      throw std::runtime_error("efSearch must be >= k");
    }

    std::size_t corpus_dimension = 0;
    std::size_t query_dimension = 0;

    const auto corpus = load_csv(argv[1], corpus_dimension);

    const auto queries = load_csv(argv[2], query_dimension);

    // Validate that at least one query exists.
    if (queries.empty()) {
      throw std::runtime_error("Query dataset contains no queries");
    }

    if (corpus_dimension != query_dimension) {
      throw std::runtime_error("Corpus and query dimensions differ");
    }

    if (k > corpus.size()) {
      throw std::runtime_error("k cannot exceed corpus size");
    }

    std::unordered_set<std::int64_t> corpus_ids;
    corpus_ids.reserve(corpus.size());

    for (const auto &record : corpus) {
      if (!corpus_ids.insert(record.id).second) {
        throw std::runtime_error("Duplicate corpus ID");
      }
    }

    for (const auto &query : queries) {
      if (query.values.size() != corpus_dimension) {
        throw std::runtime_error("Query dimension mismatch");
      }
    }

    // Establish exact ground truth.
    std::vector<std::vector<Result>> ground_truth;
    ground_truth.reserve(queries.size());

    std::vector<double> exact_latencies;
    exact_latencies.reserve(queries.size());

    for (const auto &query : queries) {
      const auto start = Clock::now();

      auto results = exact_search(query.values, corpus, k);

      const auto end = Clock::now();

      const double elapsed =
          std::chrono::duration<double, std::milli>(end - start).count();

      exact_latencies.push_back(elapsed);

      ground_truth.push_back(std::move(results));
    }

    // Build the HNSW index.
    vectorforge::HnswIndex index(config);

    const auto build_start = Clock::now();

    for (const auto &record : corpus) {
      index.add(record.id, record.values);
    }

    const auto build_end = Clock::now();

    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start)
            .count();

    // Measure HNSW queries.
    std::vector<double> hnsw_latencies;
    hnsw_latencies.reserve(queries.size());

    double recall_sum = 0.0;

    for (std::size_t i = 0; i < queries.size(); ++i) {

      const auto start = Clock::now();

      const auto approximate =
          index.search(queries[i].values, k, config.ef_search);

      const auto end = Clock::now();

      const double elapsed =
          std::chrono::duration<double, std::milli>(end - start).count();

      hnsw_latencies.push_back(elapsed);

      recall_sum += recall_at_k(ground_truth[i], approximate, k);
    }

    const double exact_p50 = percentile(exact_latencies, 0.50);

    const double exact_p95 = percentile(exact_latencies, 0.95);

    const double hnsw_p50 = percentile(hnsw_latencies, 0.50);

    const double hnsw_p95 = percentile(hnsw_latencies, 0.95);

    const double exact_mean =
        std::accumulate(exact_latencies.begin(), exact_latencies.end(), 0.0) /
        static_cast<double>(exact_latencies.size());

    const double hnsw_mean =
        std::accumulate(hnsw_latencies.begin(), hnsw_latencies.end(), 0.0) /
        static_cast<double>(hnsw_latencies.size());

    const double recall = recall_sum / static_cast<double>(queries.size());

    // Estimate vector and adjacency storage.
    // Excludes allocator overhead, IDs, capacities, and other metadata.
    std::size_t edge_count = 0;

    for (const auto &record : corpus) {
      (void)record;
      // The public API does not expose graph edges yet.
    }

    const std::size_t vector_bytes =
        corpus.size() * corpus_dimension * sizeof(float);

    const std::size_t graph_bytes_estimate =
        corpus.size() * 2 * config.M * sizeof(std::size_t);

    const std::size_t storage_estimate = vector_bytes + graph_bytes_estimate;

    (void)edge_count;

    std::ofstream output(argv[7]);

    if (!output) {
      throw std::runtime_error("Cannot create output CSV");
    }

    output << "corpus_size,dimension,queries,k,M,"
           << "efConstruction,efSearch,"
           << "build_ms,exact_p50_ms,exact_p95_ms,"
           << "hnsw_p50_ms,hnsw_p95_ms,"
           << "exact_mean_ms,hnsw_mean_ms,"
           << "exact_qps,hnsw_qps,recall_at_k,"
           << "estimated_vector_graph_bytes\n";

    output << std::fixed << std::setprecision(6);

    output << corpus.size() << ',' << corpus_dimension << ',' << queries.size()
           << ',' << k << ',' << config.M << ',' << config.ef_construction
           << ',' << config.ef_search << ',' << build_ms << ',' << exact_p50
           << ',' << exact_p95 << ',' << hnsw_p50 << ',' << hnsw_p95 << ','
           << exact_mean << ',' << hnsw_mean << ',' << 1000.0 / exact_mean
           << ',' << 1000.0 / hnsw_mean << ',' << recall << ','
           << storage_estimate << '\n';

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\n=== VectorForge Benchmark ===\n"
              << "Corpus: " << corpus.size() << " vectors x "
              << corpus_dimension << " dimensions\n"
              << "Queries: " << queries.size() << ", k=" << k << '\n'
              << "HNSW: M=" << config.M
              << ", efConstruction=" << config.ef_construction
              << ", efSearch=" << config.ef_search << "\n\n"

              << "Build time: " << build_ms << " ms\n\n"

              << "Exact p50: " << exact_p50 << " ms\n"

              << "Exact p95: " << exact_p95 << " ms\n"

              << "Exact mean: " << exact_mean << " ms\n\n"

              << "HNSW p50: " << hnsw_p50 << " ms\n"

              << "HNSW p95: " << hnsw_p95 << " ms\n"

              << "HNSW mean: " << hnsw_mean << " ms\n\n"

              << "Recall@" << k << ": " << recall * 100.0 << "%\n"

              << "Estimated vector + graph storage: "
              << static_cast<double>(storage_estimate) / (1024.0 * 1024.0)
              << " MiB\n"

              << "CSV written to: " << argv[7] << '\n';

  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';

    return 1;
  }

  return 0;
}