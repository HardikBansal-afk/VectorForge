#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Record { long long id; std::vector<double> v; };
struct Result { double score; long long id; };

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out; std::stringstream ss(line); std::string field;
    while (std::getline(ss, field, ',')) out.push_back(field);
    if (!line.empty() && line.back() == ',') out.emplace_back();
    return out;
}

std::vector<Record> read_csv(const std::string& path, std::size_t& dim) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Empty CSV: " + path);
    const auto header = split_csv(line);
    if (header.size() < 2 || header[0] != "id")
        throw std::runtime_error("Expected header id,v0,v1,... in " + path);
    dim = header.size() - 1;
    std::vector<Record> rows;
    std::size_t line_no = 1;
    while (std::getline(in, line)) {
        ++line_no; if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() != dim + 1)
            throw std::runtime_error("Wrong column count at " + path + ":" + std::to_string(line_no));
        Record r; r.v.reserve(dim);
        try {
            r.id = std::stoll(fields[0]);
            for (std::size_t i = 1; i < fields.size(); ++i) {
                double x = std::stod(fields[i]);
                if (!std::isfinite(x)) throw std::runtime_error("non-finite value");
                r.v.push_back(x);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid value at " + path + ":" + std::to_string(line_no) + " (" + e.what() + ")");
        }
        rows.push_back(std::move(r));
    }
    if (rows.empty()) throw std::runtime_error("No vectors in " + path);
    return rows;
}

double cosine(const std::vector<double>& a, const std::vector<double>& b) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i];
    }
    if (aa == 0.0 || bb == 0.0) return 0.0;
    return dot / (std::sqrt(aa) * std::sqrt(bb));
}

std::vector<Result> exact_top_k(const Record& q, const std::vector<Record>& corpus, std::size_t k) {
    // Heap top is the worst retained candidate: lowest score, then largest ID.
    auto worse_first = [](const Result& a, const Result& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.id < b.id;
    };
    std::priority_queue<Result, std::vector<Result>, decltype(worse_first)> heap(worse_first);
    for (const auto& row : corpus) {
        Result r{cosine(q.v, row.v), row.id};
        if (heap.size() < k) heap.push(r);
        else {
            const auto& worst = heap.top();
            if (r.score > worst.score || (r.score == worst.score && r.id < worst.id)) {
                heap.pop(); heap.push(r);
            }
        }
    }
    std::vector<Result> out; out.reserve(heap.size());
    while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
    std::sort(out.begin(), out.end(), [](const Result& a, const Result& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.id < b.id;
    });
    return out;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <vectors.csv> <queries.csv> <k> <output.csv>\n";
        return 2;
    }
    try {
        const long k_arg = std::stol(argv[3]);
        if (k_arg <= 0) throw std::runtime_error("k must be a positive integer");
        const auto k = static_cast<std::size_t>(k_arg);
        std::size_t corpus_dim = 0, query_dim = 0;
        const auto corpus = read_csv(argv[1], corpus_dim);
        const auto queries = read_csv(argv[2], query_dim);
        if (corpus_dim != query_dim) throw std::runtime_error("Corpus/query dimensions differ");
        std::ofstream out(argv[4]);
        if (!out) throw std::runtime_error("Cannot open output: " + std::string(argv[4]));
        out << "query_id,rank,neighbor_id,cosine_similarity\n";
        for (const auto& q : queries) {
            const auto results = exact_top_k(q, corpus, std::min(k, corpus.size()));
            for (std::size_t i = 0; i < results.size(); ++i)
                out << q.id << ',' << i + 1 << ',' << results[i].id << ',' << results[i].score << '\n';
        }
        std::cout << "Loaded " << corpus.size() << " vectors (dim=" << corpus_dim << "); processed "
                  << queries.size() << " queries; top-k=" << std::min(k, corpus.size()) << "\n";
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
