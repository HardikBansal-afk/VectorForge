CXX := clang++
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
TARGET := build/vector_search

.PHONY: all clean test
all: $(TARGET)
$(TARGET): cpp/vector_search.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@
clean:
	rm -rf build

test: all
	python3 python/generate_data.py --rows 1000 --dim 32 --queries 5 --seed 42
	./$(TARGET) data/vectors.csv data/queries.csv 5 results/topk.csv
	python3 python/verify_results.py --vectors data/vectors.csv --queries data/queries.csv --results results/topk.csv --k 5
HNSW_TEST := build/hnsw_smoke
HNSW_SRC := cpp/hnsw/hnsw_index.cpp

$(HNSW_TEST): tests/hnsw_smoke.cpp $(HNSW_SRC) cpp/hnsw/hnsw_index.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Icpp tests/hnsw_smoke.cpp $(HNSW_SRC) -o $(HNSW_TEST)

.PHONY: hnsw-test

hnsw-test: $(HNSW_TEST)
	./$(HNSW_TEST)
HNSW_BENCHMARK := build/hnsw_benchmark

$(HNSW_BENCHMARK): cpp/hnsw_benchmark.cpp cpp/hnsw/hnsw_index.cpp cpp/hnsw/hnsw_index.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Icpp cpp/hnsw_benchmark.cpp cpp/hnsw/hnsw_index.cpp -o $(HNSW_BENCHMARK)

.PHONY: hnsw-benchmark

hnsw-benchmark: $(HNSW_BENCHMARK)