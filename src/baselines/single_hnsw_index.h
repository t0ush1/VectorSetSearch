#pragma once

#include "index.h"
#include "single_hnsw.h"

namespace vss {

class SingleHNSWIndex : public RerankIndex {
public:
    int M;
    int ef_construction;
    SingleHNSW<float>* hnsw;

    SingleHNSWIndex(int dim, VSSSpace* space, int M, int ef_construction)
        : RerankIndex(dim, space), M(M), ef_construction(ef_construction) {}

    ~SingleHNSWIndex() { delete hnsw; }

    void build_index() override {
        hnsw = new SingleHNSW<float>(space->space, vec_num, M, ef_construction);

        const float* vec = vec_data;
        for (size_t i = 0; i < vec_num; i++, vec += dim) {
            hnsw->add_point(vec, i);
        }
    }

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        hnsw->ef = q_k;
        std::unordered_set<int> candidates;

        const float* q_vec = q_data;
        for (int i = 0; i < q_len; i++, q_vec += dim) {
            auto res = hnsw->search_knn(q_vec, q_k);
            while (!res.empty()) {
                auto result = res.top();
                res.pop();
                candidates.insert(vec_to_seq[result.second]);
            }
        }

        return candidates;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        auto metrics = RerankIndex::get_metrics();
        metrics.push_back({"hops", hnsw->metric_hops});
        metrics.push_back({"dist_comps", hnsw->metric_distance_computations});
        return metrics;
    }

    void reset_metrics() override {
        RerankIndex::reset_metrics();
        hnsw->metric_hops = 0;
        hnsw->metric_distance_computations = 0;
    }
};

} // namespace vss
