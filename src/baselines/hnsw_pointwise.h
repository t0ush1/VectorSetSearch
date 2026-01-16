#pragma once

#include "hnsw.h"
#include "index.h"

namespace vss {

class HNSWPointwiseIndex : public RerankIndex {
public:
    std::vector<int> vec_to_set;

    int M;
    int ef_construction;
    HNSW<float>* hnsw;

    HNSWPointwiseIndex(int dim, VSSSpace* space, int M, int ef_construction)
        : RerankIndex(dim, space), M(M), ef_construction(ef_construction) {}

    ~HNSWPointwiseIndex() { delete hnsw; }

    void build_index() override {
        vec_to_set.reserve(vec_num);
        for (int i = 0; i < set_num; i++) {
            for (int j = 0; j < set_len[i]; j++) {
                vec_to_set.push_back(i);
            }
        }

        hnsw = new HNSW<float>(space->space, vec_num, M, ef_construction);

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
                candidates.insert(vec_to_set[result.second]);
            }
        }

        return candidates;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        auto metrics = RerankIndex::get_metrics();
        metrics.push_back({"hnsw_hops", hnsw->metric_hops});
        metrics.push_back({"hnsw_dist_comps", hnsw->metric_distance_computations});
        metrics.push_back({"tot_dist_comps", hnsw->metric_distance_computations + metric_rerank_dist_comps});
        return metrics;
    }

    void reset_metrics() override {
        RerankIndex::reset_metrics();
        hnsw->metric_hops = 0;
        hnsw->metric_distance_computations = 0;
    }
};

} // namespace vss
