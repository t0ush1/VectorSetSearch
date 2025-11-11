#pragma once

#include "index.h"
#include "multi_hnsw.h"

namespace vss {

class MultiHNSWIndex : public VSSIndex {
public:
    int M;
    int ef_construction;
    MultiHNSW* hnsw;

    MultiHNSWIndex(int dim, VSSSpace* space, int M, int ef_construction)
        : VSSIndex(dim, space), M(M), ef_construction(ef_construction) {}

    ~MultiHNSWIndex() { delete hnsw; }

    void build(const VSSDataset* base_dataset) {
        hnsw = new MultiHNSW(space, base_dataset->seq_num, M, ef_construction);
        for (int i = 0; i < base_dataset->seq_num; i++) {
            hnsw->add_point(base_dataset->seq_data[i], base_dataset->seq_len[i]);
        }
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        hnsw->ef = ef;
        auto result = hnsw->search_knn(q_data, q_len, k);
        std::priority_queue<std::pair<float, int>> final_result;
        while (!result.empty()) {
            final_result.emplace(result.top());
            result.pop();
        }
        return final_result;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"hops", hnsw->metric_hops},
            {"dist_comps", hnsw->metric_distance_computations},
        };
    }

    void reset_metrics() override {
        hnsw->metric_distance_computations = 0;
        hnsw->metric_hops = 0;
    }
};

} // namespace vss
