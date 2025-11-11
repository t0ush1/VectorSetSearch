#pragma once

#include <hnswlib/hnswlib.h>

#include "index.h"

namespace vss {

class HNSWPointwiseIndex : public RerankIndex {
public:
    int M;
    int ef_construction;
    hnswlib::HierarchicalNSW<float>* hnsw;

    HNSWPointwiseIndex(int dim, VSSSpace* space, int M, int ef_construction)
        : RerankIndex(dim, space), M(M), ef_construction(ef_construction) {}

    ~HNSWPointwiseIndex() { delete hnsw; }

    void build_index() override {
        hnsw = new hnswlib::HierarchicalNSW<float>(space->space, vec_num, M, ef_construction);

        const float* vec = vec_data;
        for (size_t i = 0; i < vec_num; i++, vec += dim) {
            hnsw->addPoint(vec, i);
        }
    }

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        hnsw->ef_ = q_k;
        std::unordered_set<int> candidates;

        const float* q_vec = q_data;
        for (int i = 0; i < q_len; i++, q_vec += dim) {
            auto res = hnsw->searchKnn(q_vec, q_k);
            while (!res.empty()) {
                auto result = res.top();
                res.pop();
                candidates.insert(vec_to_seq[result.second]);
            }
        }

        return candidates;
    }
};

} // namespace vss
