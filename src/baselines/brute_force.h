#pragma once

#include "index.h"

namespace vss {

class BruteForceIndex : public RerankIndex {
public:
    BruteForceIndex(int dim, VSSSpace* space) : RerankIndex(dim, space) {}

    void build_index() override {}

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        std::unordered_set<int> candidates;
        for (int i = 0; i < set_num; i++) {
            candidates.insert(i);
        }
        return candidates;
    }
};

} // namespace vss
