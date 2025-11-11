#pragma once

#include <set>

#include "baselines/multi_hnsw.h"
#include "index.h"
#include "sample.h"

namespace vss {

class SemiHNSWIndex : public RerankIndex {
public:
    int M;
    int ef_construction;
    MultiHNSW* hnsw;

    int W;
    int C;

    size_t node_num;
    std::vector<int> node_to_set;

    std::default_random_engine sample_generator;

    SemiHNSWIndex(int dim, VSSSpace* space, int M, int ef_construction, int W = 2, int C = 5)
        : RerankIndex(dim, space), M(M), ef_construction(ef_construction), W(W), C(C) {
        sample_generator.seed(42);
    }

    ~SemiHNSWIndex() { delete hnsw; }

    std::set<std::set<int>> sampling(int len, int W, int C) {
        std::set<std::set<int>> subsets;
        size_t comb = combination(len, W);
        size_t gamma = std::min((size_t)C * len, comb);
        if (len < W) {
            std::set<int> subset;
            for (int j = 0; j < len; j++) {
                subset.insert(j);
            }
            subsets.insert(std::move(subset));
        } else if (comb < 10 * gamma) {
            auto all = all_subsets(len, W);
            std::shuffle(all.begin(), all.end(), sample_generator);
            subsets.insert(std::make_move_iterator(all.begin()), std::make_move_iterator(all.begin() + gamma));
        } else {
            subsets = sample_subsets(len, W, gamma, sample_generator);
        }
        return subsets;
    }

    void build_index() override {
        std::vector<std::set<std::set<int>>> samples;
        node_num = 0;
        for (int i = 0; i < seq_num; i++) {
            int len = seq_len[i];
            auto subsets = sampling(len, W, C);
            node_num += subsets.size();
            samples.push_back(std::move(subsets));
        }

        hnsw = new MultiHNSW(space, node_num, M, ef_construction);
        node_to_set.reserve(node_num);
        for (int i = 0; i < seq_num; i++) {
            const float* data = seq_data[i];
            for (const auto& subset : samples[i]) {
                int len = subset.size();
                float* temp_data = new float[len * dim];
                float* p = temp_data;
                for (int s : subset) {
                    memcpy(p, data + s * dim, space->data_size);
                    p += dim;
                }
                hnsw->add_point(temp_data, len);
                node_to_set.push_back(i);
                delete[] temp_data;
            }
        }
    }

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        hnsw->ef = q_k; // TODO 调参
        std::unordered_set<int> candidates;

        auto subsets = sampling(q_len, W, C);
        for (const auto& subset : subsets) {
            int len = subset.size();
            float* data = new float[len * dim];
            float* p = data;
            for (int s : subset) {
                memcpy(p, q_data + s * dim, space->data_size);
                p += dim;
            }

            auto result = hnsw->search_knn(data, len, q_k);
            while (!result.empty()) {
                candidates.insert(node_to_set[result.top().second]);
                result.pop();
            }

            delete[] data;
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
