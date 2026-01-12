#pragma once
#include <set>

#include "hnsw.h"
#include "index.h"

namespace vss {

class DualHNSWIndex : public VSSIndex {
public:
    std::vector<int> vec_to_base;

    int M;
    int ef_construction;
    HNSW<float>* hnsw;
    std::vector<HNSW<float>*> inner_hnsws;

    long metric_cand_num;
    long metric_rerank_dist_comps;

    DualHNSWIndex(int dim, VSSSpace* space, int M, int ef_construction)
        : VSSIndex(dim, space), M(M), ef_construction(ef_construction) {}

    ~DualHNSWIndex() {
        delete hnsw;
        for (auto inner_hnsw : inner_hnsws) {
            delete inner_hnsw;
        }
    }

    void build(const VSSDataset* base_dataset) override {
        vec_to_base.reserve(base_dataset->size);
        for (int i = 0; i < base_dataset->set_num; i++) {
            for (int j = 0; j < base_dataset->set_len[i]; j++) {
                vec_to_base.push_back(i);
            }
        }

        hnsw = new HNSW<float>(space->space, base_dataset->size, M, ef_construction);
        const float* data = base_dataset->data;
        for (int i = 0; i < base_dataset->size; i++, data += dim) {
            hnsw->add_point(data, i);
        }

        for (int i = 0; i < base_dataset->set_num; i++) {
            HNSW<float>* inner_hnsw =
                new HNSW<float>(space->space, base_dataset->set_len[i], M, ef_construction);
            const float* inner_data = base_dataset->set_data[i];
            for (int j = 0; j < base_dataset->set_len[i]; j++, inner_data += dim) {
                inner_hnsw->add_point(inner_data, j);
            }
            inner_hnsws.push_back(inner_hnsw);
        }
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        std::unordered_set<int> candidates;
        hnsw->ef = ef;
        for (int q = 0; q < q_len; q++) {
            auto r = hnsw->search_knn(q_data + q * dim, ef);
            while (!r.empty()) {
                candidates.insert(vec_to_base[r.top().second]);
                r.pop();
            }
        }

        std::priority_queue<std::pair<float, int>> result;
        for (int B : candidates) {
            HNSW<float>* inner_hnsw = inner_hnsws[B];
            inner_hnsw->ef = 10;
            inner_hnsw->metric_distance_computations = 0;
            
            float dist = 0;
            for (int q = 0; q < q_len; q++) {
                auto r = inner_hnsw->search_knn(q_data + q * dim, 1);
                dist += r.top().first;
            }
            result.emplace(dist, B);
            if (result.size() > k) {
                result.pop();
            }

            metric_cand_num++;
            metric_rerank_dist_comps += inner_hnsw->metric_distance_computations;
        }

        return result;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"hops", hnsw->metric_hops},
            {"dist_comps", hnsw->metric_distance_computations + metric_rerank_dist_comps},
            {"cand_num", metric_cand_num},
        };
    }

    void reset_metrics() override {
        metric_cand_num = 0;
        metric_rerank_dist_comps = 0;
        hnsw->metric_hops = 0;
        hnsw->metric_distance_computations = 0;
    }
};

} // namespace vss
