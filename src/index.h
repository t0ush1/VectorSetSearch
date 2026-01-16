#pragma once
#include <queue>

#include "dataset.h"
#include "space.h"

namespace vss {

class VSSIndex {
public:
    int dim;
    VSSSpace* space;

    VSSIndex(int dim, VSSSpace* space) : dim(dim), space(space) {}

    virtual void build(const VSSDataset* base_dataset) = 0;
    virtual std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) = 0;
    virtual void prepare(const float* q_data, int q_len, int k, int ef) {};
    virtual std::vector<std::pair<std::string, long>> get_stats() { return {}; }
    virtual std::vector<std::pair<std::string, long>> get_metrics() { return {}; }
    virtual void reset_metrics() {};
};

class RerankIndex : public VSSIndex {
public:
    size_t vec_num;
    float* vec_data;

    size_t set_num;
    std::vector<const float*> set_data;
    std::vector<int> set_len;

    long metric_cand_num;
    long metric_cand_gen_time;
    long metric_rerank_time;
    long metric_rerank_dist_comps;

    virtual void build_index() = 0;
    virtual std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) = 0;

    RerankIndex(int dim, VSSSpace* space) : VSSIndex(dim, space) {}

    ~RerankIndex() { delete[] vec_data; }

    void build(const VSSDataset* base_dataset) override {
        vec_num = base_dataset->size;
        vec_data = new float[vec_num * dim];
        memcpy(vec_data, base_dataset->data, vec_num * space->data_size);

        set_num = base_dataset->set_num;
        set_len = base_dataset->set_len;
        set_data.resize(set_num);
        set_data[0] = vec_data;
        for (int i = 1; i < set_num; i++) {
            set_data[i] = set_data[i - 1] + set_len[i - 1] * dim;
        }

        build_index();
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        auto begin = std::chrono::high_resolution_clock::now();
        auto candidates = search_candidates(q_data, q_len, ef);
        auto mid = std::chrono::high_resolution_clock::now();

        std::priority_queue<std::pair<float, int>> result;
        for (int id : candidates) {
            float dist = space->distance(q_data, q_len, set_data[id], set_len[id]);
            result.emplace(dist, id);
            if (result.size() > k) {
                result.pop();
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        metric_cand_num += candidates.size();
        metric_cand_gen_time += std::chrono::duration_cast<std::chrono::microseconds>(mid - begin).count();
        metric_rerank_time += std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        for (int id : candidates) {
            metric_rerank_dist_comps += q_len * set_len[id];
        }

        return result;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"cand_num", metric_cand_num},
            {"cand_gen_time", metric_cand_gen_time},
            {"rerank_time", metric_rerank_time},
            {"rerank_dist_comps", metric_rerank_dist_comps},
        };
    }

    void reset_metrics() override {
        metric_cand_num = 0;
        metric_cand_gen_time = 0;
        metric_rerank_time = 0;
        metric_rerank_dist_comps = 0;
    }
};

} // namespace vss
