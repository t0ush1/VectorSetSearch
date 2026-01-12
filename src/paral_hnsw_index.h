#pragma once
#include <set>

#include "hnsw.h"
#include "index.h"

namespace vss {

class ParalHNSWIndex : public VSSIndex {
public:
    int vec_num;
    float* vec_data;
    int base_num;
    std::vector<const float*> base_data;
    std::vector<int> base_len;
    std::vector<int> vec_to_base;

    int M;
    int ef_construction;
    HNSW<float>* hnsw;

    std::default_random_engine search_generator;

    long metric_cand_num;
    long metric_rerank_dist_comps;

    ParalHNSWIndex(int dim, VSSSpace* space, int M, int ef_construction)
        : VSSIndex(dim, space), M(M), ef_construction(ef_construction) {}

    ~ParalHNSWIndex() { delete hnsw; }

    void build(const VSSDataset* base_dataset) override {
        vec_num = base_dataset->size;
        vec_data = new float[vec_num * dim];
        memcpy(vec_data, base_dataset->data, vec_num * space->data_size);

        base_num = base_dataset->set_num;
        base_len = base_dataset->set_len;
        base_data.resize(base_num);
        base_data[0] = vec_data;
        for (int i = 1; i < base_num; i++) {
            base_data[i] = base_data[i - 1] + base_len[i - 1] * dim;
        }

        vec_to_base.reserve(vec_num);
        for (int i = 0; i < base_num; i++) {
            for (int j = 0; j < base_len[i]; j++) {
                vec_to_base.push_back(i);
            }
        }

        hnsw = new HNSW<float>(space->space, vec_num, M, ef_construction);

        const float* data = vec_data;
        for (int i = 0; i < vec_num; i++, data += dim) {
            hnsw->add_point(data, i);
        }
    }

    struct Context {
        const float* query;
        std::vector<bool> visited_list;
        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;
        std::vector<std::pair<float, id_t>> top_cand_back;
        std::unordered_set<int> base;
        std::vector<float> dists;
        std::vector<bool> found;
        float radius;

        Context(const float* query, int vec_num, int base_num)
            : query(query), visited_list(vec_num), dists(base_num), found(base_num) {}
    };

    std::vector<Context> contexts;
    std::vector<int> hits;

    void incremental_search(struct Context& ctx, int k, int ef) {
        for (auto& pair : ctx.top_cand_back) {
            ctx.top_candidates.push(pair);
            if (ctx.top_candidates.size() > ef) {
                ctx.top_candidates.pop();
            }
        }
        ctx.top_cand_back.clear(); // 不清除垃圾桶 ?
        float lower_bound = ctx.top_candidates.top().first;

        while (!ctx.candidate_set.empty()) {
            auto [cur_dist, cur_id] = ctx.candidate_set.top();
            if (-cur_dist > lower_bound && ctx.top_candidates.size() >= ef) {
                break;
            }
            ctx.candidate_set.pop();

            linklist_t* ll = hnsw->addr_linklist(cur_id, 0);
            int size = hnsw->get_ll_size(ll);
            id_t* neighbors = hnsw->get_ll_neighbors(ll);

            hnsw->metric_hops++;

            for (int i = 0; i < size; i++) {
                id_t nei_id = neighbors[i];
                if (ctx.visited_list[nei_id]) {
                    continue;
                }
                ctx.visited_list[nei_id] = true;

                // 在这里将元素加入垃圾桶，将topk移出垃圾桶。参考IGP

                hnsw->metric_distance_computations++;

                float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(nei_id), hnsw->dist_func_param);
                if (ctx.top_candidates.size() < ef || dist < lower_bound) {
                    ctx.candidate_set.emplace(-dist, nei_id);
                    ctx.top_candidates.emplace(dist, nei_id);
                    if (ctx.top_candidates.size() > ef) {
                        ctx.top_cand_back.push_back(ctx.top_candidates.top());
                        ctx.top_candidates.pop();
                    }
                    lower_bound = ctx.top_candidates.top().first;
                }
            }
        }

        while (ctx.top_candidates.size() > k) {
            ctx.top_cand_back.push_back(ctx.top_candidates.top());
            ctx.top_candidates.pop();
        }

        ctx.base.clear();
        ctx.radius = ctx.top_candidates.top().first;
        while (!ctx.top_candidates.empty()) {
            auto [dist, id] = ctx.top_candidates.top();
            ctx.top_candidates.pop();
            int B = vec_to_base[id];
            ctx.base.insert(B);
            if (ctx.found[B]) {
                ctx.dists[B] = std::min(ctx.dists[B], dist);
            } else {
                ctx.dists[B] = dist;
                ctx.found[B] = true;
                hits[B]++;
            }
        }
    }

    void prepare(const float* q_data, int q_len, int k, int ef) override {
        contexts.clear();
        for (int q = 0; q < q_len; q++) {
            contexts.emplace_back(q_data + q * dim, vec_num, base_num);
        }
        hits.assign(base_num, 0);
    }

    // std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
    //     std::priority_queue<std::pair<float, int>> result;

    //     auto update = [&](int B) {
    //         if (hits[B] != 2) {
    //             return;
    //         }
    //         hits[B]++;

    //         bool good = result.size() < k;
    //         if (!good) {
    //             float lower_bound = 0;
    //             for (int q = 0; q < q_len; q++) {
    //                 lower_bound += contexts[q].found[B] ? contexts[q].dists[B] : contexts[q].radius;
    //             }
    //             good = lower_bound < result.top().first;
    //         }
    //         if (!good) {
    //             return;
    //         }

    //         float dist = space->distance(q_data, q_len, base_data[B], base_len[B]);
    //         metric_cand_num++;
    //         metric_rerank_dist_comps += q_len * base_len[B];

    //         result.emplace(dist, B);
    //         if (result.size() > k) {
    //             result.pop();
    //         }
    //     };

    //     for (Context& ctx : contexts) {
    //         id_t ep_id = hnsw->search_down_to_level<true>(hnsw->enterpoint, ctx.query, 0);
    //         float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(ep_id), hnsw->dist_func_param);
    //         ctx.top_candidates.emplace(dist, ep_id);
    //         ctx.candidate_set.emplace(-dist, ep_id);
    //         ctx.visited_list[ep_id] = true;
    //         incremental_search(ctx, 10, 20);
    //         for (int B : ctx.base) {
    //             update(B);
    //         }
    //     }

    //     for (int t = 0, q = 0; t < ef; t++, q = (q + 1) % q_len) {
    //         Context& ctx = contexts[q];
    //         incremental_search(ctx, 10, 20);
    //         for (int B : ctx.base) {
    //             update(B);
    //         }
    //     }

    //     return result;
    // }

    // 均匀增量搜索
    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        std::unordered_set<int> candidates;

        for (Context& ctx : contexts) {
            id_t ep_id = hnsw->search_down_to_level<true>(hnsw->enterpoint, ctx.query, 0);
            float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(ep_id), hnsw->dist_func_param);
            ctx.top_candidates.emplace(dist, ep_id);
            ctx.candidate_set.emplace(-dist, ep_id);
            ctx.visited_list[ep_id] = true;
            incremental_search(ctx, 10, 20);
            candidates.insert(ctx.base.begin(), ctx.base.end());
        }

        for (int t = 0, q = 0; t < ef; t++, q = (q + 1) % q_len) {
            Context& ctx = contexts[q];
            incremental_search(ctx, 10, 20);
            candidates.insert(ctx.base.begin(), ctx.base.end());
        }
        metric_cand_num = candidates.size();

        std::priority_queue<std::pair<float, int>> result;
        for (int B : candidates) {
            float dist = space->distance(q_data, q_len, base_data[B], base_len[B]);
            metric_rerank_dist_comps += q_len * base_len[B];

            result.emplace(dist, B);
            if (result.size() > k) {
                result.pop();
            }
        }

        return result;
    }

    // 按排名打分
    // std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
    //     std::unordered_set<int> candidates;

    //     for (Context& ctx : contexts) {
    //         id_t ep_id = hnsw->search_down_to_level<true>(hnsw->enterpoint, ctx.query, 0);
    //         float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(ep_id), hnsw->dist_func_param);
    //         ctx.top_candidates.emplace(dist, ep_id);
    //         ctx.candidate_set.emplace(-dist, ep_id);
    //         ctx.visited_list[ep_id] = true;
    //         incremental_search(ctx, 10, 20);
    //         candidates.insert(ctx.base.begin(), ctx.base.end());
    //     }

    //     std::vector<float> dists(base_num);
    //     for (int B : candidates) {
    //         dists[B] = space->distance(q_data, q_len, base_data[B], base_len[B]);
    //         metric_rerank_dist_comps += q_len * base_len[B];
    //     }

    //     for (int t = 0; t < ef; t++) {
    //         std::vector<double> scores(q_len);
    //         std::vector<int> sorted(candidates.begin(), candidates.end());
    //         std::vector<int> ranks(base_num);
    //         std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return dists[a] < dists[b]; });
    //         for (int r = 0; r < sorted.size(); r++) {
    //             ranks[sorted[r]] = r;
    //         }
    //         for (int q = 0; q < q_len; q++) {
    //             for (int B : contexts[q].base) {
    //                 scores[q] += 1.0 / ranks[B];
    //             }
    //         }
    //         std::discrete_distribution<int> dist_scores(scores.begin(), scores.end());
    //         int sampled_q = dist_scores(search_generator);
    //         Context& ctx = contexts[sampled_q];
    //         incremental_search(ctx, 10, 20);
    //         candidates.insert(ctx.base.begin(), ctx.base.end());
    //         for (int B : ctx.base) {
    //             if (dists[B] == 0) {
    //                 dists[B] = space->distance(q_data, q_len, base_data[B], base_len[B]);
    //                 metric_rerank_dist_comps += q_len * base_len[B];
    //             }
    //         }
    //     }
    //     metric_cand_num = candidates.size();

    //     std::priority_queue<std::pair<float, int>> result;
    //     for (int B : candidates) {
    //         result.emplace(dists[B], B);
    //         if (result.size() > k) {
    //             result.pop();
    //         }
    //     }

    //     return result;
    // }

    // 按距离打分
    // std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
    //     std::unordered_set<int> candidates;

    //     for (Context& ctx : contexts) {
    //         id_t ep_id = hnsw->search_down_to_level<true>(hnsw->enterpoint, ctx.query, 0);
    //         float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(ep_id), hnsw->dist_func_param);
    //         ctx.top_candidates.emplace(dist, ep_id);
    //         ctx.candidate_set.emplace(-dist, ep_id);
    //         ctx.visited_list[ep_id] = true;
    //         incremental_search(ctx, 10, 20);
    //         candidates.insert(ctx.base.begin(), ctx.base.end());
    //     }

    //     std::vector<float> dists(base_num);
    //     std::vector<double> scores(q_len);
    //     for (int q = 0; q < q_len; q++) {
    //         for (int B : contexts[q].base) {
    //             if (dists[B] == 0) {
    //                 dists[B] = space->distance(q_data, q_len, base_data[B], base_len[B]);
    //                 metric_rerank_dist_comps += q_len * base_len[B];
    //             }
    //             scores[q] += dists[B];
    //         }
    //     }

    //     for (int t = 0; t < ef; t++) {
    //         double max_score = *std::max_element(scores.begin(), scores.end());
    //         double min_score = *std::min_element(scores.begin(), scores.end());
    //         std::vector<double> weights(q_len);
    //         for (int q = 0; q < q_len; q++) {
    //             weights[q] = 1.0 - (scores[q] - min_score) / (max_score - min_score);
    //         }
    //         std::discrete_distribution<int> dd(weights.begin(), weights.end());
    //         int q = dd(search_generator);
    //         Context& ctx = contexts[q];
    //         incremental_search(ctx, 10, 20);
    //         candidates.insert(ctx.base.begin(), ctx.base.end());
    //         scores[q] = 0;
    //         for (int B : ctx.base) {
    //             if (dists[B] == 0) {
    //                 dists[B] = space->distance(q_data, q_len, base_data[B], base_len[B]);
    //                 metric_rerank_dist_comps += q_len * base_len[B];
    //             }
    //             scores[q] += dists[B];
    //         }
    //     }
    //     metric_cand_num = candidates.size();

    //     std::priority_queue<std::pair<float, int>> result;
    //     for (int B : candidates) {
    //         result.emplace(dists[B], B);
    //         if (result.size() > k) {
    //             result.pop();
    //         }
    //     }

    //     return result;
    // }

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
        search_generator.seed(42);
    }
};

} // namespace vss
