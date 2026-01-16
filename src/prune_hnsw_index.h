#pragma once
#include <set>

#include "hnsw.h"
#include "index.h"

namespace vss {

class PruneHNSWIndex : public VSSIndex {
public:
    size_t vec_num;
    float* vec_data;
    size_t base_num;
    std::vector<const float*> base_data;
    std::vector<int> base_len;
    std::vector<int> vec_to_base;

    int M;
    int ef_construction;
    HNSW<float>* hnsw;

    long metric_cand_num;
    long metric_rerank_dist_comps;

    PruneHNSWIndex(int dim, VSSSpace* space, int M, int ef_construction)
        : VSSIndex(dim, space), M(M), ef_construction(ef_construction) {}

    ~PruneHNSWIndex() { delete hnsw; }

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
        for (size_t i = 0; i < vec_num; i++, data += dim) {
            hnsw->add_point(data, i);
        }
    }

    // IGP 式增量搜索
    // struct Context {
    //     const float* query;
    //     std::vector<bool> visited_list;
    //     std::priority_queue<std::pair<float, id_t>> top_candidates;
    //     std::priority_queue<std::pair<float, id_t>> candidate_set;
    //     std::unordered_map<id_t, float> top_cand_back;
    //     std::unordered_map<id_t, float> cand_set_back;

    //     std::unordered_set<int> bases;
    //     std::vector<float> dists;
    //     float radius;

    //     Context(const float* query, int vec_num, int base_num)
    //         : query(query), visited_list(vec_num), dists(base_num, std::numeric_limits<float>::max()) {}
    // };

    // void incremental_search(struct Context& ctx, int k, int ef) {
    //     for (auto [id, dist] : ctx.top_cand_back) {
    //         ctx.top_candidates.emplace(dist, id);
    //         if (ctx.top_candidates.size() > ef) {
    //             ctx.top_candidates.pop();
    //         }
    //     }
    //     float lower_bound = ctx.top_candidates.top().first;
    //     for (auto [id, dist] : ctx.cand_set_back) {
    //         if (ctx.top_candidates.size() < ef || dist < lower_bound) {
    //             ctx.candidate_set.emplace(-dist, id);
    //         }
    //     }

    //     while (!ctx.candidate_set.empty()) {
    //         auto [cur_dist, cur_id] = ctx.candidate_set.top();
    //         if (-cur_dist > lower_bound && ctx.top_candidates.size() >= ef) {
    //             break;
    //         }
    //         ctx.candidate_set.pop();
    //         ctx.cand_set_back.erase(cur_id);

    //         linklist_t* ll = hnsw->addr_linklist(cur_id, 0);
    //         int size = hnsw->get_ll_size(ll);
    //         id_t* neighbors = hnsw->get_ll_neighbors(ll);

    //         hnsw->metric_hops++;

    //         for (int i = 0; i < size; i++) {
    //             id_t nei_id = neighbors[i];
    //             if (ctx.visited_list[nei_id]) {
    //                 continue;
    //             }
    //             ctx.visited_list[nei_id] = true;

    //             float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(nei_id), hnsw->dist_func_param);
    //             hnsw->metric_distance_computations++;
    //             ctx.cand_set_back[nei_id] = dist;
    //             ctx.top_cand_back[nei_id] = dist;

    //             if (ctx.top_candidates.size() < ef || dist < lower_bound) {
    //                 ctx.candidate_set.emplace(-dist, nei_id);
    //                 ctx.top_candidates.emplace(dist, nei_id);
    //                 if (ctx.top_candidates.size() > ef) {
    //                     ctx.top_candidates.pop();
    //                 }
    //                 lower_bound = ctx.top_candidates.top().first;
    //             }
    //         }
    //     }

    //     while (ctx.candidate_set.size() > 0) {
    //         ctx.candidate_set.pop();
    //     }
    //     while (ctx.top_candidates.size() > k) {
    //         ctx.top_candidates.pop();
    //     }

    //     ctx.bases.clear();
    //     ctx.radius = ctx.top_candidates.top().first;
    //     while (!ctx.top_candidates.empty()) {
    //         auto [dist, id] = ctx.top_candidates.top();
    //         ctx.top_candidates.pop();
    //         ctx.top_cand_back.erase(id);
    //         int B = vec_to_base[id];
    //         ctx.bases.insert(B);
    //         ctx.dists[B] = std::min(ctx.dists[B], dist);
    //     }
    // }

    // 轻量增量搜索
    struct Context {
        const float* query;
        std::vector<bool> visited_list;
        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;
        std::vector<std::pair<float, id_t>> kicked;
        std::vector<std::pair<float, id_t>> pruned;

        std::unordered_set<int> bases;
        std::vector<float> dists;
        float radius;

        Context(const float* query, int vec_num, int base_num)
            : query(query), visited_list(vec_num), dists(base_num, std::numeric_limits<float>::max()) {}
    };

    void incremental_search(struct Context& ctx, int k, int ef) {
        std::vector<std::pair<float, id_t>> tmp_kicked;
        std::vector<std::pair<float, id_t>> tmp_pruned;
        for (auto& pair : ctx.kicked) {
            ctx.top_candidates.push(pair);
            if (ctx.top_candidates.size() > ef) {
                tmp_kicked.push_back(ctx.top_candidates.top());
                ctx.top_candidates.pop();
            }
        }
        float lower_bound = ctx.top_candidates.top().first;
        for (auto& [dist, id] : ctx.pruned) {
            if (ctx.top_candidates.size() < ef || dist < lower_bound) {
                ctx.candidate_set.emplace(-dist, id);
                ctx.top_candidates.emplace(dist, id);
                if (ctx.top_candidates.size() > ef) {
                    tmp_kicked.push_back(ctx.top_candidates.top());
                    ctx.top_candidates.pop();
                }
                lower_bound = ctx.top_candidates.top().first;
            } else {
                tmp_pruned.emplace_back(dist, id);
            }
        }
        ctx.kicked = std::move(tmp_kicked);
        ctx.pruned = std::move(tmp_pruned);

        while (!ctx.candidate_set.empty()) {
            auto [cur_dist, cur_id] = ctx.candidate_set.top();
            if (-cur_dist > lower_bound && ctx.top_candidates.size() >= ef) {
                // ctx.radius = -cur_dist;
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

                hnsw->metric_distance_computations++;

                float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(nei_id), hnsw->dist_func_param);
                if (ctx.top_candidates.size() < ef || dist < lower_bound) {
                    ctx.candidate_set.emplace(-dist, nei_id);
                    ctx.top_candidates.emplace(dist, nei_id);
                    if (ctx.top_candidates.size() > ef) {
                        ctx.kicked.push_back(ctx.top_candidates.top());
                        ctx.top_candidates.pop();
                    }
                    lower_bound = ctx.top_candidates.top().first;
                } else {
                    ctx.pruned.emplace_back(dist, nei_id);
                }
            }
        }

        ctx.radius = ctx.top_candidates.top().first;
        while (ctx.top_candidates.size() > k) {
            ctx.kicked.push_back(ctx.top_candidates.top());
            ctx.top_candidates.pop();
        }

        ctx.bases.clear();
        while (!ctx.top_candidates.empty()) {
            auto [dist, id] = ctx.top_candidates.top();
            ctx.top_candidates.pop();
            int B = vec_to_base[id];
            ctx.bases.insert(B);
            ctx.dists[B] = std::min(ctx.dists[B], dist);
        }
    }

    std::vector<Context> contexts;

    void prepare(const float* q_data, int q_len, int k, int ef) override {
        contexts.clear();
        for (int q = 0; q < q_len; q++) {
            contexts.emplace_back(q_data + q * dim, vec_num, base_num);
        }
    }

    // 测试增量搜索
    // std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
    //     for (Context& ctx : contexts) {
    //         id_t ep_id = hnsw->search_down_to_level<true>(hnsw->enterpoint, ctx.query, 0);
    //         float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(ep_id), hnsw->dist_func_param);
    //         ctx.top_candidates.emplace(dist, ep_id);
    //         ctx.candidate_set.emplace(-dist, ep_id);
    //         ctx.visited_list[ep_id] = true;
    //     }

    //     std::unordered_set<int> candidates;
    //     for (int t = 0; t < ef; t += 10) {
    //         for (int q = 0; q < q_len; q++) {
    //             Context& ctx = contexts[q];
    //             incremental_search(ctx, 10, 20);
    //             candidates.insert(ctx.bases.begin(), ctx.bases.end());
    //         }
    //     }
    //     metric_cand_num = candidates.size();

    //     std::priority_queue<std::pair<float, int>> result;
    //     for (int B : candidates) {
    //         float dist = space->distance(q_data, q_len, base_data[B], base_len[B]);
    //         result.emplace(dist, B);
    //         if (result.size() > k) {
    //             result.pop();
    //         }
    //         metric_rerank_dist_comps += q_len * base_len[B];
    //     }
    //     return result;
    // }

    // 增量搜索+剪枝+DCO
    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        for (Context& ctx : contexts) {
            id_t ep_id = hnsw->search_down_to_level<true>(hnsw->enterpoint, ctx.query, 0);
            float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(ep_id), hnsw->dist_func_param);
            ctx.top_candidates.emplace(dist, ep_id);
            ctx.candidate_set.emplace(-dist, ep_id);
            ctx.visited_list[ep_id] = true;
        }

        std::priority_queue<std::pair<float, int>> result;
        std::vector<bool> visited(base_num);
        result.emplace(std::numeric_limits<float>::max(), -1);

        for (int t = 0; t < ef; t += 10) {
            for (int q = 0; q < q_len; q++) {
                Context& ctx = contexts[q];
                incremental_search(ctx, 10, 20);
                for (int B : ctx.bases) {
                    if (visited[B]) {
                        continue;
                    }
                    visited[B] = true;
                    metric_cand_num++;

                    float worst = result.top().first;
                    if (worst < std::numeric_limits<float>::max()) {
                        float lb = 0;
                        for (int q = 0; q < q_len && lb < worst; q++) {
                            lb += std::min(contexts[q].dists[B], contexts[q].radius);
                        }
                        if (lb >= worst) {
                            continue;
                        }
                    }

                    float dist = 0;
                    std::vector<int> to_compute;
                    for (int q = 0; q < q_len; q++) {
                        if (contexts[q].dists[B] == std::numeric_limits<float>::max()) {
                            to_compute.push_back(q);
                        } else {
                            dist += contexts[q].dists[B];
                        }
                    }
                    std::sort(to_compute.begin(), to_compute.end(),
                              [&](int q1, int q2) { return contexts[q1].radius < contexts[q2].radius; });
                    for (int i = 0; i < to_compute.size() && dist < worst; i++) {
                        Context& ctx = contexts[to_compute[i]];
                        float d = std::numeric_limits<float>::max();
                        for (int b = 0; b < base_len[B]; b++) {
                            d = std::min(d, space->vdist(ctx.query, base_data[B] + b * dim));
                        }
                        ctx.dists[B] = d;
                        dist += d;
                        metric_rerank_dist_comps += base_len[B];
                    }
                    if (dist < worst) {
                        result.emplace(dist, B);
                        if (result.size() > k) {
                            result.pop();
                        }
                    }
                }
            }
        }

        return result;
    }

    // 测试剪枝+DCO
    // std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
    //     std::unordered_set<int> candidates;
    //     for (Context& ctx : contexts) {
    //         id_t ep_id = hnsw->search_down_to_level<true>(hnsw->enterpoint, ctx.query, 0);
    //         float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(ep_id), hnsw->dist_func_param);
    //         ctx.top_candidates.emplace(dist, ep_id);
    //         ctx.candidate_set.emplace(-dist, ep_id);
    //         ctx.visited_list[ep_id] = true;
    //         incremental_search(ctx, ef, ef);
    //         candidates.insert(ctx.bases.begin(), ctx.bases.end());
    //     }
    //     metric_cand_num = candidates.size();

    //     std::priority_queue<std::pair<float, int>> result;
    //     result.emplace(std::numeric_limits<float>::max(), -1);
    //     for (int B : candidates) {
    //         float worst = result.top().first;
    //         if (worst < std::numeric_limits<float>::max()) {
    //             float lb = 0;
    //             for (int q = 0; q < q_len && lb < worst; q++) {
    //                 lb += std::min(contexts[q].dists[B], contexts[q].radius);
    //             }
    //             if (lb >= worst) {
    //                 continue;
    //             }
    //         }

    //         float dist = 0;
    //         std::vector<int> to_compute(q_len);
    //         std::iota(to_compute.begin(), to_compute.end(), 0);
    //         // for (int q = 0; q < q_len; q++) {
    //         //     if (contexts[q].dists[B] == std::numeric_limits<float>::max()) {
    //         //         to_compute.push_back(q);
    //         //     } else {
    //         //         dist += contexts[q].dists[B];
    //         //     }
    //         // }
    //         std::sort(to_compute.begin(), to_compute.end(),
    //                   [&](int q1, int q2) { return contexts[q1].radius < contexts[q2].radius; });
    //         for (int i = 0; i < to_compute.size() && dist < worst; i++) {
    //             Context& ctx = contexts[to_compute[i]];
    //             float d = std::numeric_limits<float>::max();
    //             for (int b = 0; b < base_len[B]; b++) {
    //                 d = std::min(d, space->vdist(ctx.query, base_data[B] + b * dim));
    //             }
    //             ctx.dists[B] = d;
    //             dist += d;
    //             metric_rerank_dist_comps += base_len[B];
    //         }
    //         if (dist < worst) {
    //             result.emplace(dist, B);
    //             if (result.size() > k) {
    //                 result.pop();
    //             }
    //         }
    //     }

    //     return result;
    // }

    // 测试 LB 剪枝
    // std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
    //     std::unordered_set<int> candidates;
    //     for (Context& ctx : contexts) {
    //         id_t ep_id = hnsw->search_down_to_level<true>(hnsw->enterpoint, ctx.query, 0);
    //         float dist = hnsw->fstdistfunc(ctx.query, hnsw->addr_data(ep_id), hnsw->dist_func_param);
    //         ctx.top_candidates.emplace(dist, ep_id);
    //         ctx.candidate_set.emplace(-dist, ep_id);
    //         ctx.visited_list[ep_id] = true;
    //         incremental_search(ctx, ef, ef);
    //         candidates.insert(ctx.bases.begin(), ctx.bases.end());
    //     }
    //     metric_cand_num = candidates.size();

    //     std::priority_queue<std::pair<float, int>> order;
    //     for (int B : candidates) {
    //         float lb = 0;
    //         for (int q = 0; q < q_len; q++) {
    //             lb += std::min(contexts[q].dists[B], contexts[q].radius);
    //         }
    //         order.emplace(-lb, B);
    //     }
    //     std::priority_queue<std::pair<float, int>> result;
    //     while (!order.empty()) {
    //         auto [lb, B] = order.top();
    //         order.pop();
    //         if (result.size() >= k && -lb > result.top().first) {
    //             break;
    //         }
    //         float dist = space->distance(q_data, q_len, base_data[B], base_len[B]);
    //         result.emplace(dist, B);
    //         if (result.size() > k) {
    //             result.pop();
    //         }
    //         metric_rerank_dist_comps += q_len * base_len[B];
    //     }

    //     return result;
    // }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"hnsw_hops", hnsw->metric_hops},
            {"hnsw_dist_comps", hnsw->metric_distance_computations},
            {"cand_num", metric_cand_num},
            {"rerank_dist_comps", metric_rerank_dist_comps},
            {"tot_dist_comps", hnsw->metric_distance_computations + metric_rerank_dist_comps},
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
