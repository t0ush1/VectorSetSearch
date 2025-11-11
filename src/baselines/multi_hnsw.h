#pragma once

#include "space.h"

namespace vss {

typedef unsigned int id_t;
typedef unsigned int linklist_t;

class MultiHNSW {
public:
    class VisitedList {
    public:
        typedef unsigned short tag_t;

        size_t num_elements;
        tag_t tag;
        tag_t* mass;

        VisitedList(size_t num_elements) : num_elements(num_elements), tag(-1), mass(new tag_t[num_elements]) {}

        inline void reset() {
            tag++;
            if (tag == 0) {
                tag = 1;
                memset(mass, 0, sizeof(tag_t) * num_elements);
            }
        }

        inline void visit(id_t id) { mass[id] = tag; }

        inline bool is_visited(id_t id) const { return mass[id] == tag; }

        ~VisitedList() { delete[] mass; }
    };

    VSSSpace* space;

    size_t max_elements;
    size_t cur_elements;

    size_t M;
    size_t max_M;
    size_t max_M0;
    size_t ef_construction;
    size_t ef;

    int max_level;
    id_t enterpoint;
    VisitedList* visited_list;

    size_t size_links_level;
    size_t size_links_level0;

    std::vector<char*> elements;
    std::vector<char*> linklists;
    std::vector<int> element_lens;
    std::vector<int> element_levels;

    std::default_random_engine level_generator;
    std::default_random_engine update_probability_generator;

    long metric_distance_computations;
    long metric_hops;

    MultiHNSW(VSSSpace* space, size_t max_elements, size_t M = 16, size_t ef_construction = 200,
              size_t random_seed = 100) {
        this->space = space;

        this->max_elements = max_elements;
        this->cur_elements = 0;

        this->M = M;
        this->max_M = M;
        this->max_M0 = M * 2;
        this->ef_construction = std::max(ef_construction, M);
        this->ef = 10;

        this->max_level = -1;
        this->enterpoint = -1;
        this->visited_list = new VisitedList(max_elements);

        this->size_links_level = sizeof(linklist_t) + max_M * sizeof(id_t);
        this->size_links_level0 = sizeof(linklist_t) + max_M0 * sizeof(id_t);

        this->elements.resize(max_elements);
        this->linklists.resize(max_elements);
        this->element_lens.resize(max_elements);
        this->element_levels.resize(max_elements);

        this->level_generator.seed(random_seed);
        this->update_probability_generator.seed(random_seed + 1);

        this->metric_distance_computations = 0;
        this->metric_hops = 0;
    }

    ~MultiHNSW() {
        free(visited_list);
        for (id_t i = 0; i < cur_elements; i++) {
            free(elements[i]);
            if (element_levels[i] > 0) {
                free(linklists[i]);
            }
        }
    }

    inline int get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator)) / log(M);
        return (int)r;
    }

    inline linklist_t* addr_link_level0(id_t id) const { return (linklist_t*)elements[id]; }

    inline float* addr_data(id_t id) const { return (float*)(elements[id] + size_links_level0); }

    inline linklist_t* addr_link_level(id_t id, int level) const {
        return (linklist_t*)(linklists[id] + (level - 1) * size_links_level);
    }

    inline linklist_t* addr_linklist(id_t id, int level) const {
        return level == 0 ? addr_link_level0(id) : addr_link_level(id, level);
    }

    inline int get_ll_size(linklist_t* ll) const { return *((int*)ll); }

    inline id_t* get_ll_neighbors(linklist_t* ll) const { return (id_t*)(ll + 1); }

    inline void set_ll_size(linklist_t* ll, int size) { *((int*)ll) = size; }

    template<bool is_search>
    id_t search_down_to_level(id_t ep_id, const float* q_data, int q_len, int level) {
        id_t cur_id = ep_id;
        float cur_dist = space->distance(q_data, q_len, addr_data(cur_id), element_lens[cur_id]);
        for (int lev = max_level; lev > level; lev--) {
            bool changed = true;
            while (changed) {
                changed = false;

                linklist_t* ll = addr_linklist(cur_id, lev);
                int size = get_ll_size(ll);
                id_t* neighbors = get_ll_neighbors(ll);

                if (is_search) {
                    metric_hops++;
                }

                for (int i = 0; i < size; i++) {
                    id_t nei_id = neighbors[i];
                    float d = space->distance(q_data, q_len, addr_data(nei_id), element_lens[nei_id]);

                    if (is_search) {
                        metric_distance_computations += q_len * element_lens[nei_id];
                    }

                    if (d < cur_dist) {
                        cur_dist = d;
                        cur_id = nei_id;
                        changed = true;
                    }
                }
            }
        }
        return cur_id;
    }

    template<bool is_search>
    std::priority_queue<std::pair<float, id_t>> search_level(id_t ep_id, const float* q_data, int q_len, int level) {
        visited_list->reset();
        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;
        float lower_bound = space->distance(q_data, q_len, addr_data(ep_id), element_lens[ep_id]);
        top_candidates.emplace(lower_bound, ep_id);
        candidate_set.emplace(-lower_bound, ep_id);
        visited_list->visit(ep_id);

        size_t ef_ = is_search ? ef : ef_construction;
        while (!candidate_set.empty()) {
            auto [cur_dist, cur_id] = candidate_set.top();
            if (-cur_dist > lower_bound && top_candidates.size() >= ef_) {
                break;
            }
            candidate_set.pop();

            linklist_t* ll = addr_linklist(cur_id, level);
            int size = get_ll_size(ll);
            id_t* neighbors = get_ll_neighbors(ll);

            if (is_search) {
                metric_hops++;
            }

            for (int i = 0; i < size; i++) {
                id_t nei_id = neighbors[i];
                if (visited_list->is_visited(nei_id)) {
                    continue;
                }
                visited_list->visit(nei_id);

                float dist = space->distance(q_data, q_len, addr_data(nei_id), element_lens[nei_id]);

                if (is_search) {
                    metric_distance_computations += q_len * element_lens[nei_id];
                }

                if (top_candidates.size() < ef_ || dist < lower_bound) {
                    candidate_set.emplace(-dist, nei_id);
                    top_candidates.emplace(dist, nei_id);
                    if (top_candidates.size() > ef_) {
                        top_candidates.pop();
                    }
                    lower_bound = top_candidates.top().first;
                }
            }
        }

        return top_candidates;
    }

    void get_neighbors_by_heuristic2(id_t cur_id, std::priority_queue<std::pair<float, id_t>>& top_candidates,
                                     size_t M) const {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<float, id_t>> queue_closest;
        std::vector<std::pair<float, id_t>> return_list;

        while (!top_candidates.empty()) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (!queue_closest.empty() && return_list.size() < M) {
            auto [cur_dist, cur_id] = queue_closest.top();
            queue_closest.pop();
            bool good = true;
            for (auto& [_, other_id] : return_list) {
                float dist = space->distance(addr_data(cur_id), element_lens[cur_id], addr_data(other_id),
                                             element_lens[other_id]);
                if (dist < -cur_dist) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.emplace_back(cur_dist, cur_id);
            }
        }

        for (auto& [dist, id] : return_list) {
            top_candidates.emplace(-dist, id);
        }
    }

    id_t mutually_connect_new_element(id_t cur_id, std::priority_queue<std::pair<float, id_t>>& top_candidates,
                                      int level) {
        get_neighbors_by_heuristic2(cur_id, top_candidates, M);

        std::vector<id_t> selected_neighbors;
        selected_neighbors.reserve(M);
        while (!top_candidates.empty()) {
            selected_neighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        id_t next_id = selected_neighbors.back();

        linklist_t* ll = addr_linklist(cur_id, level);
        set_ll_size(ll, selected_neighbors.size());
        id_t* neighbors = get_ll_neighbors(ll);
        for (int i = 0; i < selected_neighbors.size(); i++) {
            assert(neighbors[i] == 0);
            assert(level <= element_levels[selected_neighbors[i]]);
            assert(selected_neighbors[i] != cur_id);
            neighbors[i] = selected_neighbors[i];
        }

        size_t level_M = level == 0 ? max_M0 : max_M;
        for (int i = 0; i < selected_neighbors.size(); i++) {
            id_t nei_id = selected_neighbors[i];

            linklist_t* nei_ll = addr_linklist(nei_id, level);
            int nei_size = get_ll_size(nei_ll);
            id_t* nei_neighbors = get_ll_neighbors(nei_ll);

            assert(nei_size <= level_M);
            if (nei_size < level_M) {
                set_ll_size(nei_ll, nei_size + 1);
                nei_neighbors[nei_size] = cur_id;
            } else {
                std::priority_queue<std::pair<float, id_t>> candidates;
                float dist =
                    space->distance(addr_data(nei_id), element_lens[nei_id], addr_data(cur_id), element_lens[cur_id]);
                candidates.emplace(dist, cur_id);
                for (int j = 0; j < nei_size; j++) {
                    dist = space->distance(addr_data(nei_id), element_lens[nei_id], addr_data(nei_neighbors[j]),
                                           element_lens[nei_neighbors[j]]);
                    candidates.emplace(dist, nei_neighbors[j]);
                }
                get_neighbors_by_heuristic2(nei_id, candidates, level_M);

                nei_size = 0;
                while (!candidates.empty()) {
                    nei_neighbors[nei_size++] = candidates.top().second;
                    candidates.pop();
                }
                set_ll_size(nei_ll, nei_size);
            }
        }

        return next_id;
    }

    id_t add_point(const float* data, int len) {
        id_t cur_id = cur_elements++;
        int cur_level = get_random_level();
        element_levels[cur_id] = cur_level;
        element_lens[cur_id] = len;

        elements[cur_id] = (char*)malloc(size_links_level0 + space->data_size * len);
        memset(addr_link_level0(cur_id), 0, size_links_level0);
        memcpy(addr_data(cur_id), data, space->data_size * len);

        if (cur_level > 0) {
            linklists[cur_id] = (char*)malloc(size_links_level * cur_level);
            memset(linklists[cur_id], 0, size_links_level * cur_level);
        }

        if (enterpoint == -1) {
            enterpoint = 0;
            max_level = cur_level;
            return cur_id;
        }

        id_t ep_id = enterpoint;

        if (cur_level < max_level) {
            ep_id = search_down_to_level<false>(enterpoint, data, len, cur_level);
        }

        for (int level = std::min(cur_level, max_level); level >= 0; level--) {
            auto top_candidates = search_level<false>(ep_id, data, len, level);
            ep_id = mutually_connect_new_element(cur_id, top_candidates, level);
        }

        if (cur_level > max_level) {
            enterpoint = cur_id;
            max_level = cur_level;
        }
        
        return cur_id;
    }

    std::priority_queue<std::pair<float, id_t>> search_knn(const float* query, int len, size_t k) {
        id_t ep_id = search_down_to_level<true>(enterpoint, query, len, 0);
        auto top_candidates = search_level<true>(ep_id, query, len, 0);
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        return top_candidates;
    }
};

} // namespace vss
