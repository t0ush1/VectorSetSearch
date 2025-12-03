#pragma once

#include <hnswlib/hnswlib.h>

namespace vss {

typedef size_t label_t;
typedef unsigned int id_t;
typedef unsigned int linklist_t;

template<typename dist_t>
class SingleHNSW {
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

    size_t max_elements;
    size_t cur_elements;

    size_t M;
    size_t max_M;
    size_t max_M0;
    size_t ef_construction;
    size_t ef;

    size_t data_size;
    hnswlib::DISTFUNC<dist_t> fstdistfunc;
    void* dist_func_param;

    int max_level;
    id_t enterpoint;
    VisitedList* visited_list;

    size_t size_links_level;
    size_t size_links_level0;
    size_t size_element;

    size_t offset_links;
    size_t offset_data;
    size_t offset_label;

    char* elements;
    char** linklists;
    std::vector<int> element_levels;

    std::default_random_engine level_generator;

    long metric_distance_computations;
    long metric_hops;

    SingleHNSW(hnswlib::SpaceInterface<dist_t>* space, size_t max_elements, size_t M = 16, size_t ef_construction = 200,
               size_t random_seed = 100) {
        this->max_elements = max_elements;
        this->cur_elements = 0;

        this->M = M;
        this->max_M = M;
        this->max_M0 = M * 2;
        this->ef_construction = std::max(ef_construction, M);
        this->ef = 10;

        this->data_size = space->get_data_size();
        this->fstdistfunc = space->get_dist_func();
        this->dist_func_param = space->get_dist_func_param();

        this->max_level = -1;
        this->enterpoint = -1;
        this->visited_list = new VisitedList(max_elements);

        this->size_links_level = sizeof(linklist_t) + max_M * sizeof(id_t);
        this->size_links_level0 = sizeof(linklist_t) + max_M0 * sizeof(id_t);
        this->size_element = size_links_level0 + data_size + sizeof(label_t);

        this->offset_links = 0;
        this->offset_data = size_links_level0;
        this->offset_label = size_links_level0 + data_size;

        this->elements = (char*)malloc(max_elements * size_element);
        this->linklists = (char**)malloc(max_elements * sizeof(void*));
        this->element_levels.resize(max_elements);

        this->level_generator.seed(random_seed);

        this->metric_distance_computations = 0;
        this->metric_hops = 0;
    }

    ~SingleHNSW() {
        free(visited_list);
        free(elements);
        for (id_t i = 0; i < cur_elements; i++) {
            if (element_levels[i] > 0) {
                free(linklists[i]);
            }
        }
        free(linklists);
    }

    inline int get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator)) / log(M);
        return (int)r;
    }

    inline char* addr_element(id_t id) const { return elements + id * size_element; }

    inline linklist_t* addr_link_level0(id_t id) const { return (linklist_t*)(addr_element(id) + offset_links); }

    inline char* addr_data(id_t id) const { return addr_element(id) + offset_data; }

    inline label_t* addr_label(id_t id) const { return (label_t*)(addr_element(id) + offset_label); }

    inline linklist_t* addr_link_level(id_t id, int level) const {
        return (linklist_t*)(linklists[id] + (level - 1) * size_links_level);
    }

    inline linklist_t* addr_linklist(id_t id, int level) const {
        return level == 0 ? addr_link_level0(id) : addr_link_level(id, level);
    }

    inline int get_ll_size(linklist_t* ll) const { return *((int*)ll); }

    inline id_t* get_ll_neighbors(linklist_t* ll) const { return (id_t*)(ll + 1); }

    inline void set_ll_size(linklist_t* ll, int size) { *((int*)ll) = size; }

    template<bool collect_metrics>
    id_t search_down_to_level(id_t ep_id, const void* query, int level) {
        id_t cur_id = ep_id;
        dist_t cur_dist = fstdistfunc(query, addr_data(cur_id), dist_func_param);
        for (int lev = max_level; lev > level; lev--) {
            bool changed = true;
            while (changed) {
                changed = false;

                linklist_t* ll = addr_linklist(cur_id, lev);
                int size = get_ll_size(ll);
                id_t* neighbors = get_ll_neighbors(ll);

                if (collect_metrics) {
                    metric_hops++;
                    metric_distance_computations += size;
                }

                for (int i = 0; i < size; i++) {
                    id_t nei_id = neighbors[i];
                    dist_t d = fstdistfunc(query, addr_data(nei_id), dist_func_param);
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

    template<bool collect_metrics>
    std::priority_queue<std::pair<dist_t, id_t>> search_level(id_t ep_id, const void* query, int level) {
        visited_list->reset();
        std::priority_queue<std::pair<dist_t, id_t>> top_candidates;
        std::priority_queue<std::pair<dist_t, id_t>> candidate_set;
        dist_t lower_bound = fstdistfunc(query, addr_data(ep_id), dist_func_param);
        top_candidates.emplace(lower_bound, ep_id);
        candidate_set.emplace(-lower_bound, ep_id);
        visited_list->visit(ep_id);

        size_t ef_ = collect_metrics ? ef : ef_construction;
        while (!candidate_set.empty()) {
            auto [cur_dist, cur_id] = candidate_set.top();
            if (-cur_dist > lower_bound && top_candidates.size() >= ef_) {
                break;
            }
            candidate_set.pop();

            linklist_t* ll = addr_linklist(cur_id, level);
            int size = get_ll_size(ll);
            id_t* neighbors = get_ll_neighbors(ll);

            if (collect_metrics) {
                metric_hops++;
            }

#ifdef USE_SSE
            _mm_prefetch((char*)(visited_list->mass + neighbors[0]), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_list->mass + neighbors[0] + 64), _MM_HINT_T0);
            _mm_prefetch(addr_data(neighbors[0]), _MM_HINT_T0);
            _mm_prefetch((char*)neighbors, _MM_HINT_T0);
#endif

            for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                _mm_prefetch((char*)(visited_list->mass + neighbors[i + 1]), _MM_HINT_T0);
                _mm_prefetch(addr_data(neighbors[i + 1]), _MM_HINT_T0);
#endif

                id_t nei_id = neighbors[i];
                if (visited_list->is_visited(nei_id)) {
                    continue;
                }
                visited_list->visit(nei_id);

                if (collect_metrics) {
                    metric_distance_computations++;
                }

                dist_t dist = fstdistfunc(query, addr_data(nei_id), dist_func_param);
                if (top_candidates.size() < ef_ || dist < lower_bound) {
                    candidate_set.emplace(-dist, nei_id);
#ifdef USE_SSE
                    _mm_prefetch(addr_data(candidate_set.top().second), _MM_HINT_T0);
#endif
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

    void get_neighbors_by_heuristic2(std::priority_queue<std::pair<dist_t, id_t>>& top_candidates,
                                     const size_t M) const {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, id_t>> queue_closest;
        std::vector<std::pair<dist_t, id_t>> return_list;

        while (!top_candidates.empty()) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        // 防止三角形长边：如果 C-A > C-B 且 C-A > B-A，则不连 C-A，通过 A-B-C 访问
        while (!queue_closest.empty() && return_list.size() < M) {
            auto [cur_dist, cur_id] = queue_closest.top();
            queue_closest.pop();
            bool good = true;
            for (auto& [_, other_id] : return_list) {
                dist_t dist = fstdistfunc(addr_data(cur_id), addr_data(other_id), dist_func_param);
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

    id_t mutually_connect_new_element(id_t cur_id, std::priority_queue<std::pair<dist_t, id_t>>& top_candidates,
                                      int level) {
        get_neighbors_by_heuristic2(top_candidates, M);

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
            linklist_t* other_ll = addr_linklist(selected_neighbors[i], level);
            int other_size = get_ll_size(other_ll);
            id_t* other_neighbors = get_ll_neighbors(other_ll);

            assert(other_size <= level_M);
            if (other_size < level_M) {
                set_ll_size(other_ll, other_size + 1);
                other_neighbors[other_size] = cur_id;
            } else {
                dist_t max_dist = fstdistfunc(addr_data(cur_id), addr_data(selected_neighbors[i]), dist_func_param);
                std::priority_queue<std::pair<dist_t, id_t>> candidates;
                candidates.emplace(max_dist, cur_id);
                for (int j = 0; j < other_size; j++) {
                    candidates.emplace(
                        fstdistfunc(addr_data(other_neighbors[j]), addr_data(selected_neighbors[i]), dist_func_param),
                        other_neighbors[j]);
                }
                get_neighbors_by_heuristic2(candidates, level_M);

                other_size = 0;
                while (!candidates.empty()) {
                    other_neighbors[other_size++] = candidates.top().second;
                    candidates.pop();
                }
                set_ll_size(other_ll, other_size);
            }
        }

        return next_id;
    }

    void add_point(const void* data, label_t label) {
        id_t cur_id = cur_elements++;
        int cur_level = get_random_level();
        element_levels[cur_id] = cur_level;

        memset(addr_element(cur_id), 0, size_element);
        memcpy(addr_data(cur_id), data, data_size);
        memcpy(addr_label(cur_id), &label, sizeof(label_t));

        if (cur_level > 0) {
            linklists[cur_id] = (char*)malloc(size_links_level * cur_level);
            memset(linklists[cur_id], 0, size_links_level * cur_level);
        }

        if (enterpoint == -1) {
            enterpoint = 0;
            max_level = cur_level;
            return;
        }

        id_t ep_id = enterpoint;

        if (cur_level < max_level) {
            ep_id = search_down_to_level<false>(enterpoint, data, cur_level);
        }

        for (int level = std::min(cur_level, max_level); level >= 0; level--) {
            auto top_candidates = search_level<false>(ep_id, data, level);
            ep_id = mutually_connect_new_element(cur_id, top_candidates, level);
        }

        if (cur_level > max_level) {
            enterpoint = cur_id;
            max_level = cur_level;
        }
    }

    std::priority_queue<std::pair<dist_t, label_t>> search_knn(const void* query, size_t k) {
        id_t ep_id = search_down_to_level<true>(enterpoint, query, 0);
        auto top_candidates = search_level<true>(ep_id, query, 0);
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        std::priority_queue<std::pair<dist_t, label_t>> result;
        while (!top_candidates.empty()) {
            auto [dist, id] = top_candidates.top();
            top_candidates.pop();
            result.emplace(dist, *addr_label(id));
        }
        return result;
    }
};

} // namespace vss
