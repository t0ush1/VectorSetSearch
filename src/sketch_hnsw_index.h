#pragma once
#include <set>

#include <Eigen/Dense>

#include "hnsw.h"
#include "index.h"

namespace vss {

class SketchHNSWIndex : public VSSIndex {
public:
    int vec_num;
    float* vec_data;
    int base_num;
    std::vector<const float*> base_data;
    std::vector<int> base_len;
    std::vector<int> vec_to_base;

    int dim0;
    float* proj_matrix;
    float* vec_sketch;
    std::vector<const float*> base_sketch;

    int M;
    int ef_construction;
    hnswlib::SpaceInterface<float>* sketch_space;
    HNSW<float>* sketch_hnsw;

    long metric_cand_num;
    long metric_rerank_dist_comps;

    SketchHNSWIndex(int dim, VSSSpace* space, int M, int ef_construction)
        : VSSIndex(dim, space), M(M), ef_construction(ef_construction) {}

    ~SketchHNSWIndex() {
        delete sketch_hnsw;
        delete sketch_space;
    }

    inline void project(const float* data, float* sketch, int vec_num) const {
        Eigen::Map<const Eigen::MatrixXf> P(proj_matrix, dim, dim0);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(data, vec_num, dim);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y(sketch, vec_num, dim0);
        Y.noalias() = X * P;
    }

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

        // 向量 sketch 计算
        dim0 = 64;
        proj_matrix = new float[dim * dim0];
        {
            std::default_random_engine rng;
            std::uniform_int_distribution<int> bit(0, 1);
            float* p = proj_matrix;
            const float v = 1.0f / std::sqrt(dim0);
            const int n = dim * dim0;
            for (int i = 0; i < n; i++) {
                p[i] = bit(rng) ? v : -v;
            }
        }

        vec_sketch = new float[vec_num * dim0];
        project(vec_data, vec_sketch, vec_num);

        base_sketch.resize(base_num);
        base_sketch[0] = vec_sketch;
        for (int i = 1; i < base_num; i++) {
            base_sketch[i] = base_sketch[i - 1] + base_len[i - 1] * dim0;
        }

        // TODO 将集合sketch拼成长向量
        sketch_space = new hnswlib::InnerProductSpace(dim0);
        sketch_hnsw = new HNSW<float>(sketch_space, vec_num, M, ef_construction);

        const float* sketch = vec_sketch;
        for (int i = 0; i < vec_num; i++, sketch += dim0) {
            sketch_hnsw->add_point(sketch, i);
        }
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        float* q_sketch = new float[q_len * dim0];
        project(q_data, q_sketch, q_len);

        std::unordered_set<int> candidates;
        sketch_hnsw->ef = ef;
        for (int q = 0; q < q_len; q++) {
            auto r = sketch_hnsw->search_knn(q_sketch + q * dim0, ef);
            while (!r.empty()) {
                candidates.insert(vec_to_base[r.top().second]);
                r.pop();
            }
        }

        std::priority_queue<std::pair<float, int>> result;
        for (int B : candidates) {
            float dist = space->distance(q_data, q_len, base_data[B], base_len[B]);
            result.emplace(dist, B);
            if (result.size() > k) {
                result.pop();
            }
            metric_cand_num++;
            metric_rerank_dist_comps += base_len[B] * q_len;
        }

        return result;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"hops", sketch_hnsw->metric_hops},
            {"dist_comps", sketch_hnsw->metric_distance_computations + metric_rerank_dist_comps},
            {"cand_num", metric_cand_num},
        };
    }

    void reset_metrics() override {
        metric_cand_num = 0;
        metric_rerank_dist_comps = 0;
        sketch_hnsw->metric_hops = 0;
        sketch_hnsw->metric_distance_computations = 0;
    }
};

} // namespace vss
