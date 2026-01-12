#pragma once

#include <Eigen/Dense>
#include <hnswlib/hnswlib.h>

#include "index.h"

namespace vss {

class JLProjector {
public:
    int dim;
    int dim0;
    Eigen::MatrixXf proj_matrix;

    JLProjector(int dim, int dim0) : dim(dim), dim0(dim0) {
        proj_matrix.resize(dim, dim0);

        std::default_random_engine rng;
        std::uniform_int_distribution<int> bit(0, 1);
        float* p = proj_matrix.data();
        const float v = 1.0f / std::sqrt(dim0);
        const int n = proj_matrix.size();
        for (int i = 0; i < n; i++) {
            p[i] = bit(rng) ? v : -v;
        }
    }

    void project(const float* data, int len, float* proj_data) const {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(data, len, dim);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y(proj_data, len, dim0);
        Y.noalias() = X * proj_matrix;
    }
};

class SimHasher {
public:
    int dim;
    int ksim;
    float* gauss_vectors;

    hnswlib::InnerProductSpace* space;
    hnswlib::DISTFUNC<float> dist_func;
    void* dist_func_param;

    SimHasher(int dim, int ksim) : dim(dim), ksim(ksim) {
        space = new hnswlib::InnerProductSpace(dim);
        dist_func = space->get_dist_func();
        dist_func_param = space->get_dist_func_param();

        std::default_random_engine rng;
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        gauss_vectors = new float[dim * ksim];
        float* g = gauss_vectors;
        for (int i = 0; i < ksim; i++, g += dim) {
            for (int j = 0; j < dim; j++) {
                g[j] = gauss(rng);
            }
        }
    }

    ~SimHasher() {
        delete[] gauss_vectors;
        delete space;
    }

    int hash(const float* data) const {
        int hash_value = 0;
        const float* g = gauss_vectors;
        for (int i = 0; i < ksim; i++, g += dim) {
            float sim = dist_func(data, g, dist_func_param);
            if (sim > 0) {
                hash_value |= (1 << i);
            }
        }
        return hash_value;
    }
};

class MuveraFDE {
public:
    int dim;
    int Rreps;
    int ksim;
    int dproj;
    int B;
    int dfde;
    int dFDE;

    std::vector<JLProjector*> projectors;
    std::vector<SimHasher*> hashers;

    MuveraFDE(int dim, int Rreps = 20, int ksim = 5, int dproj = 32)
        : dim(dim), Rreps(Rreps), ksim(ksim), dproj(dproj) {
        B = 1 << ksim;
        dfde = B * dproj;
        dFDE = dfde * Rreps;

        projectors.reserve(Rreps);
        hashers.reserve(Rreps);
        for (int r = 0; r < Rreps; r++) {
            projectors.push_back(new JLProjector(dim, dproj));
            hashers.push_back(new SimHasher(dproj, ksim));
        }
    }

    ~MuveraFDE() {
        for (int r = 0; r < Rreps; r++) {
            delete projectors[r];
            delete hashers[r];
        }
    }

    void encode(const float* data, int len, float* FDE, bool average, bool fill_empty) const {
        for (int r = 0; r < Rreps; r++) {
            std::vector<std::vector<int>> buckets(B);
            float* centroids = new float[B * dim];
            memset(centroids, 0, sizeof(float) * B * dim);

            // 分桶
            const float* vec = data;
            for (int i = 0; i < len; i++, vec += dim) {
                int hash = hashers[r]->hash(vec);
                buckets[hash].push_back(i);
            }

            // 计算每个桶的质心
            float* centroid = centroids;
            for (int b = 0; b < B; b++, centroid += dim) {
                if (buckets[b].empty()) {
                    if (fill_empty) {
                        // 选择 Hamming(b,b') 最近的非空桶 b' 中的一个向量
                        std::vector<std::vector<int>> buckets_by_dist(ksim + 1);
                        for (int bb = 0; bb < B; bb++) {
                            if (buckets[bb].empty()) {
                                continue;
                            }
                            int d = __builtin_popcount(b ^ bb);
                            buckets_by_dist[d].insert(buckets_by_dist[d].end(), buckets[bb].begin(), buckets[bb].end());
                        }
                        int mid_d = 1;
                        while (buckets_by_dist[mid_d].empty()) {
                            mid_d++;
                        }
                        std::uniform_int_distribution<int> distrib(0, buckets_by_dist[mid_d].size() - 1);
                        std::default_random_engine rng;
                        int rand = distrib(rng);
                        memcpy(centroid, data + buckets_by_dist[mid_d][rand] * dim, sizeof(float) * dim);
                    }
                    continue;
                }

                for (int idx : buckets[b]) {
                    const float* vec = data + idx * dim;
                    for (int i = 0; i < dim; i++) {
                        centroid[i] += vec[i];
                    }
                }

                if (average) {
                    float inv_size = 1.0f / buckets[b].size();
                    for (int i = 0; i < dim; i++) {
                        centroid[i] *= inv_size;
                    }
                }
            }

            // 投影并存储
            projectors[r]->project(centroids, B, FDE + r * dfde);

            delete[] centroids;
        }
    }
};

template<bool use_hnsw>
class MuveraIndex : public RerankIndex {
public:
    MuveraFDE* mufde;
    std::vector<float*> set_fde;
    hnswlib::SpaceInterface<float>* hnsw_space;
    hnswlib::HierarchicalNSW<float>* hnsw_index;

    MuveraIndex(int dim, VSSSpace* space, int Rreps, int ksim, int dproj) : RerankIndex(dim, space) {
        mufde = new MuveraFDE(dim, Rreps, ksim, dproj);
    }

    ~MuveraIndex() {
        delete mufde;
        if (use_hnsw) {
            delete hnsw_index;
            delete hnsw_space;
        } else {
            for (auto fde : set_fde) {
                delete[] fde;
            }
        }
    }

    void build_index() override {
        if (use_hnsw) {
            hnsw_space = new hnswlib::InnerProductSpace(mufde->dFDE);
            hnsw_index = new hnswlib::HierarchicalNSW<float>(hnsw_space, set_num, 16, 200);
            float* fde = new float[mufde->dFDE];
            for (int i = 0; i < set_num; i++) {
                mufde->encode(set_data[i], set_len[i], fde, true, true);
                hnsw_index->addPoint(fde, i);
            }
            delete[] fde;
        } else {
            for (int i = 0; i < set_num; i++) {
                float* fde = new float[mufde->dFDE];
                mufde->encode(set_data[i], set_len[i], fde, true, true);
                set_fde.push_back(fde);
            }
        }
    };

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> candidates;
        float* q_fde = new float[mufde->dFDE];
        mufde->encode(q_data, q_len, q_fde, false, false);
        if (use_hnsw) {
            hnsw_index->ef_ = q_k;
            candidates = hnsw_index->searchKnn(q_fde, q_k);
        } else {
            for (int i = 0; i < set_num; i++) {
                float dist = space->dist_func(q_fde, set_fde[i], space->dist_func_param);
                candidates.emplace(dist, i);
                if (candidates.size() > q_k) {
                    candidates.pop();
                }
            }
        }
        delete[] q_fde;

        std::unordered_set<int> result;
        while (!candidates.empty()) {
            result.insert(candidates.top().second);
            candidates.pop();
        }
        return result;
    }
};

} // namespace vss
