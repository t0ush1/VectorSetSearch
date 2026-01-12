#pragma once
#include <algorithm>
#include <limits>
#include <vector>

#include <hnswlib/hnswlib.h>

namespace vss {

enum VSSMetric { MAXSIM, DTW, SDTW };

class VSSSpace {
public:
    int dim;
    VSSMetric metric;
    hnswlib::SpaceInterface<float>* space;
    size_t data_size;
    hnswlib::DISTFUNC<float> dist_func;
    void* dist_func_param;

    VSSSpace(int dim, VSSMetric metric, hnswlib::SpaceInterface<float>* space)
        : dim(dim), metric(metric), space(space) {
        data_size = space->get_data_size();
        dist_func = space->get_dist_func();
        dist_func_param = space->get_dist_func_param();
    }

    ~VSSSpace() { delete space; }

    virtual float distance(const float* data1, int len1, const float* data2, int len2) const = 0;
};

class MaxSimSpace : public VSSSpace {
public:
    MaxSimSpace(int dim) : VSSSpace(dim, MAXSIM, new hnswlib::InnerProductSpace(dim)) {}

    float distance(const float* data1, int len1, const float* data2, int len2) const override {
        float sum = 0.0f;
        const float* v1 = data1;
        for (int i = 0; i < len1; i++, v1 += dim) {
            float sim = std::numeric_limits<float>::infinity();
            const float* v2 = data2;
            for (int j = 0; j < len2; j++, v2 += dim) {
                sim = std::min(sim, dist_func(v1, v2, dist_func_param));
            }
            sum += sim;
        }
        return sum;
    }
};

class DTWSpace : public VSSSpace {
public:
    DTWSpace(int dim) : VSSSpace(dim, DTW, new hnswlib::L2Space(dim)) {}

    float distance(const float* data1, int len1, const float* data2, int len2) const override {
        const float INF = std::numeric_limits<float>::infinity();
        std::vector<float> pre(len2 + 1, INF), cur(len2 + 1, INF);
        pre[0] = 0;

        const float* v1 = data1;
        for (int i = 1; i <= len1; i++, v1 += dim) {
            cur[0] = INF;
            const float* v2 = data2;
            for (int j = 1; j <= len2; j++, v2 += dim) {
                cur[j] = dist_func(v1, v2, dist_func_param) + std::min({pre[j], cur[j - 1], pre[j - 1]});
            }
            std::swap(pre, cur);
        }
        return pre[len2];
    }
};

class SDTWSpace : public VSSSpace {
public:
    SDTWSpace(int dim) : VSSSpace(dim, SDTW, new hnswlib::L2Space(dim)) {}

    float distance(const float* data1, int len1, const float* data2, int len2) const override {
        const float INF = std::numeric_limits<float>::infinity();
        std::vector<float> pre(len2 + 1, 0), cur(len2 + 1, 0);

        const float* v1 = data1;
        for (int i = 1; i <= len1; i++, v1 += dim) {
            cur[0] = INF;
            const float* v2 = data2;
            for (int j = 1; j <= len2; j++, v2 += dim) {
                cur[j] = dist_func(v1, v2, dist_func_param) + std::min({pre[j], cur[j - 1], pre[j - 1]});
            }
            std::swap(pre, cur);
        }
        return *std::min_element(pre.begin() + 1, pre.end());
    }
};

} // namespace vss