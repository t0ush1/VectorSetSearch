#pragma once
#include <algorithm>
#include <limits>
#include <vector>

#include <hnswlib/hnswlib.h>

namespace vss {

class VSSSpace {
public:
    int dim;
    hnswlib::SpaceInterface<float>* space;
    size_t data_size;
    hnswlib::DISTFUNC<float> dist_func;
    void* dist_func_param;

    VSSSpace(int dim) : dim(dim) {
        space = new hnswlib::InnerProductSpace(dim);
        data_size = space->get_data_size();
        dist_func = space->get_dist_func();
        dist_func_param = space->get_dist_func_param();
    }

    ~VSSSpace() { delete space; }

    inline float vdist(const float* vec1, const float* vec2) const { return dist_func(vec1, vec2, dist_func_param); }

    inline float distance(const float* data1, int len1, const float* data2, int len2) const {
        float sum = 0.0f;
        const float* v1 = data1;
        for (int i = 0; i < len1; i++, v1 += dim) {
            float sim = std::numeric_limits<float>::max();
            const float* v2 = data2;
            for (int j = 0; j < len2; j++, v2 += dim) {
                sim = std::min(sim, vdist(v1, v2));
            }
            sum += sim;
        }
        return sum;
    }
};

} // namespace vss