#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

#include "index.h"

namespace vss {

class IVFPQPointwiseIndex : public RerankIndex {
public:
    std::vector<int> vec_to_set;

    int nlist;  // 倒排表数量
    int m;      // PQ分块数
    int nbits;  // 每个子量化器bit数
    int nprobe; // 搜索时访问的倒排表数量

    faiss::IndexFlat* quantizer;
    faiss::IndexIVFPQ* index;

    IVFPQPointwiseIndex(int dim, VSSSpace* space, int nlist = 100, int m = 8, int nbits = 8)
        : RerankIndex(dim, space), nlist(nlist), m(m), nbits(nbits) {}

    ~IVFPQPointwiseIndex() {
        delete index;
        delete quantizer;
    }

    void build_index() override {
        vec_to_set.reserve(vec_num);
        for (int i = 0; i < set_num; i++) {
            for (int j = 0; j < set_len[i]; j++) {
                vec_to_set.push_back(i);
            }
        }

        quantizer = new faiss::IndexFlatIP(dim);
        index = new faiss::IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss::METRIC_INNER_PRODUCT);

        index->train(vec_num, vec_data);
        index->add(vec_num, vec_data);
    }

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        index->nprobe = 10;
        std::vector<float> D(q_len * q_k);
        std::vector<faiss::idx_t> I(q_len * q_k);
        index->search(q_len, q_data, q_k, D.data(), I.data());
        std::unordered_set<int> candidates;
        for (auto id : I) {
            candidates.insert(vec_to_set[id]);
        }
        return candidates;
    }
};

} // namespace vss
