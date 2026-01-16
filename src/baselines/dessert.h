#include "index.h"

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <dessert/DocSearch.h>

namespace py = pybind11;
namespace tdsearch = thirdai::search;

namespace vss {

class DessertIndex : public VSSIndex {
public:
    std::unique_ptr<py::scoped_interpreter> _py_guard;
    std::unique_ptr<tdsearch::DocSearch> _doc;
    int _hashes_per_table;
    int _num_tables;

    DessertIndex(int dim, VSSSpace* space) : VSSIndex(dim, space), _hashes_per_table(8), _num_tables(32) {}

    ~DessertIndex() {}

    void build(const VSSDataset* base_dataset) override {
        if (!_py_guard) {
            _py_guard = std::make_unique<py::scoped_interpreter>();
            py::module::import("numpy");
        }

        // DocSearch 需要质心；用单个零向量即可
        py::array_t<float> centroids({static_cast<size_t>(1), static_cast<size_t>(dim)});
        {
            auto buf = centroids.mutable_unchecked<2>();
            for (ssize_t j = 0; j < buf.shape(1); ++j)
                buf(0, j) = 0.f;
        }

        _doc = std::make_unique<tdsearch::DocSearch>(static_cast<uint32_t>(_hashes_per_table),
                                                     static_cast<uint32_t>(_num_tables), static_cast<uint32_t>(dim),
                                                     centroids);

        for (int id = 0; id < base_dataset->set_num; ++id) {
            const float* data = base_dataset->set_data[id];
            const int len = base_dataset->set_len[id];
            if (len <= 0)
                continue;

            // 零拷贝视图包装 (len, dim)
            py::array_t<float> doc_embeddings({static_cast<size_t>(len), static_cast<size_t>(dim)},
                                              {static_cast<size_t>(dim) * sizeof(float), sizeof(float)},
                                              const_cast<float*>(data));

            _doc->addDocument(doc_embeddings, std::to_string(id));
        }
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        std::priority_queue<std::pair<float, int>> result;
        if (!_doc || q_len <= 0)
            return result;

        py::array_t<float> query_embeddings({static_cast<size_t>(q_len), static_cast<size_t>(dim)},
                                            {static_cast<size_t>(dim) * sizeof(float), sizeof(float)},
                                            const_cast<float*>(q_data));

        int num_to_rerank = std::max(k, ef);
        auto ids = _doc->query(query_embeddings, static_cast<uint32_t>(k), static_cast<uint32_t>(num_to_rerank));

        std::vector<uint32_t> internal_ids;
        internal_ids.reserve(ids.size());
        for (const auto& s : ids) {
            try {
                internal_ids.push_back(static_cast<uint32_t>(std::stoul(s)));
            } catch (...) {
            }
        }

        auto scores = _doc->getScores(query_embeddings, internal_ids);
        for (size_t i = 0; i < internal_ids.size() && i < scores.size(); ++i) {
            result.emplace(-scores[i], static_cast<int>(internal_ids[i]));
            if (static_cast<int>(result.size()) > k)
                result.pop();
        }

        return result;
    }
};

}; // namespace vss