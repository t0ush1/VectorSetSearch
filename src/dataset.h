#pragma once
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vss {

namespace fs = std::filesystem;

template<typename... Args>
void cerr_if(bool condition, Args&&... args) {
    if (condition) {
        (std::cerr << ... << std::forward<Args>(args)) << std::endl;
        std::exit(-1);
    }
}

class Dataset {
public:
    int dim;
    int size;
    float* data;

    Dataset(int dim, fs::path path) : dim(dim) {
        std::ifstream in(path, std::ios::binary);
        cerr_if(!in.is_open(), "Fail to open vector file: ", path);

        int file_dim = 0;
        in.read((char*)&file_dim, 4);
        cerr_if(file_dim != dim, "Dimension mismatch: ", file_dim, ", ", dim);

        in.seekg(0, std::ios::end);
        size_t fsize = (size_t)in.tellg();
        in.seekg(0, std::ios::beg);

        size = fsize / ((dim + 1) * 4);
        data = new float[size * dim];

        for (int i = 0; i < size; i++) {
            in.seekg(4, std::ios::cur);
            in.read((char*)(data + i * dim), dim * 4);
        }
        in.close();
    }

    ~Dataset() { delete[] data; }
};

class VSSDataset : public Dataset {
public:
    int set_num;
    std::vector<const float*> set_data;
    std::vector<int> set_len;

    VSSDataset(int dim, fs::path vector_path, fs::path length_path) : Dataset(dim, vector_path) {
        std::ifstream in(length_path, std::ios::binary);
        cerr_if(!in.is_open(), "Fail to open length file: ", length_path);

        in.seekg(0, std::ios::end);
        size_t fsize = (size_t)in.tellg();
        in.seekg(0, std::ios::beg);

        set_num = fsize / 4;
        set_data.resize(set_num);
        set_len.resize(set_num);

        in.read((char*)set_len.data(), set_num * 4);
        in.close();

        set_data[0] = data;
        for (int i = 1; i < set_num; i++) {
            set_data[i] = set_data[i - 1] + set_len[i - 1] * dim;
        }
    }

    std::pair<const float*, int> get_data_len(int set_id) const { return {set_data[set_id], set_len[set_id]}; }
};

std::vector<std::unordered_set<int>> read_groundtruth(fs::path path) {
    std::ifstream in(path, std::ios::binary);
    cerr_if(!in.is_open(), "Fail to open groundtruth file: ", path);

    int k = 0;
    in.read((char*)&k, 4);

    in.seekg(0, std::ios::end);
    size_t fsize = (size_t)in.tellg();
    in.seekg(0, std::ios::beg);

    int size = fsize / ((k + 1) * 4);
    std::vector<std::unordered_set<int>> gts;

    for (int i = 0; i < size; i++) {
        std::vector<int> tmp(k);
        in.seekg(4, std::ios::cur);
        in.read((char*)tmp.data(), k * 4);
        gts.emplace_back(tmp.begin(), tmp.end());
    }
    return gts;
}

} // namespace vss