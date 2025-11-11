#pragma once
#include <chrono>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "dataset.h"
#include "index.h"
#include "space.h"

#include "baselines/brute_force.h"
#include "baselines/hnsw_pointwise.h"
#include "baselines/ivfpq_pointwise.h"
#include "baselines/multi_hnsw_index.h"
#include "baselines/single_hnsw_index.h"

#include "semi_hnsw_index.h"

namespace vss {
class VSSRunner {
public:
    struct BuildRecord {
        size_t time;
        size_t memory;
    };

    struct QueryRecord {
        int ef;
        int q_num;
        size_t time;
        int hit;
        int total;
        std::vector<std::pair<std::string, long>> metrics;
    };

    std::string log_time;

    int dim;
    std::string metric_name;
    std::string data_dir;
    std::string index_name;

    VSSDataset* base_dataset;
    VSSDataset* query_dataset;
    std::vector<std::unordered_set<int>> groundtruth;

    VSSSpace* space;
    VSSIndex* index;
    std::vector<int> efs;

    VSSRunner(int dim, std::string metric_name, std::string data_dir, std::string index_name)
        : dim(dim), metric_name(metric_name), data_dir(data_dir), index_name(index_name) {
        fs::path data_path = fs::path("../datasets") / data_dir;
        base_dataset = new VSSDataset(dim, data_path / "base.fvecs", data_path / "base.lens");
        query_dataset = new VSSDataset(dim, data_path / "query.fvecs", data_path / "query.lens");
        groundtruth = read_groundtruth(data_path / ("groundtruth-" + metric_name + ".ivecs"));
        // groundtruth = read_groundtruth(data_path / "groundtruth.ivecs");

        if (metric_name == "maxsim") {
            space = new MaxSimSpace(dim);
        } else if (metric_name == "dtw") {
            space = new DTWSpace(dim);
        } else if (metric_name == "sdtw") {
            space = new SDTWSpace(dim);
        } else {
            std::cerr << "Unknown similarity metric: " << metric_name << std::endl;
            std::exit(-1);
        }

        if (index_name == "brute_force") {
            index = new BruteForceIndex(dim, space);
            efs = {0};
        } else if (index_name == "hnsw") {
            index = new HNSWPointwiseIndex(dim, space, 16, 200);
            efs = {10, 20, 40, 60, 80, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000};
        } else if (index_name == "ivfpq") {
            index = new IVFPQPointwiseIndex(dim, space, 100, 8, 8);
            efs = {10, 20, 50, 100, 200, 500};
        } else if (index_name == "single_hnsw") {
            index = new SingleHNSWIndex(dim, space, 16, 200);
            efs = {10, 20, 40, 60, 80, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000};
        } else if (index_name == "multi_hnsw") {
            index = new MultiHNSWIndex(dim, space, 16, 200);
            efs = {10, 20, 30, 40, 50, 60, 80, 100, 200};
        } else if (index_name == "semi_hnsw") {
            index = new SemiHNSWIndex(dim, space, 16, 200, 2, 1);
            efs = {10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500};
        } else {
            std::cerr << "Unknown index: " << index_name << std::endl;
            std::exit(-1);
        }

        std::time_t t = std::time(nullptr);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%y%m%d-%H%M%S", std::localtime(&t));
        log_time = buf;
    }

    ~VSSRunner() {
        delete base_dataset;
        delete query_dataset;
        delete space;
        delete index;
    }

    void run_build() {
        BuildRecord record;
        auto begin = std::chrono::high_resolution_clock::now();
        index->build(base_dataset);
        auto end = std::chrono::high_resolution_clock::now();
        record.time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "Build Time: " << record.time << " us" << std::endl;
        std::cout << std::endl;

        // TODO 查看内存开销

        save_build_record(record);
    }

    void run_search() {
        std::vector<QueryRecord> records;
        int k = groundtruth[0].size();

        for (int i = 0; i < efs.size(); i++) {
            QueryRecord r = run_search_once(k, efs[i]);
            records.push_back(r);

            std::cout << "EF: " << r.ef << std::endl;
            std::cout << "Time: " << r.time << " us, " << r.time / r.q_num << " us" << std::endl;
            std::cout << "Recall: " << r.hit << "/" << r.total << "=" << r.hit * 1.0 / r.total << std::endl;
            for (const auto& [name, value] : r.metrics) {
                std::cout << "Metric (" << name << "): " << value << ", " << value / r.q_num << std::endl;
            }
            std::cout << std::endl;

            if (r.hit >= 0.999 * r.total) {
                break;
            }
        }

        save_query_records(records);
    }

    QueryRecord run_search_once(int k, int ef) {
        QueryRecord record = {};
        record.ef = ef;
        index->reset_metrics();
        record.metrics = index->get_metrics();

        for (int i = 0; i < query_dataset->seq_num; i++) {
            auto [q_data, q_len] = query_dataset->get_data_len(i);

            index->reset_metrics();

            auto begin = std::chrono::high_resolution_clock::now();
            auto result = index->search(q_data, q_len, k, ef);
            auto end = std::chrono::high_resolution_clock::now();
            record.time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            auto metrics = index->get_metrics();
            for (int i = 0; i < metrics.size(); i++) {
                record.metrics[i].second += metrics[i].second;
            }

            assert(result.size() <= k);
            std::unordered_set<int> unique;
            while (result.size() > 0) {
                unique.insert(result.top().second);
                result.pop();
            }

            for (int id : unique) {
                if (groundtruth[i].find(id) != groundtruth[i].end()) {
                    record.hit++;
                }
            }

            record.total += groundtruth[i].size();
            record.q_num++;
        }

        return record;
    }

    void save_build_record(BuildRecord record) {
        std::string log_name = index_name + "-build-" + log_time + ".log";
        fs::path log_path = fs::path("../log") / data_dir / metric_name / log_name;
        fs::create_directories(log_path.parent_path());

        std::ofstream ofs(log_path);
        cerr_if(!ofs.is_open(), "Failed to open " + log_name);

        ofs << "time: " << record.time << std::endl;
        ofs << "memory: " << record.memory << std::endl;
        ofs.close();
        std::cout << "Build record written to " << log_path << std::endl;
    }

    void save_query_records(std::vector<QueryRecord>& records) {
        std::string csv_name = index_name + "-query-" + log_time + ".csv";
        fs::path csv_path = fs::path("../log") / data_dir / metric_name / csv_name;
        fs::create_directories(csv_path.parent_path());

        std::ofstream ofs(csv_path);
        cerr_if(!ofs.is_open(), "Failed to open " + csv_name);

        assert(!records.empty());
        ofs << "ef,time,hit,total,q_num";
        for (const auto& m : records[0].metrics) {
            ofs << "," << m.first;
        }
        ofs << std::endl;

        for (const auto& r : records) {
            ofs << r.ef << "," << r.time << "," << r.hit << "," << r.total << "," << r.q_num;
            for (const auto& m : r.metrics) {
                ofs << "," << m.second;
            }
            ofs << std::endl;
        }

        ofs.close();
        std::cout << "Query records written to " << csv_path << std::endl;
    }
};

} // namespace vss