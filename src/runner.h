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

#include "paral_hnsw_index.h"

namespace vss {
class VSSRunner {
public:
    struct BuildRecord {
        size_t time;
        size_t peak_memory;
        size_t current_memory;
        std::vector<std::pair<std::string, long>> stats;
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
            efs = {10, 20, 40, 60, 80, 100, 200, 500, 1000, 1500, 2000};
        } else if (index_name == "ivfpq") {
            index = new IVFPQPointwiseIndex(dim, space, 100, 8, 8);
            efs = {10, 20, 50, 100, 200, 500};
        } else if (index_name == "single_hnsw") {
            index = new SingleHNSWIndex(dim, space, 16, 200);
            efs = {10, 20, 30, 40, 50, 60, 80, 100, 200};
        } else if (index_name == "multi_hnsw") {
            index = new MultiHNSWIndex(dim, space, 16, 200);
            efs = {10, 20, 30, 40, 50, 60, 80, 100, 200};
        } else if (index_name == "paral_hnsw") {
            index = new ParalHNSWIndex(dim, space, 16, 200);
            // efs = {10, 20, 30, 40, 50, 60, 80, 100, 200};
            efs = {5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200};
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

    void run() {
        BuildRecord build_record;
        run_build(build_record);
        record_memory_usage(build_record);

        std::vector<QueryRecord> query_records(efs.size());
        run_search(query_records);

        save_build_record(build_record);
        save_query_records(query_records);
    }

    void run_build(BuildRecord& record) {
        auto begin = std::chrono::high_resolution_clock::now();
        index->build(base_dataset);
        auto end = std::chrono::high_resolution_clock::now();
        record.time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "Build Time: " << record.time << " us" << std::endl;
        std::cout << std::endl;

        record.stats = index->get_stats();
        for (const auto& [name, value] : record.stats) {
            std::cout << "Stat (" << name << "): " << value << std::endl;
        }
    }

    void run_search(std::vector<QueryRecord>& records) {
        int k = groundtruth[0].size();

        for (int i = 0; i < efs.size(); i++) {
            QueryRecord& r = records[i];
            run_search_once(r, k, efs[i]);

            float recall = r.hit * 1.0 / r.total;
            std::cout << "EF: " << r.ef << std::endl;
            std::cout << "Time: " << r.time << " us, " << r.time / r.q_num << " us" << std::endl;
            std::cout << "Recall: " << r.hit << "/" << r.total << "=" << recall << std::endl;
            for (const auto& [name, value] : r.metrics) {
                std::cout << "Metric (" << name << "): " << value << ", " << value / r.q_num << std::endl;
            }
            std::cout << std::endl;

            if (recall > 0.999) {
                break;
            }
        }
    }

    QueryRecord run_search_once(QueryRecord& record, int k, int ef) {
        record.ef = ef;
        index->reset_metrics();
        record.metrics = index->get_metrics();

        for (int i = 0; i < query_dataset->seq_num; i++) {
            auto [q_data, q_len] = query_dataset->get_data_len(i);

            index->reset_metrics();
            index->prepare(q_data, q_len, k, ef);

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

    void record_memory_usage(BuildRecord& record) {
        std::ifstream file_stream("/proc/self/status");
        std::string line;

        auto record_mem_field = [&](std::string key, std::string label, size_t& value) {
            if (line.find(key) != std::string::npos) {
                size_t begin = line.find_first_of("0123456789");
                size_t end = line.find_last_of("0123456789");
                value = std::stoull(line.substr(begin, end - begin + 1)) / 1024;
                std::cout << label << ": " << value << " MB" << std::endl;
            }
        };

        while (std::getline(file_stream, line)) {
            record_mem_field("VmHWM", "Peak Physical Memory Usage", record.peak_memory);
            record_mem_field("VmRSS", "Current Physical Memory Usage", record.current_memory);
            // print_mem_field("VmPeak", "Peak Virtual Memory Usage");
            // print_mem_field("VmSize", "Current Virtual Memory Usage");
            // print_mem_field("VmData", "Data Segment Virtual Memory Usage");
        }

        std::cout << std::endl;
    }

    void save_build_record(BuildRecord record) {
        std::string log_name = index_name + "-build-" + log_time + ".log";
        fs::path log_path = fs::path("../log") / data_dir / metric_name / log_name;
        fs::create_directories(log_path.parent_path());

        std::ofstream ofs(log_path);
        cerr_if(!ofs.is_open(), "Failed to open " + log_name);

        ofs << "time: " << record.time << " us" << std::endl;
        ofs << "peak memory: " << record.peak_memory << " MB" << std::endl;
        ofs << "current memory: " << record.current_memory << " MB" << std::endl;
        for (const auto& [name, value] : record.stats) {
            ofs << name << ": " << value << std::endl;
        }

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
            if (r.q_num == 0) {
                break;
            }
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