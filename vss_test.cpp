#include "runner.h"
using namespace vss;

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <dim> <similarity_metric> <data_dir> <index_name>\n";
        return 1;
    }

    VSSRunner runner(std::stoi(argv[1]), argv[2], argv[3], argv[4]);
    runner.run();

    return 0;
}