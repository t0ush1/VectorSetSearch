#include "runner.h"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dim> <data_dir> <index_name>\n";
        return 1;
    }

    vss::VSSRunner runner(std::stoi(argv[1]), argv[2], argv[3]);
    runner.run();

    return 0;
}