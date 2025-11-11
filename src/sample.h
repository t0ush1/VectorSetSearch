#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

size_t combination(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k > n - k) {
        return combination(n, n - k);
    }
    if (k == 0) {
        return 1;
    }
    size_t result = 1;
    for (int i = 1; i <= k; ++i) {
        result = result * (n - i + 1) / i;
    }
    return result;
}

std::vector<std::set<int>> all_subsets(int n, int k) {
    std::vector<std::set<int>> result;
    std::vector<int> combination(k);
    std::iota(combination.begin(), combination.end(), 0);
    while (true) {
        result.emplace_back(combination.begin(), combination.end());
        int i = k - 1;
        while (i >= 0 && combination[i] == n - k + i) {
            i--;
        }
        if (i < 0) {
            break;
        }
        combination[i]++;
        for (int j = i + 1; j < k; j++) {
            combination[j] = combination[j - 1] + 1;
        }
    }
    return result;
}

std::set<std::set<int>> sample_subsets(int n, int k, int gamma, std::default_random_engine& sample_generator) {
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);

    std::set<std::set<int>> subsets;
    while (subsets.size() < gamma) {
        std::shuffle(base.begin(), base.end(), sample_generator);
        subsets.insert(std::set<int>(base.begin(), base.begin() + k));
    }

    return subsets;
}