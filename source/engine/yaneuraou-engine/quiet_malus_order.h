#pragma once
#include <cassert>
#include <cstdint>

// Experiment 148. Symmetric truncation cancels every rounding residual in
// pairs: delta(j) == -delta(N-1-j). Thus sum == N*m without redistribution.
// int64 return also makes the allocator safe for every signed int input.
namespace QuietMalus148 {
inline int64_t assigned(int m, int n, int j, int lambda_q8) {
    assert(n > 0 && j >= 0 && j < n && lambda_q8 >= 0 && lambda_q8 <= 64);
    if (n == 1 || lambda_q8 == 0) return m;
    return int64_t(m) + int64_t(m) * lambda_q8 * (n - 1 - 2 * j)
                           / (int64_t(256) * (n - 1));
}
}
