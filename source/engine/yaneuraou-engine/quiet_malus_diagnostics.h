#pragma once
#if defined(SEARCH_QUIET_MALUS_DIAGNOSTICS)
#include <array>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <type_traits>
namespace QuietMalus148 {
struct Delta {
    uint64_t count = 0, clipped = 0, low = 0, high = 0;
    int64_t requested = 0, actual = 0, abs_actual = 0;
    void add(int input, int before, int after, int limit) {
        ++count; requested += input; actual += after-before;
        abs_actual += std::abs(after-before);
        clipped += input < -limit || input > limit;
        low += after == -limit; high += after == limit;
    }
};
struct Diagnostics {
    std::array<Delta, 4> tables{}; // main, low-ply, continuation, pawn
    std::array<Delta, 5> kinds{}; // overlapping categories, not additive
    std::array<uint64_t, 5> ranks{}, moves{};
    std::array<double,5> multiplier_sum{};
    std::array<std::array<int,5>,64> samples{};
    int sample_count=0;
    std::array<int64_t, 5> assigned_sum{}, original_sum{};
    std::array<uint64_t, 7> histogram{};
    uint64_t events=0, quiet_moves=0, cutoffs=0, first_cutoffs=0;
    uint64_t cutoff_moves=0, cutoff_quiets=0, quiet_cutoffs=0, quiet_cutoff_ranks=0;
    uint64_t lmr=0, failhigh=0, research=0, history_pruned=0, see_pruned=0;
    uint64_t sequence=1469598103934665603ULL, deltas=1469598103934665603ULL;
    int table=-1, categories=0;
    void hash(uint64_t& h, int x) { h ^= uint32_t(x); h *= 1099511628211ULL; }
    void update(int input, int before, int after, int limit) {
        hash(sequence,input); hash(sequence,limit);
        hash(deltas,before); hash(deltas,after);
        if(table<0) return;
        tables[table].add(input,before,after,limit);
        if(sample_count<64) samples[sample_count++]={table,input,before,after,limit};
        for(int k=0;k<5;++k) if(categories & (1<<k)) kinds[k].add(input,before,after,limit);
    }
    void print() const {
        std::cout << "info string QM148 search " << events << ' ' << quiet_moves << ' '
          << cutoffs << ' ' << first_cutoffs << ' ' << cutoff_moves << ' ' << cutoff_quiets
          << ' ' << quiet_cutoffs << ' ' << quiet_cutoff_ranks << ' ' << lmr << ' '
          << failhigh << ' ' << research << ' ' << history_pruned << ' ' << see_pruned
          << ' ' << sequence << ' ' << deltas << '\n';
        for(int k=0;k<7;++k) std::cout << "info string QM148 N " << k << ' ' << histogram[k] << '\n';
        auto emit=[](const char* label,int k,const Delta& d) {
            std::cout << "info string QM148 " << label << ' ' << k << ' ' << d.count
              << ' ' << d.requested << ' ' << d.actual << ' ' << d.abs_actual << ' '
              << d.clipped << ' ' << d.low << ' ' << d.high << '\n';
        };
        for(int k=0;k<4;++k) emit("table",k,tables[k]);
        for(int k=0;k<5;++k) {
            emit("kind",k,kinds[k]);
            std::cout << "info string QM148 rank " << k << ' ' << ranks[k] << ' '
                      << assigned_sum[k] << ' ' << original_sum[k] << '\n';
            std::cout << "info string QM148 moves " << k << ' ' << moves[k] << '\n';
            std::cout << "info string QM148 multiplier " << k << ' ' << multiplier_sum[k] << '\n';
        }
        for(int i=0;i<sample_count;++i) {
            std::cout << "info string QM148 sample";
            for(auto x:samples[i]) std::cout << ' ' << x;
            std::cout << '\n';
        }
    }
};
inline thread_local Diagnostics diagnostics;
template<class T> void hash_table(uint64_t& hash,const T& table) {
    if constexpr(std::is_convertible_v<T,int>) {
        hash ^= uint32_t(int(table)); hash *= 1099511628211ULL;
    } else for(const auto& child:table) hash_table(hash,child);
}
}
#endif
