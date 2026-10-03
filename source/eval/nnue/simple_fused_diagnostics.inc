// Experiment 153, test-only. Included inside NNUE's anonymous namespace.
void SimpleProfile153(Position& pos, std::uint64_t repeats) {
    if (!repeats) repeats = 100000;
    alignas(kCacheLineSize) TransformedFeatureType transformed[1536];
    alignas(kCacheLineSize) Network::Buffer b{};
    const auto& n = *network[SimpleBucket::selected(pos)];
    feature_transformer->Transform(pos, transformed, true);
    n.Propagate(transformed, reinterpret_cast<char*>(&b));
#if defined(USE_AVX2) && !defined(USE_AVX512)
    alignas(kCacheLineSize) std::uint8_t captured[1536];
    alignas(kCacheLineSize) Network::Buffer verified{};
    feature_transformer->TransformFc0<true>(pos, n.fc_0, verified.fc0, false, captured);
    n.PropagateFromFc0(reinterpret_cast<char*>(&verified));
    std::cout << "FIXED153," << SimpleBucket::selected(pos) << ','
              << (std::memcmp(transformed, captured,1536)!=0) << ','
              << (std::memcmp(b.fc0, verified.fc0,64)!=0) << ','
              << b.fc2[0] << ',' << verified.fc2[0] << ','
              << b.fc2[0]/FV_SCALE << ',' << verified.fc2[0]/FV_SCALE << std::endl;
#endif
    volatile std::uint64_t sink = 0;
    auto measure = [&](const char* stage, auto operation) {
        for (int i = 0; i < 1000; ++i) operation();
        const auto start = std::chrono::steady_clock::now();
        for (std::uint64_t i = 0; i < repeats; ++i) {
            operation();
            sink = sink + static_cast<std::uint32_t>(b.fc0[0])
                 + static_cast<std::uint32_t>(b.fc2[0]) + transformed[i % 1536];
        }
        const double ns = std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - start).count() / repeats;
        std::cout << "PROFILE153," << stage << ',' << repeats << ',' << ns << std::endl;
    };
    measure("ensure_computed", [&] { feature_transformer->EnsureAccumulator(pos); });
    measure("transform", [&] { feature_transformer->Transform(pos, transformed, false); });
    measure("fc0", [&] { n.fc_0.Propagate(transformed, b.fc0); });
    measure("transform_fc0", [&] {
        feature_transformer->Transform(pos, transformed, false);
        n.fc_0.Propagate(transformed, b.fc0);
    });
    measure("remaining_dense", [&] {
        n.ac_0.Propagate(b.fc0, b.ac0);
        n.ac_sqr_0.Propagate(b.fc0, b.concat);
        std::memcpy(b.concat + 15, b.ac0, 15);
        n.fc_1.Propagate(b.concat, b.fc1);
        n.ac_1.Propagate(b.fc1, b.ac1);
        n.fc_2.Propagate(b.ac1, b.fc2);
        b.fc2[0] += b.fc0[15];
    });
    measure("materialized_evaluate", [&] {
        feature_transformer->Transform(pos, transformed, false);
        n.Propagate(transformed, reinterpret_cast<char*>(&b));
    });
    measure("refresh_evaluate", [&] {
        feature_transformer->Transform(pos, transformed, true);
        n.Propagate(transformed, reinterpret_cast<char*>(&b));
    });
#if defined(USE_AVX2) && !defined(USE_AVX512)
    measure("fused_transform_fc0", [&] { feature_transformer->TransformFc0(pos, n.fc_0, b.fc0); });
    measure("fused_materialized_evaluate", [&] {
        feature_transformer->TransformFc0(pos, n.fc_0, b.fc0);
        n.PropagateFromFc0(reinterpret_cast<char*>(&b));
    });
#endif
    std::cout << "PROFILE153 sink=" << sink << std::endl;
    MoveList<LEGAL_ALL> legal(pos);
    if (legal.size()) {
        StateInfo child;
        double update_ns = 0, clock_floor = 0;
        const auto begin_all = std::chrono::steady_clock::now();
        for (std::uint64_t i = 0; i < repeats; ++i) {
            const Move m = legal.begin()[i % legal.size()];
            pos.do_move(m, child);
            const auto begin = std::chrono::steady_clock::now();
            feature_transformer->EnsureAccumulator(pos);
            update_ns += std::chrono::duration<double,std::nano>(
                std::chrono::steady_clock::now()-begin).count();
            pos.undo_move(m);
        }
        const double all_ns=std::chrono::duration<double,std::nano>(
            std::chrono::steady_clock::now()-begin_all).count()/repeats;
        for (std::uint64_t i=0;i<repeats;++i) {
            const auto begin=std::chrono::steady_clock::now();
            clock_floor+=std::chrono::duration<double,std::nano>(
                std::chrono::steady_clock::now()-begin).count();
        }
        std::cout << "UPDATE153," << repeats << ',' << update_ns/repeats << ','
                  << clock_floor/repeats << ',' << all_ns << std::endl;
    }
}

#if defined(USE_AVX2) && !defined(USE_AVX512)
void SimpleDelayed153(Position& pos) {
    const auto original = pos.sfen();
    StateInfo* original_state = pos.state();
    auto states = std::make_unique<StateInfo[]>(16);
    auto compare = [&] {
        alignas(kCacheLineSize) std::uint8_t before[1536], scratch[1536];
        alignas(kCacheLineSize) Network::Buffer a{}, b{};
        feature_transformer->Transform(pos, before, false);
        alignas(kCacheLineSize) std::int16_t accumulated[2][1536];
        std::memcpy(accumulated, pos.state()->accumulator.accumulation, sizeof(accumulated));
        const auto& n = *network[SimpleBucket::selected(pos)];
        feature_transformer->TransformFc0(pos, n.fc_0, a.fc0);
        n.PropagateFromFc0(reinterpret_cast<char*>(&a));
        feature_transformer->TestRefreshAccumulatorFromScratch(pos);
        feature_transformer->Transform(pos, scratch, false);
        n.Propagate(scratch, reinterpret_cast<char*>(&b));
        return std::memcmp(accumulated, pos.state()->accumulator.accumulation, sizeof(accumulated)) != 0
            || std::memcmp(before,scratch,1536) != 0
            || std::memcmp(a.fc0,b.fc0,64) != 0 || a.fc2[0] != b.fc2[0];
    };
    const char* names[] = {"immediate", "one_child_do_undo", "multiple_child_do_undo", "deferred_leaf_parent", "finny_restore"};
    for (int c = 0; c < 5; ++c) {
        std::uint64_t tested = 0, mismatches = 0;
        for (const char* sfen : {
             "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
             "4k4/9/9/9/9/9/3SG4/9/4K4 b - 1"}) {
            pos.set(sfen, &states[0]);
            feature_transformer->EnsureAccumulator(pos);
            std::vector<Move> path;
            const int depth = c == 3 ? 5 : 1;
            for (int d = 0; d < depth; ++d) {
                MoveList<LEGAL_ALL> legal(pos);
                const Move m = legal.begin()[0];
                pos.do_move(m, states[d+1]); path.push_back(m);
            }
            if (c == 1 || c == 2) {
                MoveList<LEGAL_ALL> children(pos);
                const unsigned count = c == 1 ? 1 : std::min<unsigned>(3, children.size());
                for (unsigned i = 0; i < count; ++i) {
                    const Move child = children.begin()[i];
                    pos.do_move(child, states[10+i]);
                    pos.undo_move(child);
                }
            }
            if (c == 4) {
                // Warm Finny from this leaf, undo, refresh parent, return to
                // leaf: exercise cache restoration independently of scratch.
                feature_transformer->ForceRefreshAccumulator(pos);
                pos.undo_move(path.back());
                feature_transformer->ForceRefreshAccumulator(pos);
                pos.do_move(path.back(), states[1]);
                feature_transformer->ForceRefreshAccumulator(pos);
            }
            mismatches += compare(); ++tested;
            if (c == 3) {
                pos.undo_move(path.back()); path.pop_back();
                mismatches += compare(); ++tested;
            }
            for (auto i=path.rbegin();i!=path.rend();++i) pos.undo_move(*i);
        }
        std::cout << "DELAYED153," << names[c] << ',' << tested << ',' << mismatches << std::endl;
    }
    pos.set(original, original_state);
}

void SimpleCorrectness153(Position& pos, unsigned games) {
    if (!games) games = 1000;
    PRNG rng(20171128);
    auto states = std::make_unique<StateInfo[]>(MAX_PLY);
    std::vector<Move> moves;
    std::uint64_t tested = 0, mismatch[9]{};
    std::uint64_t captures = 0, promotions = 0, drops = 0, kings = 0;
    auto check = [&] {
        alignas(kCacheLineSize) std::uint8_t legacy[1536], captured[1536];
        alignas(kCacheLineSize) Network::Buffer a{}, b{};
        feature_transformer->Transform(pos, legacy, false);
        for (int bucket = 0; bucket < 9; ++bucket) {
            const auto& n = *network[bucket];
            n.Propagate(legacy, reinterpret_cast<char*>(&a));
            feature_transformer->TransformFc0<true>(pos, n.fc_0, b.fc0, false, captured);
            n.PropagateFromFc0(reinterpret_cast<char*>(&b));
            mismatch[0] += std::memcmp(legacy, captured, 1536) != 0;
            mismatch[1] += std::memcmp(a.fc0, b.fc0, 16*sizeof(std::int32_t)) != 0;
            mismatch[2] += std::memcmp(a.ac0, b.ac0, 15) != 0;
            mismatch[3] += std::memcmp(a.concat, b.concat, 30) != 0;
            mismatch[4] += std::memcmp(a.fc1, b.fc1, 32*sizeof(std::int32_t)) != 0;
            mismatch[5] += std::memcmp(a.ac1, b.ac1, 32) != 0;
            mismatch[6] += a.fc0[15] != b.fc0[15];
            mismatch[7] += (a.fc2[0]-a.fc0[15]) != (b.fc2[0]-b.fc0[15]);
            mismatch[8] += a.fc2[0] != b.fc2[0];
        }
        ++tested;
    };
    check();
    for (unsigned game = 0; game < games; ++game) {
        moves.clear();
        for (int ply = 0; ply < MAX_PLY; ++ply) {
            MoveList<LEGAL_ALL> legal(pos);
            if (!legal.size()) break;
            const Move m = legal.begin()[rng.rand(legal.size())];
            captures += pos.capture(m);
            promotions += m.is_promote();
            drops += m.is_drop();
            kings += !m.is_drop() && type_of(pos.piece_on(m.from_sq())) == KING;
            pos.do_move(m, states[ply]);
            moves.push_back(m);
            check();
        }
        for (auto i = moves.rbegin(); i != moves.rend(); ++i) pos.undo_move(*i);
    }
    // SIMD-boundary/extreme tests, preserving the caller's accumulator.
    auto& acc = pos.state()->accumulator;
    alignas(kCacheLineSize) std::int16_t backup[2][1536];
    std::memcpy(backup, acc.accumulation, sizeof(backup));
    constexpr std::int16_t values[] = {-32768,-1,0,1,126,127,128,253,254,255,32767};
    for (unsigned pattern = 0; pattern < 64; ++pattern) {
        for (Color c : {BLACK, WHITE})
            for (int i = 0; i < 1536; ++i)
                acc.accumulation[c][0][i] = values[rng.rand(11)];
        check();
    }
    std::memcpy(acc.accumulation, backup, sizeof(backup));
    acc.computed_score = false;
    constexpr const char* names[] = {"transformed","fc0","clipped","concat","fc1_pre","fc1_act","shortcut","deep","final"};
    for (int i = 0; i < 9; ++i)
        std::cout << "CORRECT153," << names[i] << ',' << tested << ',' << mismatch[i] << std::endl;
    std::cout << "CORRECT153 coverage captures=" << captures << " promotions=" << promotions
              << " drops=" << drops << " king_moves=" << kings << " extremes=64 all_heads=9" << std::endl;
}

void SimpleMixed153(Position& root, std::uint64_t repeats) {
    if (!repeats) repeats = 100000;
    // 256 naturally encountered positions, mixed accumulator/head working set.
    PRNG rng(153);
    auto chain = std::make_unique<StateInfo[]>(MAX_PLY);
    std::vector<Move> moves;
    std::vector<std::string> sfens{root.sfen()};
    for (int ply = 0; ply < MAX_PLY && sfens.size() < 256; ++ply) {
        MoveList<LEGAL_ALL> legal(root);
        if (!legal.size()) break;
        const Move m = legal.begin()[rng.rand(legal.size())];
        root.do_move(m, chain[ply]); moves.push_back(m);
        sfens.push_back(root.sfen());
    }
    for (auto i = moves.rbegin(); i != moves.rend(); ++i) root.undo_move(*i);
    auto states = std::make_unique<StateInfo[]>(sfens.size());
    std::vector<std::unique_ptr<Position>> positions;
    for (std::size_t i = 0; i < sfens.size(); ++i) {
        positions.push_back(std::make_unique<Position>());
        positions.back()->set(sfens[i], &states[i]);
        feature_transformer->EnsureAccumulator(*positions.back());
    }
    alignas(kCacheLineSize) std::uint8_t transformed[1536];
    alignas(kCacheLineSize) Network::Buffer b{};
    volatile std::uint64_t sink = 0;
    for (int pass = 0; pass < 10; ++pass) {
        for (int ordinal = 0; ordinal < 2; ++ordinal) {
            const bool fused = (pass + ordinal) % 2;
            const auto begin = std::chrono::steady_clock::now();
            for (std::uint64_t i = 0; i < repeats; ++i) {
                const auto& pos = *positions[i % positions.size()];
                const auto& n = *network[SimpleBucket::selected(pos)];
                if (fused) feature_transformer->TransformFc0(pos, n.fc_0, b.fc0);
                else {
                    feature_transformer->Transform(pos, transformed, false);
                    n.fc_0.Propagate(transformed, b.fc0);
                }
                sink = sink + static_cast<std::uint32_t>(b.fc0[i % 16]);
            }
            const double ns = std::chrono::duration<double,std::nano>(
              std::chrono::steady_clock::now()-begin).count()/repeats;
            std::cout << "MIXED153," << pass << ',' << (fused?"fused":"legacy") << ','
                      << positions.size() << ',' << repeats << ',' << ns << std::endl;
        }
    }
    std::cout << "MIXED153 sink=" << sink << std::endl;
}
#endif
