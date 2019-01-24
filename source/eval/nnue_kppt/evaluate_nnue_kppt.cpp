// NNUE-KPPT評価関数の計算に関するコード

#include "../../shogi.h"

#if defined(EVAL_NNUE_KPPT)

#include "evaluate_nnue_kppt.h"

namespace Eval {

	void init() {
		Eval::NNUE::init();
		Eval::KPPT::init();
	}

	void load_eval() {
		Eval::NNUE::load_eval();
		Eval::KPPT::load_eval();
	}

	Value eval_mix(Value value_nnue, Value value_kppt) {
		double kpptRatio = Options["EvalKpptRatio"] / 100.0f;
		return Value(int((double)value_nnue * (1 - kpptRatio) + (double)value_kppt * kpptRatio));
	}

	Value compute_eval(const Position& pos) {
		Value value_nnue = Eval::NNUE::compute_eval(pos);
		Value value_kppt = Eval::KPPT::compute_eval(pos);
		return eval_mix(value_nnue, value_kppt);
	}

	Value evaluate(const Position& pos) {
		Value value_nnue = Eval::NNUE::evaluate(pos);
		Value value_kppt = Eval::KPPT::evaluate(pos);
		return eval_mix(value_nnue, value_kppt);
	}

	void evaluate_with_no_return(const Position& pos) {
		Eval::NNUE::evaluate_with_no_return(pos);
		Eval::KPPT::evaluate_with_no_return(pos);
	}

	void print_eval_stat(Position& pos) {
		Value value_nnue = Eval::NNUE::evaluate(pos);
		Value value_kppt = Eval::KPPT::evaluate(pos);
		Value value_mix = eval_mix(value_nnue, value_kppt);

		int ratio_kppt = (int)Options["EvalKpptRatio"];
		int ratio_nnue = 100 - ratio_kppt;

		std::cout << "eval_NNUE = " << value_nnue << std::endl;
		std::cout << "eval_KPPT = " << value_kppt << std::endl;
		std::cout << "eval_MIX  = " << value_mix << std::endl;
		std::cout << "(NNUE:" << ratio_nnue << "%, KPPT:" << ratio_kppt << "%)" << std::endl;
	}

	void prefetch_evalhash(const Key key) {
		Eval::NNUE::prefetch_evalhash(key);
		Eval::KPPT::prefetch_evalhash(key);
	}

	u64 calc_check_sum() {
		return Eval::KPPT::calc_check_sum();
	}

	void foreach_eval_param(std::function<void(s32, s32)>f, int type) {
		Eval::KPPT::foreach_eval_param(f, type);
	}

}  // namespace Eval

#endif  // defined(EVAL_NNUE_KPPT)
