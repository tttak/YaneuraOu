// NNUE-KPPT評価関数で用いるheader

#ifndef _EVALUATE_NNUE_KPPT_H_
#define _EVALUATE_NNUE_KPPT_H_

#if defined(EVAL_NNUE_KPPT)

namespace Eval {

	namespace NNUE {
		void init();
		void load_eval();
		Value evaluate(const Position& pos);
		void evaluate_with_no_return(const Position& pos);
		Value compute_eval(const Position& pos);
		void prefetch_evalhash(const Key key);
	}

	namespace KPPT {
		void init();
		void load_eval();
		Value evaluate(const Position& pos);
		void evaluate_with_no_return(const Position& pos);
		Value compute_eval(const Position& pos);
		void prefetch_evalhash(const Key key);

		u64 calc_check_sum();
		void foreach_eval_param(std::function<void(s32, s32)>f, int type);
	}


}  // namespace Eval

#endif  // defined(EVAL_NNUE_KPPT)

#endif
