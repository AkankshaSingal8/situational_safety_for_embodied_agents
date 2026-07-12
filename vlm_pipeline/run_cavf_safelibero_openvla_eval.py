"""Run the CAVF static POC through the unmodified OpenVLA evaluator."""

from __future__ import annotations

import run_safelibero_openvla_oft_eval as evaluator

from cavf_static_filter import CAVFStaticSafetyFilter


# The base evaluator imports this symbol at module scope. Rebinding it creates
# an additive POC entry point without editing the baseline evaluator.
evaluator.ContextualPredictiveSafetyFilter = CAVFStaticSafetyFilter


if __name__ == "__main__":
    evaluator.eval_safelibero()
