"""Run the CAVF static POC through the unmodified pi0.5 evaluator."""

from __future__ import annotations

import logging

import run_safelibero_pi05_eval as evaluator

from cavf_static_filter import CAVFStaticSafetyFilter


evaluator.ContextualPredictiveSafetyFilter = CAVFStaticSafetyFilter


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    evaluator.run_eval(evaluator.parse_args())
