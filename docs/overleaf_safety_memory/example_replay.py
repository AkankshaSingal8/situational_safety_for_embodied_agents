"""Deterministic worked example, not a learned mechanism or robot controller."""

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True)
class Mechanism:
    harm_exposed: float = 0.80
    harm_unexposed: float = 0.01
    cleaning_residual: float = 0.03

    def risk(self, exposure: float) -> float:
        if not 0 <= exposure <= 1:
            raise ValueError("Exposure must be a probability")
        return exposure * self.harm_exposed + (1 - exposure) * self.harm_unexposed

    def after_cleaning(self, exposure: float, verified: bool) -> float:
        return exposure * self.cleaning_residual if verified else exposure


def replay() -> None:
    model = Mechanism()
    exposure, threshold = 0.90, 0.05
    # Planning assumes a validated recovery; execution must verify it separately.
    plans = [
        ("direct_push", model.risk(exposure), 0.0),
        ("wait_then_push", model.risk(exposure), 0.5),
        ("clean_verify_then_push", model.risk(model.after_cleaning(exposure, True)), 1.0),
        ("verified_replacement", model.risk(0.01), 2.0),
    ]
    for name, risk, effort in plans:
        print(f"{name:24s} risk={risk:.5f} effort={effort:g} acceptable={risk <= threshold}")
    feasible = [plan for plan in plans if plan[1] <= threshold]
    chosen = min(feasible, key=lambda plan: plan[2]) if feasible else None
    assert chosen and chosen[0] == "clean_verify_then_push"
    assert isclose(plans[0][1], 0.721)
    assert isclose(chosen[1], 0.03133)
    assert isclose(plans[3][1], 0.0179)
    assert model.after_cleaning(exposure, False) == exposure
    print(f"Selected: {chosen[0]}")

    updated = model.after_cleaning(exposure, verified=True)
    assert model.risk(updated) <= threshold
    record = {
        "proposed": "direct_push",
        "executed": chosen[0],
        "counterfactual_outcome_of_proposed": None,
        "executed_outcome_source": "stipulated_example_feedback",
    }
    assert record["counterfactual_outcome_of_proposed"] is None
    print(f"After verified cleaning: exposure={updated:.5f}; direct push accepted")
    clean_history_exposure = 0.01
    assert model.risk(clean_history_exposure) <= threshold
    print("Same current image, clean history: direct push accepted")
    print("Failed verification: old exposure retained; direct push rejected")
    print("All worked-example assertions passed. No robot or learning model was run.")


if __name__ == "__main__":
    replay()
