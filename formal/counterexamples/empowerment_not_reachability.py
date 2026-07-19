"""Finite Z3 counterexample: action influence does not imply arbitrary reachability.

The agent has two actions with distinguishable consequences, so its one-step action
channel has two reachable outcomes. A separately selected correction target remains
unreachable. This does not formalize the full Shannon-capacity definition; it checks
the logical error in inferring target reachability from action influence alone.
"""

from z3 import Int, Solver, sat


def main() -> None:
    next_if_action_0 = Int("next_if_action_0")
    next_if_action_1 = Int("next_if_action_1")
    correction_target = Int("correction_target")

    solver = Solver()
    solver.add(next_if_action_0 == 0)
    solver.add(next_if_action_1 == 1)
    solver.add(next_if_action_0 != next_if_action_1)  # nontrivial action influence
    solver.add(correction_target == 2)
    solver.add(correction_target != next_if_action_0)
    solver.add(correction_target != next_if_action_1)

    result = solver.check()
    if result != sat:
        raise AssertionError(f"Expected a counterexample, got {result}")

    model = solver.model()
    print("SAT: distinguishable action outcomes coexist with an unreachable target")
    print(model)


if __name__ == "__main__":
    main()
