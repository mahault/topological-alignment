import TopologicalAlignment.Definitions

namespace TopologicalAlignment

/-- A deterministic policy-controlled transition system. Finite instances are the
first intended models, but the invariance result does not require finiteness. -/
structure PolicySystem (State Action : Type) where
  step : State -> Action -> State
  policy : State -> Action

/-- The phenotype-relative viability region. Its empirical interpretation must be
justified independently of the theorem. -/
structure Phenotype (State : Type) where
  viable : State -> Prop

/-- The closed-loop successor produced by the system's policy. -/
def PolicySystem.next {State Action : Type}
    (system : PolicySystem State Action) (state : State) : State :=
  system.step state (system.policy state)

/-- A state remains viable for `horizon` closed-loop transitions. The initial state
is included, so horizon zero requires current viability. -/
def ViableFor {State Action : Type}
    (phenotype : Phenotype State) (system : PolicySystem State Action) :
    Nat -> State -> Prop
  | 0, state => phenotype.viable state
  | Nat.succ horizon, state =>
      phenotype.viable state /\
      ViableFor phenotype system horizon (system.next state)

/-- The viability region is invariant under the declared policy. -/
def ClosedUnderPolicy {State Action : Type}
    (phenotype : Phenotype State) (system : PolicySystem State Action) : Prop :=
  forall state, phenotype.viable state -> phenotype.viable (system.next state)

/-- Policy closure is sufficient for phenotype-relative viability at every finite
horizon. This is a conditional functional result, not a theorem of moral goodness. -/
theorem viableFor_of_closedUnderPolicy {State Action : Type}
    (phenotype : Phenotype State) (system : PolicySystem State Action)
    (closed : ClosedUnderPolicy phenotype system) :
    forall horizon state, phenotype.viable state ->
      ViableFor phenotype system horizon state := by
  intro horizon
  induction horizon with
  | zero =>
      intro state viableNow
      exact viableNow
  | succ horizon inductionHypothesis =>
      intro state viableNow
      exact And.intro viableNow
        (inductionHypothesis (system.next state) (closed state viableNow))

/-- The state reached after a finite number of closed-loop policy steps. -/
def IterateNext {State Action : Type}
    (system : PolicySystem State Action) : Nat -> State -> State
  | 0, state => state
  | Nat.succ steps, state => IterateNext system steps (system.next state)

/-- Recovery at an exact time means that the state reached after `steps` transitions
is again inside the phenotype's viability region. -/
def RecoversAt {State Action : Type}
    (phenotype : Phenotype State) (system : PolicySystem State Action)
    (steps : Nat) (state : State) : Prop :=
  phenotype.viable (IterateNext system steps state)

/-- A perturbation is represented explicitly rather than hidden inside the dynamics. -/
abbrev Perturbation (State : Type) := State -> State

/-- A policy is robustly recoverable from a declared perturbation within an exact
bound when every viable starting state returns to viability at that bound. -/
def RecoversFrom {State Action : Type}
    (phenotype : Phenotype State) (system : PolicySystem State Action)
    (perturb : Perturbation State) (steps : Nat) : Prop :=
  forall state, phenotype.viable state ->
    RecoversAt phenotype system steps (perturb state)

/-- Once an exact recovery point has been reached, policy closure guarantees
continued finite-horizon viability from that recovered point. -/
theorem viableAfterRecovery {State Action : Type}
    (phenotype : Phenotype State) (system : PolicySystem State Action)
    (closed : ClosedUnderPolicy phenotype system)
    {recoverySteps futureHorizon : Nat} {perturbedState : State}
    (recovered : RecoversAt phenotype system recoverySteps perturbedState) :
    ViableFor phenotype system futureHorizon
      (IterateNext system recoverySteps perturbedState) := by
  exact viableFor_of_closedUnderPolicy phenotype system closed futureHorizon
    (IterateNext system recoverySteps perturbedState) recovered

/-- A closed viability region can nevertheless fail completely after a perturbation:
in this system state zero is viable and stable, while perturbation enters the
non-viable absorbing state one. -/
theorem closure_does_not_imply_recovery :
    exists (phenotype : Phenotype Nat) (system : PolicySystem Nat Unit)
      (perturb : Perturbation Nat),
      ClosedUnderPolicy phenotype system /\
      (forall steps, Not (RecoversFrom phenotype system perturb steps)) := by
  let phenotype : Phenotype Nat := { viable := fun state => state = 0 }
  let system : PolicySystem Nat Unit := {
    step := fun state _ => state
    policy := fun _ => ()
  }
  let perturb : Perturbation Nat := fun _ => 1
  have stuck : forall steps, IterateNext system steps 1 = 1 := by
    intro steps
    induction steps with
    | zero => rfl
    | succ steps inductionHypothesis => exact inductionHypothesis
  refine Exists.intro phenotype (Exists.intro system (Exists.intro perturb ?_))
  apply And.intro
  · intro state viableState
    exact viableState
  · intro steps recovers
    have recoveredOne := recovers 0 rfl
    change phenotype.viable (IterateNext system steps 1) at recoveredOne
    rw [stuck steps] at recoveredOne
    exact Nat.one_ne_zero recoveredOne

end TopologicalAlignment
