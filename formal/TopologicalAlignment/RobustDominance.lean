import TopologicalAlignment.Viability

namespace TopologicalAlignment

/-- A deliberately non-scalar phenotype-relative outcome profile. Larger values are
better for the first three coordinates; lower recovery cost is better. Coordinates
must be operationalized independently in an empirical application. -/
structure FunctionalOutcome where
  viabilityMargin : Nat
  availableOptions : Nat
  epistemicFit : Nat
  recoveryCost : Nat
  deriving DecidableEq, Repr

/-- Componentwise weak dominance. No exchange rate between viability, options,
epistemic fit, and recovery cost is assumed. -/
def OutcomeDominates (better worse : FunctionalOutcome) : Prop :=
  worse.viabilityMargin <= better.viabilityMargin /\
  worse.availableOptions <= better.availableOptions /\
  worse.epistemicFit <= better.epistemicFit /\
  better.recoveryCost <= worse.recoveryCost

/-- A regime is robustly at least as good as another when it weakly dominates in
every environment in the declared comparison class. -/
def RobustlyDominates {Environment Regime : Type}
    (outcome : Regime -> Environment -> FunctionalOutcome)
    (better worse : Regime) : Prop :=
  forall environment, OutcomeDominates (outcome better environment)
    (outcome worse environment)

theorem outcomeDominates_refl (outcome : FunctionalOutcome) :
    OutcomeDominates outcome outcome := by
  exact And.intro (Nat.le_refl outcome.viabilityMargin)
    (And.intro (Nat.le_refl outcome.availableOptions)
      (And.intro (Nat.le_refl outcome.epistemicFit)
        (Nat.le_refl outcome.recoveryCost)))

theorem outcomeDominates_trans {a b c : FunctionalOutcome}
    (hab : OutcomeDominates a b) (hbc : OutcomeDominates b c) :
    OutcomeDominates a c := by
  exact And.intro (Nat.le_trans hbc.1 hab.1)
    (And.intro (Nat.le_trans hbc.2.1 hab.2.1)
      (And.intro (Nat.le_trans hbc.2.2.1 hab.2.2.1)
        (Nat.le_trans hab.2.2.2 hbc.2.2.2)))

theorem robustlyDominates_refl {Environment Regime : Type}
    (outcome : Regime -> Environment -> FunctionalOutcome) (regime : Regime) :
    RobustlyDominates outcome regime regime := by
  intro environment
  exact outcomeDominates_refl (outcome regime environment)

theorem robustlyDominates_trans {Environment Regime : Type}
    (outcome : Regime -> Environment -> FunctionalOutcome) {a b c : Regime}
    (hab : RobustlyDominates outcome a b)
    (hbc : RobustlyDominates outcome b c) :
    RobustlyDominates outcome a c := by
  intro environment
  exact outcomeDominates_trans (hab environment) (hbc environment)

/-- Performance at one selected environment cannot establish robust dominance. The
witness reverses the viability ordering between two environments. -/
theorem one_environment_dominance_not_robust :
    exists (outcome : Bool -> Bool -> FunctionalOutcome),
      OutcomeDominates (outcome true false) (outcome false false) /\
      Not (RobustlyDominates outcome true false) := by
  let high : FunctionalOutcome := {
    viabilityMargin := 1
    availableOptions := 1
    epistemicFit := 1
    recoveryCost := 0
  }
  let low : FunctionalOutcome := {
    viabilityMargin := 0
    availableOptions := 0
    epistemicFit := 0
    recoveryCost := 1
  }
  let outcome : Bool -> Bool -> FunctionalOutcome := fun regime environment =>
    if regime = environment then low else high
  refine Exists.intro outcome ?_
  apply And.intro
  · exact And.intro (Nat.zero_le 1)
      (And.intro (Nat.zero_le 1)
        (And.intro (Nat.zero_le 1) (Nat.zero_le 1)))
  · intro robust
    have reversed := robust true
    exact Nat.not_succ_le_zero 0 reversed.1

end TopologicalAlignment
