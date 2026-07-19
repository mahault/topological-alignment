import TopologicalAlignment.RobustDominance

namespace TopologicalAlignment

/-- Outcomes for a focal phenotype are paired with the remaining capability of an
affected other. This makes externalized costs part of the modeled result. -/
structure RelationalOutcome where
  focal : FunctionalOutcome
  affectedCapability : Nat
  deriving DecidableEq, Repr

/-- Robust dominance viewed only from the focal phenotype. This intentionally omits
the moral constraint so that the omission can be exposed by counterexample. -/
def FocallyRobustlyDominates {Environment Regime : Type}
    (outcome : Regime -> Environment -> RelationalOutcome)
    (better worse : Regime) : Prop :=
  forall environment,
    OutcomeDominates (outcome better environment).focal
      (outcome worse environment).focal

/-- A hard capability floor for every affected other in every declared environment.
The floor is not traded against gains to the focal phenotype. -/
def RespectsAffectedFloor {Environment Regime : Type}
    (outcome : Regime -> Environment -> RelationalOutcome)
    (floor : Nat) (regime : Regime) : Prop :=
  forall environment, floor <= (outcome regime environment).affectedCapability

/-- Conditional admissible dominance requires both robust focal improvement and a
declared non-externalization floor. This is a definition of a constraint, not a proof
that the floor or its normative authority is correct. -/
def AdmissiblyDominates {Environment Regime : Type}
    (outcome : Regime -> Environment -> RelationalOutcome)
    (floor : Nat) (better worse : Regime) : Prop :=
  FocallyRobustlyDominates outcome better worse /\
  RespectsAffectedFloor outcome floor better

theorem admissible_implies_focal_dominance {Environment Regime : Type}
    {outcome : Regime -> Environment -> RelationalOutcome}
    {floor : Nat} {better worse : Regime}
    (admissible : AdmissiblyDominates outcome floor better worse) :
    FocallyRobustlyDominates outcome better worse :=
  admissible.1

theorem admissible_implies_affected_floor {Environment Regime : Type}
    {outcome : Regime -> Environment -> RelationalOutcome}
    {floor : Nat} {better worse : Regime}
    (admissible : AdmissiblyDominates outcome floor better worse) :
    RespectsAffectedFloor outcome floor better :=
  admissible.2

/-- Even robust focal superiority in every environment can coexist with violating an
affected other's capability floor. Functional goodness for one phenotype therefore
does not entail relational or moral admissibility. -/
theorem focal_robust_dominance_does_not_imply_nonexternalization :
    exists (outcome : Bool -> Unit -> RelationalOutcome) (floor : Nat),
      FocallyRobustlyDominates outcome true false /\
      Not (RespectsAffectedFloor outcome floor true) := by
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
  let outcome : Bool -> Unit -> RelationalOutcome := fun regime _ =>
    if regime then
      { focal := high, affectedCapability := 0 }
    else
      { focal := low, affectedCapability := 1 }
  refine Exists.intro outcome (Exists.intro 1 ?_)
  apply And.intro
  · intro environment
    exact And.intro (Nat.zero_le 1)
      (And.intro (Nat.zero_le 1)
        (And.intro (Nat.zero_le 1) (Nat.zero_le 1)))
  · intro respectsFloor
    have violation := respectsFloor ()
    exact Nat.not_succ_le_zero 0 violation

end TopologicalAlignment
