import TopologicalAlignment.RelationalViability

namespace TopologicalAlignment

/-- Procedural conditions required by the project's adopted definition of moral
goodness. Their truth must be established by an application-specific interpretation. -/
structure NormativeProcedure (Regime : Type) where
  justifiable : Regime -> Prop
  contestable : Regime -> Prop
  epistemicallyResponsive : Regime -> Prop
  repairable : Regime -> Prop

/-- At least one functional coordinate or the affected party's capability strictly
improves in at least one declared environment. -/
def HasRelationalImprovement {Environment Regime : Type}
    (outcome : Regime -> Environment -> RelationalOutcome)
    (better baseline : Regime) : Prop :=
  exists environment,
    (outcome baseline environment).focal.viabilityMargin <
        (outcome better environment).focal.viabilityMargin \/
    (outcome baseline environment).focal.availableOptions <
        (outcome better environment).focal.availableOptions \/
    (outcome baseline environment).focal.epistemicFit <
        (outcome better environment).focal.epistemicFit \/
    (outcome better environment).focal.recoveryCost <
        (outcome baseline environment).focal.recoveryCost \/
    (outcome baseline environment).affectedCapability <
        (outcome better environment).affectedCapability

/-- The affected party is not made worse off on its represented capability dimension
in any declared environment. -/
def DoesNotReduceAffectedCapability {Environment Regime : Type}
    (outcome : Regime -> Environment -> RelationalOutcome)
    (better baseline : Regime) : Prop :=
  forall environment,
    (outcome baseline environment).affectedCapability <=
      (outcome better environment).affectedCapability

/-- The adopted finite interpretation of positive moral goodness relative to a
baseline. It combines admissibility, non-externalized plural improvement, and the
procedural conditions in the project's normative definition. -/
def MorallyGoodRelativeTo {Environment Regime : Type}
    (outcome : Regime -> Environment -> RelationalOutcome)
    (procedure : NormativeProcedure Regime) (floor : Nat)
    (better baseline : Regime) : Prop :=
  AdmissiblyDominates outcome floor better baseline /\
  DoesNotReduceAffectedCapability outcome better baseline /\
  HasRelationalImprovement outcome better baseline /\
  procedure.justifiable better /\
  procedure.contestable better /\
  procedure.epistemicallyResponsive better /\
  procedure.repairable better

theorem morallyGood_implies_admissible {Environment Regime : Type}
    {outcome : Regime -> Environment -> RelationalOutcome}
    {procedure : NormativeProcedure Regime} {floor : Nat}
    {better baseline : Regime}
    (good : MorallyGoodRelativeTo outcome procedure floor better baseline) :
    AdmissiblyDominates outcome floor better baseline :=
  good.1

theorem morallyGood_implies_plural_improvement {Environment Regime : Type}
    {outcome : Regime -> Environment -> RelationalOutcome}
    {procedure : NormativeProcedure Regime} {floor : Nat}
    {better baseline : Regime}
    (good : MorallyGoodRelativeTo outcome procedure floor better baseline) :
    DoesNotReduceAffectedCapability outcome better baseline /\
    HasRelationalImprovement outcome better baseline :=
  And.intro good.2.1 good.2.2.1

theorem morallyGood_implies_procedural_conditions {Environment Regime : Type}
    {outcome : Regime -> Environment -> RelationalOutcome}
    {procedure : NormativeProcedure Regime} {floor : Nat}
    {better baseline : Regime}
    (good : MorallyGoodRelativeTo outcome procedure floor better baseline) :
    procedure.justifiable better /\
    procedure.contestable better /\
    procedure.epistemicallyResponsive better /\
    procedure.repairable better :=
  good.2.2.2

end TopologicalAlignment
