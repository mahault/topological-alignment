import Init

namespace TopologicalAlignment

/-- A deliberately minimal finite profile. It keeps distinct the control available
to an agent, to affected others, and to the group. The distribution coordinate is
not included in the componentwise order until its normative interpretation is fixed. -/
structure RelationalEmpowerment where
  self : Nat
  other : Nat
  joint : Nat
  distributionShift : Int
  deriving DecidableEq, Repr

/-- Componentwise capability dominance. This is a partial order over the three
capability coordinates, not a scalar moral score. -/
def CapabilityDominates (a b : RelationalEmpowerment) : Prop :=
  b.self <= a.self /\ b.other <= a.other /\ b.joint <= a.joint

end TopologicalAlignment
