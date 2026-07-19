import TopologicalAlignment.Definitions

namespace TopologicalAlignment

theorem capabilityDominates_refl (a : RelationalEmpowerment) :
    CapabilityDominates a a := by
  exact And.intro (Nat.le_refl a.self)
    (And.intro (Nat.le_refl a.other) (Nat.le_refl a.joint))

theorem capabilityDominates_trans {a b c : RelationalEmpowerment}
    (hab : CapabilityDominates a b) (hbc : CapabilityDominates b c) :
    CapabilityDominates a c := by
  exact And.intro (Nat.le_trans hbc.1 hab.1)
    (And.intro (Nat.le_trans hbc.2.1 hab.2.1)
      (Nat.le_trans hbc.2.2 hab.2.2))

theorem capabilityDominates_antisymm_on_capabilities
    {a b : RelationalEmpowerment}
    (hab : CapabilityDominates a b) (hba : CapabilityDominates b a) :
    a.self = b.self /\ a.other = b.other /\ a.joint = b.joint := by
  exact And.intro (Nat.le_antisymm hba.1 hab.1)
    (And.intro (Nat.le_antisymm hba.2.1 hab.2.1)
      (Nat.le_antisymm hba.2.2 hab.2.2))

end TopologicalAlignment
