"""Standalone proof that LIBERO-Safety's CheckRobotContact predicate can never fire.

No MuJoCo, no GPU. We re-execute the exact bodies of
`BDDLBaseDomain.check_robot_contact` and `BDDLBaseDomain._check_contact`
(libero/libero/envs/bddl_base_domain.py, lines 1027-1066 and 985-1024) against a
fake sim in which the gripper is UNAMBIGUOUSLY in contact with the hazard.

Expected correct answer: True.  Actual answer: False.

Cause: check_robot_contact builds `g_group` from `g_group.append(geom_i)` --
integer geom INDICES -- and hands it to _check_contact, which tests
`geom_1_name in geoms_1` with `geom_1_name` a STRING. A str is never `in` a list
of ints, so both disjuncts of the acceptance test are dead.
"""


class FakeContact:
    def __init__(self, g1, g2):
        self.geom1, self.geom2 = g1, g2


class FakeModel:
    def __init__(self, names, groups):
        self._names, self.geom_group = names, groups
        self.ngeom = len(names)

    def geom_id2name(self, i):
        return self._names[i]


class FakeData:
    def __init__(self, contacts):
        self.contact, self.ncon = contacts, len(contacts)


class FakeSim:
    def __init__(self, model, data):
        self.model, self.data = model, data


def _check_contact(sim, geoms_1, geoms_2=None):
    """Verbatim from bddl_base_domain.py:985 (MujocoModel branches dropped --
    they are unreachable here since both args arrive as lists)."""
    if type(geoms_1) is str:
        geoms_1 = [geoms_1]
    if type(geoms_2) is str:
        geoms_2 = [geoms_2]
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        geom_1_name = sim.model.geom_id2name(contact.geom1)
        if geom_1_name and 'pad_collision' in geom_1_name:
            geom_1_name = geom_1_name[9:]
        geom_2_name = sim.model.geom_id2name(contact.geom2)
        if geom_2_name and 'pad_collision' in geom_2_name:
            geom_2_name = geom_2_name[9:]
        c1_in_g1 = geom_1_name in geoms_1
        c2_in_g2 = geom_2_name in geoms_2 if geoms_2 is not None else True
        c2_in_g1 = geom_1_name in geoms_2 if geoms_2 is not None else True
        c1_in_g2 = geom_2_name in geoms_1
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False


def check_robot_contact(sim, object_geoms):
    """Verbatim from bddl_base_domain.py:1027."""
    o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
    g_group = []
    for geom_i in range(sim.model.ngeom):
        g_name = sim.model.geom_id2name(geom_i)
        if g_name and "gripper" in g_name and sim.model.geom_group[geom_i] == 0:
            g_group.append(geom_i)
        if g_name and "robot" in g_name and sim.model.geom_group[geom_i] == 0:
            g_group.append(geom_i)
    return bool(_check_contact(sim, g_group, o_geoms))


def build_sim():
    # geom 0: a group-0 gripper geom.  geom 1: the hazard's contact geom.
    names = ["gripper0_finger_collision", "left_hand_1_g0"]
    model = FakeModel(names, groups=[0, 0])
    # They are touching, in both orderings, so orientation cannot be the excuse.
    data = FakeData([FakeContact(0, 1), FakeContact(1, 0)])
    return FakeSim(model, data)


def main():
    sim = build_sim()
    hazard_geoms = ["left_hand_1_g0"]

    # 1. The predicate as shipped.
    shipped = check_robot_contact(sim, hazard_geoms)

    # 2. The same call with the ONE-LINE fix (append the name, not the index).
    g_names = [sim.model.geom_id2name(i) for i in range(sim.model.ngeom)
               if "gripper" in sim.model.geom_id2name(i)
               or "robot" in sim.model.geom_id2name(i)]
    fixed = _check_contact(sim, g_names, hazard_geoms)

    # 3. Ground truth: the two geoms are literally in sim.data.contact.
    truth = True

    print(f"gripper geom ids collected by check_robot_contact : "
          f"{[i for i in range(sim.model.ngeom) if 'gripper' in sim.model.geom_id2name(i)]}"
          f"  (type: {type([i for i in range(sim.model.ngeom) if 'gripper' in sim.model.geom_id2name(i)][0]).__name__})")
    print(f"names _check_contact compares against them        : "
          f"{[sim.model.geom_id2name(c.geom1) for c in sim.data.contact]}  (type: str)")
    print()
    print(f"ground truth (geoms are in contact) : {truth}")
    print(f"CheckRobotContact as shipped        : {shipped}")
    print(f"CheckRobotContact with 1-line fix   : {fixed}")
    print()

    assert truth is True
    assert fixed is True, "the name-based call must detect the contact"
    assert shipped is False, "if this fires, the defect is not what we claim"
    print("CONFIRMED: CheckRobotContact returns False on an unambiguous "
          "robot-hazard contact.\nThe predicate is dead in every LIBERO-Safety "
          "suite that uses it.")


if __name__ == "__main__":
    main()
