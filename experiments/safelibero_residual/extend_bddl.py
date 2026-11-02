"""Write SafeLIBERO-Spatial BDDL variants that also contain the 13 SafeLIBERO obstacle categories the
Spatial suite never uses (19 obstacle types in total), for the object-scaling study.

    python extend_bddl.py            # -> bddl_x/<task>.bddl for the 4 Spatial tasks

The extra objects start parked off the table on a grid, each in its own region, so the
constructor's random placement cannot fail on a crowded table. Scenes are built from the
official layouts by sl_env.apply_layout (joint states and fixture poses copied by name),
then scenes.py teleports the chosen obstacles onto the table.
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "SafeLIBERO", "safelibero", "libero", "libero", "bddl_files", "safelibero_spatial")
OUT = os.path.join(HERE, "bddl_x")
# SafeLIBERO obstacle classes (libero/envs/objects/obstacle_objects.py) not used by the Spatial suite
EXTRA = ["orange_juice", "alphabet_soup", "bbq_sauce", "butter", "chocolate_pudding", "cream_cheese", "ketchup",
         "new_salad_dressing", "popcorn", "basket", "basin_faucet_base", "chefmate_8_frypan", "simple_rack"]


def extend(text):
    regions, objects, init = [], [], []
    for k, o in enumerate(EXTRA):
        x0, y0 = 1.2 + 0.45 * (k % 5), -0.9 + 0.45 * (k // 5)  # off the table, 45 cm apart
        regions.append(f"      (x_{o}_region\n          (:target main_table)\n"
                       f"          (:ranges (({x0:.2f} {y0:.2f} {x0 + 0.02:.2f} {y0 + 0.02:.2f})))\n      )")
        objects.append(f"    {o}_obstacle_1 - {o}_obstacle")
        init.append(f"    (On {o}_obstacle_1 main_table_x_{o}_region)")
    text, n1 = re.subn(r"\(:regions\n", "(:regions\n" + "\n".join(regions) + "\n", text, count=1)
    text, n2 = re.subn(r"(box_base_1 - box_base\n)", r"\1" + "\n".join(objects) + "\n", text, count=1)
    text, n3 = re.subn(r"(\(On box_base_1 main_table_box_base_region\)\n)", r"\1" + "\n".join(init) + "\n", text, count=1)
    assert n1 == n2 == n3 == 1, (n1, n2, n3)
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.bddl")))
    assert len(files) == 4, files
    for f in files:
        open(os.path.join(OUT, os.path.basename(f)), "w").write(extend(open(f).read()))
    print("wrote", len(files), "files to", OUT)


if __name__ == "__main__":
    main()
