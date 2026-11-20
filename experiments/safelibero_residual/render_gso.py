"""Contact sheets of every GSO obstacle candidate (index ok == True), for auditing object names by eye.

    ./pleiades_run.sh python render_gso.py --out runs_v2/gso_sheets
Writes sheet_XX.png (8 x 8 tiles, 160 px, numbered) and sheet_index.json {number: model name}.
"""
import argparse
import json
import os

import numpy as np


def scene_xml(d, size):
    h = max(size)
    return f"""<mujoco><asset>
  <texture file="{d}/texture.png" name="tex" type="2d"/>
  <material name="mat" texture="tex"/>
  <mesh file="{d}/model.obj" name="vis"/></asset>
  <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/><global offwidth="640" offheight="640"/></visual>
  <worldbody>
    <geom type="plane" size="1 1 0.01" rgba="0.85 0.85 0.85 1"/>
    <geom type="mesh" mesh="vis" material="mat" contype="0" conaffinity="0"/>
    <camera name="c" pos="{1.6 * h:.3f} {-1.6 * h:.3f} {1.4 * h:.3f}" xyaxes="0.707 0.707 0 -0.4 0.4 0.82"/>
  </worldbody></mujoco>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_v2/gso_sheets")
    ap.add_argument("--tile", type=int, default=160)
    ap.add_argument("--grid", type=int, default=8)
    a = ap.parse_args()
    import cv2
    import mujoco
    import gso_objects as G
    G.ensure_local()
    idx = G.load_index()
    names = sorted(n for n, d in idx.items() if d["ok"])
    os.makedirs(a.out, exist_ok=True)
    T, g = a.tile, a.grid
    tiles = []
    for n in names:
        d = os.path.join(G.local_root(), "models", n)
        try:
            m = mujoco.MjModel.from_xml_string(scene_xml(d, idx[n]["size"]))
            dat = mujoco.MjData(m)
            mujoco.mj_forward(m, dat)
            r = mujoco.Renderer(m, 320, 320)
            r.update_scene(dat, camera="c")
            img = cv2.resize(r.render(), (T, T), interpolation=cv2.INTER_AREA)
            r.close()
        except Exception as ex:  # noqa: BLE001
            img = np.full((T, T, 3), 255, np.uint8)
            print("fail", n, repr(ex)[:120], flush=True)
        tiles.append(img)
    index = {}
    for s in range(0, len(names), g * g):
        sheet = np.full((g * T, g * T, 3), 255, np.uint8)
        for j, img in enumerate(tiles[s:s + g * g]):
            k = s + j
            index[k] = names[k]
            y, x = (j // g) * T, (j % g) * T
            img = img.copy()
            cv2.rectangle(img, (0, 0), (44, 22), (255, 255, 255), -1)
            cv2.putText(img, str(k), (2, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 0, 0), 2)
            sheet[y:y + T, x:x + T] = img
        cv2.imwrite(os.path.join(a.out, f"sheet_{s // (g * g):02d}.png"), sheet[:, :, ::-1])
    json.dump(index, open(os.path.join(a.out, "sheet_index.json"), "w"), indent=0)
    print("wrote", len(names), "objects", flush=True)


if __name__ == "__main__":
    main()
