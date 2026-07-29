"""Render sphere vs superquadric keep-out surfaces from dump_sq_geometry.py.

Draws the EFFECTIVE barrier surface each arm enforces — the set where
b = dist - r_shape(u) - eef_radius - d_safe = 0 — so the two arms are
compared on the surface the robot actually feels, not the raw object fit.
"""

import argparse
import json
import pathlib

import numpy as np
import plotly.graph_objects as go

GRID = 60


def _dirs(n=GRID):
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n)
    T, P = np.meshgrid(th, ph, indexing="ij")
    return np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)


def sq_radius(u, scales, eps):
    """Radial distance from center to the superquadric surface along u."""
    s = np.asarray(scales, dtype=float)
    return np.power(np.sum(np.abs(u / s) ** (2.0 / eps), axis=-1), -eps / 2.0)


def surface(center, radial, offset, color, name, opacity=0.28, visible=True):
    u = _dirs()
    r = radial(u) + offset
    p = np.asarray(center) + u * r[..., None]
    return go.Surface(
        x=p[..., 0], y=p[..., 1], z=p[..., 2],
        surfacecolor=np.zeros(p.shape[:2]),
        colorscale=[[0, color], [1, color]], showscale=False,
        opacity=opacity, name=name, showlegend=True, visible=visible,
        hovertemplate=name + "<extra></extra>",
    )


def box_wire(lo, hi, color, name):
    lo, hi = np.asarray(lo), np.asarray(hi)
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [c[a, 0], c[b, 0], None]
        ys += [c[a, 1], c[b, 1], None]
        zs += [c[a, 2], c[b, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                        line=dict(color=color, width=5), hoverinfo="skip")


def marker(pos, color, name, symbol="circle", size=7):
    p = np.asarray(pos)
    return go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text",
                        marker=dict(size=size, color=color, symbol=symbol),
                        text=[name], textposition="top center", name=name)


def scene_figure(s, eps):
    off = s["eef_radius"] + s["d_safe"]
    c = np.asarray(s["center"])
    traces = [
        box_wire(s["aabb_lo"], s["aabb_hi"], "#111827", "object true AABB"),
        surface(c, lambda u: np.full(u.shape[:-1], s["sphere_r_obs"]), off,
                "#2563eb", f"SPHERE keep-out (r={s['sphere_r_obs']:.3f}) — production"),
        surface(c, lambda u: sq_radius(u, s["sym_extents"], eps), off,
                "#dc2626", f"SUPERQUADRIC sym fit (eps={eps}) — the n=200 arm"),
        surface(np.asarray(s["mid_center"]), lambda u: sq_radius(u, s["mid_extents"], eps), off,
                "#16a34a", "SUPERQUADRIC mid fit (tight, untested at scale)", visible="legendonly"),
        marker(s["eef_pos"], "#f59e0b", "EEF start", "diamond", 8),
    ]
    if s.get("target_pos"):
        traces.append(marker(s["target_pos"], "#7c3aed", "grasp target", "square", 8))
    if s.get("dest_pos"):
        traces.append(marker(s["dest_pos"], "#0891b2", "destination", "cross", 8))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Task {s['task_id']} — {s['description']}<br>"
              f"<sub>obstacle {s['obstacle']} · sphere r {s['sphere_r_obs']:.3f} m vs "
              f"sym half-extents {np.round(s['sym_extents'], 3).tolist()} m · "
              f"both surfaces shown inflated by eef_radius+d_safe = {off:.3f} m</sub>",
        scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"),
        height=680, margin=dict(l=0, r=0, t=90, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    return fig


def volumes(s, eps):
    """Monte-Carlo volume of each EFFECTIVE keep-out region (offset included)."""
    off = s["eef_radius"] + s["d_safe"]
    u = _dirs(90)
    solid = np.sin(np.linspace(0, np.pi, 90))[:, None]  # sin(theta) weight
    w = solid / solid.sum() / u.shape[1]

    def vol(radial):
        r = radial(u) + off
        return float((4 * np.pi / 3) * np.sum(w * r ** 3))

    return (vol(lambda uu: np.full(uu.shape[:-1], s["sphere_r_obs"])),
            vol(lambda uu: sq_radius(uu, s["sym_extents"], eps)),
            vol(lambda uu: sq_radius(uu, s["mid_extents"], eps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--eps", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.dump).read_text())
    scenes = data["scenes"]

    rows = []
    for s in scenes:
        vs, vq, vm = volumes(s, args.eps)
        rows.append(
            f"<tr><td>{s['task_id']}</td><td>{s['obstacle']}</td>"
            f"<td>{s['sphere_r_obs']:.3f}</td>"
            f"<td>{np.round(s['sym_extents'], 3).tolist()}</td>"
            f"<td>{np.round(s['mid_extents'], 3).tolist()}</td>"
            f"<td>{vs * 1e3:.2f}</td><td>{vq * 1e3:.2f}</td>"
            f"<td class='{'bad' if vq > vs else 'good'}'>{vq / vs:.2f}×</td>"
            f"<td>{vm / vs:.2f}×</td></tr>")

    head = """<meta charset="utf-8"><title>SafeLIBERO keep-out geometry: sphere vs superquadric</title>
<style>
body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;
     max-width:1180px;color:#111827;background:#fff}
h1{font-size:22px} h2{font-size:17px;margin-top:34px}
table{border-collapse:collapse;margin:14px 0;font-size:14px;width:100%}
th,td{border:1px solid #d1d5db;padding:6px 9px;text-align:left}
th{background:#f3f4f6} td.bad{background:#fee2e2;font-weight:600}
td.good{background:#dcfce7;font-weight:600}
.note{background:#f9fafb;border-left:4px solid #2563eb;padding:12px 16px;margin:16px 0;font-size:15px}
code{background:#f3f4f6;padding:1px 4px;border-radius:3px}
</style>"""

    body = [head, "<h1>SafeLIBERO Spatial keep-out geometry — why the superquadric arm lost TSR</h1>",
            "<div class='note'>Each surface is the <b>effective barrier surface</b> the robot feels: "
            f"<code>b = dist − r_shape(u) − eef_radius − d_safe = 0</code>, i.e. the object fit inflated by "
            f"{scenes[0]['eef_radius'] + scenes[0]['d_safe']:.3f} m. "
            "Blue = production sphere. Red = the sym-fit superquadric actually evaluated at n=200. "
            "Green (hidden by default, click the legend) = the tight mid fit. "
            "Rotate/zoom; toggle surfaces in the legend.</div>",
            "<h2>Keep-out volume: superquadric vs sphere</h2>",
            "<table><tr><th>task</th><th>obstacle</th><th>sphere r [m]</th>"
            "<th>sym half-extents [m]</th><th>mid half-extents [m]</th>"
            "<th>sphere vol [L]</th><th>sym vol [L]</th><th>sym / sphere</th><th>mid / sphere</th></tr>"
            + "".join(rows) + "</table>"]

    first = True
    for s in scenes:
        body.append(f"<h2>Task {s['task_id']} — {s['description']}</h2>")
        body.append(scene_figure(s, args.eps).to_html(
            full_html=False, include_plotlyjs=("inline" if first else False)))
        first = False

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body))
    print(f"[render] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
