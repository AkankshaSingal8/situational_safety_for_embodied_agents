"""Step-by-step verification of the superquadric barrier as implemented.

Three independent checks, all offline (numpy mirror of the jnp server code in
openpi.models.pi0_guided._eff_dist):

  C1  Level set. Does b = 0 recover the fitted superquadric surface?
  C2  Distance semantics. The barrier inflates by a RADIAL offset along
      u = (p - o)/||p - o||. The physically meaningful set is the Minkowski
      dilation (every point whose distance to the object is < eef_radius +
      d_safe). Radial offset is a SUBSET of that for convex shapes, so the
      enforced keep-out is smaller than the collision condition it claims to
      encode. This measures the gap, in mm, per direction.
  C3  Volume floor. Given the objects' true extents, what is the smallest
      keep-out volume ANY correct cover can have? If that floor already
      exceeds the production sphere, no improvement in fitting can recover
      the lost success rate — the sphere wins by under-covering.
"""

import numpy as np

EPS_DEFAULT = 0.4
OFF = 0.10  # eef_radius 0.09 + d_safe 0.01

# From diag_obstacle_extent.py on the live scenes.
OBJECTS = {
    "moka_pot": {
        "sphere_r": 0.080,
        "sym": [0.0654, 0.1460, 0.1393],       # what the n=200 arm sent
        "mid": [0.0651, 0.1216, 0.1234],       # tight AABB fit, whole object
        "body_only": [0.0634, 0.0903, 0.1107],  # mesh g0: the pot itself
        "handle": [0.0126, 0.0745, 0.0780],     # mesh g1: the handle
    },
    "wine_bottle": {
        "sphere_r": 0.060,
        "sym": [0.0355, 0.0351, 0.2540],
        "mid": [0.0354, 0.0348, 0.1270],
        "body_only": [0.0345, 0.0345, 0.1199],
        "handle": None,
    },
}


def sq_radius(u, s, eps=EPS_DEFAULT):
    """Server's r_shape(u): radial distance from center to the SQ surface."""
    s = np.asarray(s, dtype=float)
    q = np.sum(np.abs(u / s) ** (2.0 / eps), axis=-1)
    return (q + 1e-8) ** (-eps / 2.0)


def sq_inside_outside(p, s, eps=EPS_DEFAULT):
    """Paper eq. (1): g(x) = sum |x_i/a_i|^(2/eps2) ... ; g >= 1 is outside."""
    s = np.asarray(s, dtype=float)
    return np.sum(np.abs(p / s) ** (2.0 / eps), axis=-1) ** (eps / 2.0)


def sphere_dirs(n_th=180, n_ph=360):
    th = (np.arange(n_th) + 0.5) / n_th * np.pi
    ph = np.arange(n_ph) / n_ph * 2 * np.pi
    T, P = np.meshgrid(th, ph, indexing="ij")
    u = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    return u, np.sin(T)


def volume_radial(radial_fn, offset=OFF):
    """Volume of {p : ||p|| <= radial_fn(u) + offset} by solid-angle integral."""
    u, w = sphere_dirs()
    r = radial_fn(u) + offset
    return (4 * np.pi / 3) * np.sum(w * r ** 3) / np.sum(w)


def surface_points(s, eps=EPS_DEFAULT, n=200):
    """Dense sample of the superquadric surface."""
    u, _ = sphere_dirs(n, 2 * n)
    r = sq_radius(u, s, eps)
    return (u * r[..., None]).reshape(-1, 3)


def true_distance_to_surface(p, surf):
    """Brute-force Euclidean distance from p to the sampled SQ surface."""
    return np.min(np.linalg.norm(surf - p, axis=-1))


def check_level_set(s, eps=EPS_DEFAULT):
    """C1: b = dist - r_shape(u) must vanish exactly on the surface."""
    surf = surface_points(s, eps, n=60)
    idx = np.linspace(0, len(surf) - 1, 400).astype(int)
    err = []
    for p in surf[idx]:
        d = np.linalg.norm(p)
        u = p / d
        err.append(d - sq_radius(u, s, eps))
    err = np.abs(np.asarray(err))
    g = sq_inside_outside(surf[idx], s, eps)
    return err.max(), np.abs(g - 1.0).max()


def check_offset_gap(s, eps=EPS_DEFAULT, offset=OFF, n_probe=1500):
    """C2: for points ON the enforced boundary (radial offset), what is the
    TRUE distance to the object surface? It should be `offset` everywhere;
    any shortfall is keep-out the barrier claims but does not enforce."""
    surf = surface_points(s, eps, n=220)
    rng = np.random.default_rng(0)
    u = rng.normal(size=(n_probe, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    r = sq_radius(u, s, eps)
    boundary = u * (r + offset)[:, None]
    true_d = np.array([true_distance_to_surface(p, surf) for p in boundary])
    return true_d


def main():
    np.set_printoptions(suppress=True)

    print("=" * 78)
    print("C1  LEVEL SET — is b=0 the fitted superquadric surface?")
    print("=" * 78)
    for name, o in OBJECTS.items():
        e, g = check_level_set(o["sym"])
        print(f"  {name:12s} max |dist - r_shape(u)| on surface = {e:.2e} m   "
              f"max |g(x) - 1| = {g:.2e}")
    print("  -> the radial form and the paper's implicit form share the zero level set.\n")

    print("=" * 78)
    print(f"C2  OFFSET SEMANTICS — true clearance on the enforced boundary")
    print(f"    (should be exactly {OFF:.3f} m everywhere; less = under-enforced)")
    print("=" * 78)
    for name, o in OBJECTS.items():
        for eps, tag in ((0.4, "boxy eps=0.4 (as run)"), (1.0, "ellipsoid eps=1.0")):
            d = check_offset_gap(o["sym"], eps)
            print(f"  {name:12s} {tag:24s} min {d.min()*1000:6.1f} mm  "
                  f"p05 {np.percentile(d,5)*1000:6.1f}  median {np.median(d)*1000:6.1f}  "
                  f"max {d.max()*1000:6.1f}   shortfall up to {(OFF-d.min())*1000:5.1f} mm")
    print("  -> a radial offset is NOT the Minkowski dilation: on an anisotropic")
    print("     shape the enforced surface sits CLOSER to the object than the")
    print("     nominal margin in the oblique directions.\n")

    print("=" * 78)
    print("C3  VOLUME FLOOR — can any correct cover beat the sphere on volume?")
    print("=" * 78)
    for name, o in OBJECTS.items():
        r = o["sphere_r"]
        v_sphere = volume_radial(lambda u: np.full(u.shape[:-1], r))
        print(f"\n  {name}  (production sphere r={r:.3f})")
        print(f"    {'variant':46s} {'volume':>9s} {'vs sphere':>10s}")
        rows = [("production sphere (UNDER-covers the object)", v_sphere)]
        for key, tag in (("sym", "SQ sym fit, boxy eps=0.4  [the n=200 arm]"),
                         ("mid", "SQ tight AABB fit, boxy eps=0.4"),
                         ("body_only", "SQ tight fit, main part only, eps=0.4")):
            if o[key] is None:
                continue
            rows.append((tag, volume_radial(lambda u, s=o[key]: sq_radius(u, s, 0.4))))
        for key, tag in (("mid", "ellipsoid eps=1.0, tight AABB fit"),
                         ("body_only", "ellipsoid eps=1.0, main part only")):
            if o[key] is None:
                continue
            rows.append((tag, volume_radial(lambda u, s=o[key]: sq_radius(u, s, 1.0))))
        # Minkowski-correct variant: inflate the semi-axes instead of the radius
        for key, tag in (("body_only", "ellipsoid, semi-axes inflated (Minkowski-safe)"),):
            if o[key] is None:
                continue
            s2 = [x + OFF for x in o[key]]
            rows.append((tag, volume_radial(lambda u, s=s2: sq_radius(u, s, 1.0), offset=0.0)))
        # smallest possible: a sphere that actually CONTAINS the object
        rmax = max(o["mid"])
        rows.append((f"smallest CONTAINING sphere (r={rmax:.3f})",
                     volume_radial(lambda u: np.full(u.shape[:-1], rmax))))
        for tag, v in rows:
            print(f"    {tag:46s} {v*1000:7.1f} L {v/v_sphere:9.2f}x")

    print("\n" + "=" * 78)
    print("READ")
    print("=" * 78)
    print("""  The production sphere UNDER-covers: at r=0.080 it does not contain a moka pot
  whose largest half-extent is 0.123. The smallest CONTAINING sphere costs 1.91x
  the volume — that is the price of staying isotropic.

  Shape is what buys the containment back. A tight ELLIPSOID on the pot's main
  part covers the body at 1.08x the production sphere's volume (1.12x with the
  Minkowski-safe semi-axis inflation). The arm that actually ran cost 1.99x.
  The gap is entirely fitting choices, none of which were the shape idea itself:
    sym mirroring about the body origin   1.70x -> 1.99x
    boxy eps=0.4 instead of ellipsoidal   1.32x -> 1.70x
    whole-object AABB incl. the handle    1.08x -> 1.32x
  So the n=200 result did not test shape. It tested a triply-inflated fit.""")


if __name__ == "__main__":
    main()
