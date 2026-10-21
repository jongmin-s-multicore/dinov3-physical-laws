"""How smooth are token paths compared with pixel paths: ||d2|| / ||d1|| and cos(d_t, d_t+1).

usage: python physical_laws/jitter.py out/sep/sep_features.npz out/sep/jitter.json [--src data/phyworld]

Token side: the features written by separation.py (cleaned and stored frames; CLS token and mean patch token), where
d1 = z_{t+1} - z_t, d2 = z_{t+2} - 2 z_{t+1} + z_t and the next change is d1 + d2.
Pixel side: ball centres from the PhyWorld position streams of every ID video, in pixels (25.6 px per world unit at
256 px), at steps where the centre is at least one radius inside the image. Collision videos give one path per ball.
A path at constant velocity has ||d2|| / ||d1|| = 0 and cos(d_t, d_t+1) = 1. If every frame adds independent noise
that dominates the change, ||d2|| / ||d1|| = sqrt(3) = 1.73 and the cosine is -0.5.
"""
import argparse
import json

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("features")
ap.add_argument("out")
ap.add_argument("--src", default="data/phyworld", help="folder with the PhyWorld eval .hdf5 files")
args = ap.parse_args()

LAWS = [("uniform", "uniform_motion_eval", False), ("parabola", "parabola_eval", False), ("collision", "collision_eval", True)]
PX = 25.6
res = {"token": {}, "pixel": {}}

d = np.load(args.features)
lab = d["lab"]
for m in ("clean", "stored"):
    res["token"][m] = {}
    for k in ("cls", "mean"):
        a = d[f"{m}_{k}_d1"]
        b = a + d[f"{m}_{k}_d2"]
        n1, n2 = np.linalg.norm(a, axis=1), np.linalg.norm(d[f"{m}_{k}_d2"], axis=1)
        cos = (a * b).sum(1) / (n1 * np.linalg.norm(b, axis=1))
        res["token"][m][k] = {law: {"d1": round(float(n1[lab == i].mean()), 3), "d2": round(float(n2[lab == i].mean()), 3),
                                    "ratio": round(float(np.median(n2[lab == i] / n1[lab == i])), 3),
                                    "cos": round(float(cos[lab == i].mean()), 3)}
                              for i, (law, _, _) in enumerate(LAWS)}

for law, fname, col in LAWS:
    f = h5py.File(f"{args.src}/{fname}.hdf5", "r")
    P = np.concatenate([f["init_streams"][g][:] for g in sorted(f["init_streams"])])
    X = np.concatenate([f["position_streams"][g][:] for g in sorted(f["position_streams"])])
    r, v = (P[:, :2], P[:, 2:]) if col else (P[:, :1], P[:, 1:])
    idm = np.all((r >= 0.7) & (r <= 1.5), 1) & np.all((v >= 1) & (v <= 4), 1)
    X, r = X[idm] * PX, r[idm] * PX
    paths = [(X[:, :, b], r[:, b]) for b in range(2)] if col else [(X, r[:, 0])]
    d1s, d2s, cs = [], [], []
    for x, rad in paths:
        d1 = x[:, 1:-1] - x[:, :-2]
        d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        nxt = d1 + d2
        lo, hi = rad[:, None], 256 - rad[:, None]
        ok = np.ones(d1.shape[:2], bool)
        for s in (slice(0, -2), slice(1, -1), slice(2, None)):
            ok &= np.all((x[:, s] > lo[..., None]) & (x[:, s] < hi[..., None]), axis=2)
        n1, n2, nn = np.linalg.norm(d1, axis=2), np.linalg.norm(d2, axis=2), np.linalg.norm(nxt, axis=2)
        ok &= n1 > 1e-6
        d1s.append(n1[ok]); d2s.append(n2[ok]); cs.append(((d1 * nxt).sum(2) / (n1 * np.maximum(nn, 1e-9)))[ok])
    n1, n2, c = np.concatenate(d1s), np.concatenate(d2s), np.concatenate(cs)
    res["pixel"][law] = {"d1": round(float(n1.mean()), 3), "d2": round(float(n2.mean()), 3),
                         "ratio": round(float(np.median(n2 / n1)), 4), "cos": round(float(c.mean()), 4)}

json.dump(res, open(args.out, "w"), indent=1)
for law in res["pixel"]:
    t = res["token"]["clean"]["cls"][law]
    p = res["pixel"][law]
    print(f"{law:9s} pixel ratio {p['ratio']:.3f} cos {p['cos']:+.3f} | token (clean CLS) ratio {t['ratio']:.2f} cos {t['cos']:+.2f}")
