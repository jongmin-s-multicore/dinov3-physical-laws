"""DINOv3 ViT-S/16 tokens of PhyWorld eval videos for the viewer (four scenarios).

usage: python physical_laws/extract.py out/viewer [--src data/phyworld] [--bg 80]

Scenarios: uniform motion (inertia), parabola (gravity), collision (elastic collision), combinatorial (PHYRE scenes,
out-of-template eval: gravity, collisions and contact between several objects). The first three come as 256 px, 32 frame
videos; combinatorial videos are 512 px, 50 frames and are cut to the first 32 frames and resized to 256 px (area).
Per scenario 6 videos are shown. For the first three: 2 ID, 2 OOD radius (small, large), 2 OOD velocity (slow, fast).
For combinatorial: one video from each of 6 templates, the one with the most pixel change among the first 10.

Each frame goes through DINOv3 ViT-S/16 at 256 px: 1 CLS + 4 register + 16x16 patch tokens of 384 dims
(last_hidden_state after the final norm). Two frame modes: as stored (H.264) and cleaned (pixels with all channels >= 235
set to 255). PCA RGB basis, clipping, the 32-d uncentred basis and the |dz| scale are fit on stored frames of all shown
videos and reused for cleaned frames.

Per-frame series per video and mode: mean |dz| over patches, |dCLS|, mean register |d|, red ball speed (stored frames),
and two transport measures over the patches near moving objects (pixel change > 40 in any channel, dilated by one patch):
  moved = 1 - sum_near min_{j' within 2 patches} |z_{t,j} - z_{t-1,j'}|^2 / sum_near |z_{t,j} - z_{t-1,j}|^2
  away  = share of sum_j |z_{t,j} - z_{t-1,j}|^2 that lies outside the near patches
Both are null when nothing moves.

Written to <out>: frames/<vid><m>.png (8x4 sprite), feat/<vid><m>.bin, meta.json; <m> is "" (stored) or "_clean".
"""
import argparse, json, os, subprocess
import h5py
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--src", default="data/phyworld", help="folder with the PhyWorld eval .hdf5 files")
ap.add_argument("--bg", type=int, default=80, help="extra videos per scenario for the CLS paths")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
SRC, OUT, dev = args.src, args.out, args.device
MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
G, NT, K, T, R = 16, 256, 32, 32, 2
for d in ("frames", "feat", "tmp"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
rng = np.random.default_rng(0)

SCEN = [
    ("uniform", "uniform_motion_eval", "Uniform motion", "Inertia: constant velocity, no gravity"),
    ("parabola", "parabola_eval", "Parabola", "Gravity: constant downward acceleration"),
    ("collision", "collision_eval", "Collision", "Elastic collision: momentum and energy conserved"),
    ("combinatorial", "combinatorial_out_of_template_eval_1K", "Combinatorial", "Several objects: gravity, collisions, contact"),
]
SLOTS = ["ID 1", "ID 2", "Small r", "Large r", "Slow v", "Fast v"]
TARGETS = {
    "parabola": [("ID", (1.1, 2.5)), ("ID", (0.8, 3.6)), ("OOD radius", (0.4, 2.5)), ("OOD radius", (1.9, 2.5)),
                 ("OOD velocity", (1.1, 0.4)), ("OOD velocity", (1.1, 5.5))],
    "collision": [("ID", (1.1, 1.1, 2.5, 2.5)), ("ID", (0.8, 1.4, 3.5, 1.5)), ("OOD radius", (0.4, 1.1, 2.5, 2.5)),
                  ("OOD radius", (1.9, 1.1, 2.5, 2.5)), ("OOD velocity", (1.1, 1.1, 0.4, 0.4)),
                  ("OOD velocity", (1.1, 1.1, 5.5, 5.5))],
}
TARGETS["uniform"] = TARGETS["parabola"]
COMB_TEMPLATES = ["10060", "10061", "10063", "10065", "10066", "10069"]


def category(p, scen):
    r, v = (p[:2], p[2:]) if scen == "collision" else (p[:1], p[1:])
    rin = bool(np.all((r >= 0.7) & (r <= 1.5)))
    vin = bool(np.all((v >= 1.0) & (v <= 4.0)))
    return {(True, True): "ID", (False, True): "OOD radius", (True, False): "OOD velocity", (False, False): "OOD both"}[(rin, vin)]


def decode(raw, big):
    path = f"{OUT}/tmp/v.mp4"
    open(path, "wb").write(raw)
    vf = ["-vf", "scale=256:256:flags=area", "-frames:v", str(T)] if big else []
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, *vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    v = np.frombuffer(out, np.uint8).reshape(-1, 256, 256, 3)
    assert v.shape[0] == T, v.shape
    return v


def clean(frames):
    out = frames.copy()
    out[frames.min(-1) >= 235] = 255
    return out


def red_mask(f):
    f = f.astype(np.int16)
    return (f[..., 0] > 150) & (f[..., 1] < 110) & (f[..., 2] < 110)


def blue_mask(f):
    f = f.astype(np.int16)
    return (f[..., 2] > 150) & (f[..., 0] < 100) & (f[..., 1] < 100)


def visible(frames):
    return (red_mask(frames).sum((1, 2)) >= 20) | (blue_mask(frames).sum((1, 2)) >= 20) | ((frames.min(-1) < 220).sum((1, 2)) >= 20)


def centres(m):
    ys, xs = np.mgrid[0:256, 0:256]
    n = m.sum((1, 2))
    c = np.stack([(m * xs).sum((1, 2)), (m * ys).sum((1, 2))], 1) / np.maximum(n, 1)[:, None]
    c[n < 20] = np.nan
    return c


model = AutoModel.from_pretrained(MODEL).to(dev).eval()


@torch.inference_mode()
def encode(frames):
    x = (frames.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
    h = torch.cat([model(pixel_values=x[i : i + 64]).last_hidden_state.float().cpu() for i in range(0, len(x), 64)]).numpy()
    assert h.shape[1] == 1 + 4 + NT, h.shape
    return h[:, 0], h[:, 1:5], h[:, 5:]


ys_, xs_ = np.divmod(np.arange(NT), G)
WIN = (np.abs(ys_[:, None] - ys_[None]) <= R) & (np.abs(xs_[:, None] - xs_[None]) <= R)


def transport_series(z, frames):
    """moved and away shares per frame (index t compares t-1 -> t)"""
    moved, away = [None], [None]
    f = frames.astype(np.int16)
    for t in range(1, T):
        pm = (np.abs(f[t] - f[t - 1]).max(-1) > 40).reshape(G, 16, G, 16).any((1, 3))
        near = pm.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sh = np.zeros_like(pm)
                sh[max(dy, 0) : G + min(dy, 0), max(dx, 0) : G + min(dx, 0)] = pm[max(-dy, 0) : G + min(-dy, 0), max(-dx, 0) : G + min(-dx, 0)]
                near |= sh
        near = near.ravel()
        a, b = z[t], z[t - 1]
        d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T
        inplace = np.diag(d2).clip(0)
        best = np.where(WIN, d2, np.inf).min(1).clip(0)
        if not near.any() or inplace[near].sum() < 1e-6:
            moved.append(None); away.append(None); continue
        moved.append(round(float(1 - best[near].sum() / inplace[near].sum()), 4))
        away.append(round(float(inplace[~near].sum() / inplace.sum()), 4))
    return moved, away


meta = {"model": MODEL, "input_px": 256, "grid": G, "k": K, "slots": SLOTS, "scenarios": []}
sel = []
for key, fname, label, law in SCEN:
    f = h5py.File(f"{SRC}/{fname}.hdf5", "r")
    comb = key == "combinatorial"
    if comb:
        index = sorted(f["video_streams"].keys())
        params = None
        cats = np.array(["Out of template"] * len(index))
        raw = lambda i: f["video_streams"][index[i]][0].tobytes()
        chosen = []
        for tp in COMB_TEMPLATES:
            cand = [i for i, k in enumerate(index) if k.startswith(tp + ":")][:10]
            score = [float(np.abs(np.diff(decode(raw(i), True).astype(np.int16), axis=0)).mean()) for i in cand]
            chosen.append(cand[int(np.argmax(score))])
        rest = [i for i in range(len(index)) if i not in chosen]
        bg = chosen + [int(i) for i in rng.choice(rest, min(args.bg, len(rest)), replace=False)]
    else:
        index, params = [], []
        for g in sorted(f["init_streams"]):
            p = f["init_streams"][g][:]
            index += [(g, i) for i in range(len(p))]
            params.append(p)
        params = np.concatenate(params)
        cats = np.array([category(p, key) for p in params])
        raw = lambda i: f["video_streams"][index[i][0]][index[i][1]].tobytes()
        chosen = []
        for cat, tgt in TARGETS[key]:
            ok = [i for i in np.where(cats == cat)[0] if i not in chosen]
            chosen.append(ok[int(np.argmin(np.abs(params[ok] - np.array(tgt)).sum(1)))])
        ids, oods = np.where(cats == "ID")[0], np.where(cats != "ID")[0]
        bg = list(rng.choice(ids, min(args.bg // 2, len(ids)), replace=False)) + list(rng.choice(oods, min(args.bg // 2, len(oods)), replace=False))
        bg = list(dict.fromkeys(chosen + [int(i) for i in bg]))
    cls_all, cls_all_c, vis_all, traj = [], [], [], []
    for i in bg:
        frames = decode(raw(i), comb)
        framesc = clean(frames)
        cls, reg, pat = encode(frames)
        clsc, regc, patc = encode(framesc)
        vis = visible(frames)
        cls_all.append(cls); cls_all_c.append(clsc); vis_all.append(vis)
        p = [] if comb else np.round(params[i], 3).tolist()
        src = f"{fname}/{index[i]}" if comb else f"{fname}/{index[i][0]}/{index[i][1]}"
        if i in chosen:
            n = chosen.index(i)
            sel.append(dict(scen=key, vid=f"{key}_{n}", slot=n, frames=frames, framesc=framesc, cls=cls, reg=reg, pat=pat,
                            clsc=clsc, regc=regc, patc=patc, params=p, cat=str(cats[i]), src=src,
                            label=(f"template {index[i].split(':')[0]}, video {int(index[i].split(':')[1])}" if comb else None)))
        traj.append(dict(i=int(i), cat=str(cats[i]), params=p, src=src))
    C = np.concatenate([c[v] for c, v in zip(cls_all, vis_all)])
    mu = C.mean(0)
    _, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    for t, c, cc, v in zip(traj, cls_all, cls_all_c, vis_all):
        for name, x in (("xy", c), ("xy_clean", cc)):
            xy = np.round((x - mu) @ Vt[:2].T, 3).tolist()
            t[name] = [q if ok else None for q, ok in zip(xy, v)]
    meta["scenarios"].append(dict(key=key, label=label, law=law, file=fname, cls_pca_var=np.round(ev[:2], 4).tolist(),
                                  cls_traj=traj, videos=[]))
    print(key, "chosen", [(str(cats[i]), (index[i] if comb else np.round(params[i], 2).tolist())) for i in chosen], "bg", len(bg), flush=True)

# ---- bases on stored frames of all shown videos
P = np.concatenate([s["pat"].reshape(-1, 384) for s in sel])
pm = P.mean(0)
w, V = np.linalg.eigh(np.cov((P - pm).T))
order = np.argsort(w)[::-1]
w, V = w[order], V[:, order]
pc3 = (P - pm) @ V[:, :3]
lo, hi = np.percentile(pc3, 1, 0), np.percentile(pc3, 99, 0)
_, _, Vt2 = np.linalg.svd(P[rng.choice(len(P), min(len(P), 60000), replace=False)], full_matrices=False)
B = Vt2[:K].T
coef = P @ B
energy = float((coef ** 2).sum() / (P ** 2).sum())
cscale = np.abs(coef).max(0) / 127.0
resid = np.sqrt(np.maximum((P ** 2).sum(1) - (coef ** 2).sum(1), 0))
rscale = float(resid.max() / 255.0) or 1.0
idx = rng.choice(len(P), 4000, replace=False)
jdx = (idx // NT) * NT + rng.integers(0, NT, len(idx))
ca = np.round(coef[idx] / cscale).clip(-127, 127) * cscale
cb = np.round(coef[jdx] / cscale).clip(-127, 127) * cscale
ra, rb = np.round(resid[idx] / rscale) * rscale, np.round(resid[jdx] / rscale) * rscale
qcos = (ca * cb).sum(1) / np.sqrt(((ca ** 2).sum(1) + ra ** 2) * ((cb ** 2).sum(1) + rb ** 2))
exact = (P[idx] * P[jdx]).sum(1) / (np.linalg.norm(P[idx], axis=1) * np.linalg.norm(P[jdx], axis=1))
err = np.abs(qcos - exact)

for s in sel:
    for m in ("", "c"):
        z = s["pat" + m]
        dz = np.zeros((T, NT), np.float32)
        dz[1:] = np.linalg.norm(z[1:] - z[:-1], axis=2)
        s["dz" + m] = dz
dscale = float(np.percentile(np.concatenate([s["dz"][1:].ravel() for s in sel]), 99.5) / 255.0)

# background change with and without codec noise (patches at least one patch away from any moving object)
nz = {"bg": [], "bg_clean": [], "blank": [], "blank_clean": []}
for s in sel:
    f = s["frames"].astype(np.int16)
    obj = (f.min(-1) < 220).reshape(T, G, 16, G, 16).any((2, 4))
    objd = obj.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            objd |= np.roll(np.roll(obj, dy, 1), dx, 2)
    vis = obj.reshape(T, -1).any(1)
    near = (objd[1:] | objd[:-1]).reshape(T - 1, -1)
    both, blank = vis[1:] & vis[:-1], ~vis[1:] & ~vis[:-1]
    for m, k in (("", "bg"), ("c", "bg_clean")):
        d = s["dz" + m][1:]
        if both.any() and (~near[both]).any():
            nz[k].append(d[both][~near[both]])
        if blank.any():
            nz[k.replace("bg", "blank")].append(d[blank].ravel())
noise = {k: round(float(np.concatenate(v).mean()), 3) for k, v in nz.items() if v}
print("noise check", noise, flush=True)


def pack(s, m):
    z = s["pat" + m].reshape(-1, 384)
    rgb = ((((z - pm) @ V[:, :3]) - lo) / (hi - lo)).clip(0, 1)
    rgb = (rgb * 255).round().astype(np.uint8).reshape(T, NT, 3)
    cf = z @ B
    rs_ = np.sqrt(np.maximum((z ** 2).sum(1) - (cf ** 2).sum(1), 0))
    c8 = np.round(cf / cscale).clip(-127, 127).astype(np.int8).reshape(T, NT, K)
    r8 = np.round(rs_ / rscale).clip(0, 255).astype(np.uint8).reshape(T, NT)
    d8 = np.round(s["dz" + m] / dscale).clip(0, 255).astype(np.uint8)
    tag = "" if m == "" else "_clean"
    with open(f"{OUT}/feat/{s['vid']}{tag}.bin", "wb") as fh:
        fh.write(rgb.tobytes()); fh.write(d8.tobytes()); fh.write(c8.tobytes()); fh.write(r8.tobytes())
    fr = s["frames" if m == "" else "framesc"]
    Image.fromarray(fr.reshape(4, 8, 256, 256, 3).transpose(0, 2, 1, 3, 4).reshape(1024, 2048, 3)).save(f"{OUT}/frames/{s['vid']}{tag}.png", optimize=True)


j = lambda a: [None if (x is None or not np.isfinite(x)) else round(float(x), 4) for x in a]
for s in sel:
    pack(s, "")
    pack(s, "c")
    red = centres(red_mask(s["frames"]))
    blue = centres(blue_mask(s["frames"]))
    sp = np.full(T, np.nan)
    sp[1:] = np.linalg.norm(red[1:] - red[:-1], axis=1)
    ser = {"speed": j(sp)}
    for m, tag in (("", ""), ("c", "_clean")):
        cls, reg = s["cls" + m], s["reg" + m]
        dcls = np.zeros(T); dcls[1:] = np.linalg.norm(cls[1:] - cls[:-1], axis=1)
        dreg = np.zeros(T); dreg[1:] = np.linalg.norm(reg[1:] - reg[:-1], axis=2).mean(1)
        moved, away = transport_series(s["pat" + m], s["frames"])
        ser.update({"dz_mean" + tag: j(s["dz" + m].mean(1)), "dcls" + tag: j(dcls), "dreg" + tag: j(dreg),
                    "moved" + tag: moved, "away" + tag: away})
    q0 = red[0] if np.all(np.isfinite(red[0])) else np.array([128.0, 128.0])
    sc = next(x for x in meta["scenarios"] if x["key"] == s["scen"])
    sc["videos"].append(dict(id=s["vid"], slot=s["slot"], cat=s["cat"], params=s["params"], label=s["label"], src=s["src"],
                             query=[int(q0[1] // 16), int(q0[0] // 16)], series=ser))
for sc in meta["scenarios"]:
    sc["videos"].sort(key=lambda v: v["slot"])

meta.update(dict(rgb_pca_var=np.round(w[:3] / w.sum(), 4).tolist(), sim_energy=round(energy, 4),
                 sim_cos_err_mean=round(float(err.mean()), 4), sim_cos_err_p99=round(float(np.percentile(err, 99)), 4),
                 cscale=np.round(cscale, 6).tolist(), rscale=rscale, dscale=dscale, window=R,
                 id_ranges=dict(radius=[0.7, 1.5], velocity=[1.0, 4.0]), noise=noise, clean_threshold=235))
json.dump(meta, open(f"{OUT}/meta.json", "w"), separators=(",", ":"))
print("rgb pca var", meta["rgb_pca_var"], "sim energy", round(energy, 4), "cos err", meta["sim_cos_err_mean"], meta["sim_cos_err_p99"], "dscale", dscale)
for sc in meta["scenarios"]:
    for v in sc["videos"]:
        mv = [x for x in v["series"]["moved"] if x is not None]
        aw = [x for x in v["series"]["away"] if x is not None]
        print(v["id"], v["cat"], "moved mean", round(float(np.mean(mv)), 3) if mv else None, "away mean", round(float(np.mean(aw)), 3) if aw else None)
