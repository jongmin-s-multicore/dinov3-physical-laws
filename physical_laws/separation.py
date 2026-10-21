"""Do DINOv3 tokens separate by physical law? PhyWorld single-law sets only: uniform motion, parabola, collision.

usage: python physical_laws/separation.py out/sep [--src data/phyworld] [--limit N]

Videos: every ID video of the three eval files (radius 0.7-1.5, velocity 1-4). Frames: stored (H.264) and cleaned
(pixels with all channels >= 235 set to 255). Tokens: CLS and the mean of the 256 patch tokens (DINOv3 ViT-S/16, 256 px).
Samples: frame t where t, t+1, t+2 all have every object fully inside the image (no object pixel within 2 px of the border).
Features per sample: state z_t, first difference d1 = z_{t+1} - z_t, second difference d2 = z_{t+2} - 2 z_{t+1} + z_t.

1. PCA: 2-D projection of each feature, fit on the pooled samples of the three laws (no labels).
2. Linear probe: logistic regression on standardised features reduced to 64 PCs, 5-fold cross-validation split by video.
   Balanced accuracy for the 3-way law label and for uniform motion vs parabola.
3. Cross-law dynamics: ridge regression z_{t+2} ~ [z_{t+1}, z_t] in a 64-PC space, trained on 70 % of the videos of one
   law and tested on the other 30 % of each law. Error = mean |pred - z_{t+2}|^2 / mean |z_{t+1} - z_{t+2}|^2
   (1 = as bad as copying the last frame). Extra row: linear extrapolation 2 z_{t+1} - z_t without training.
Written: <out>/sep.json (page data) and <out>/sep_features.npz (all features, for reuse).
"""
import argparse, json, os, subprocess
import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--src", default="data/phyworld", help="folder with the PhyWorld eval .hdf5 files")
ap.add_argument("--limit", type=int, default=0, help="use at most this many ID videos per law (0 = all)")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
SRC, OUT = args.src, args.out
os.makedirs(OUT, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
T = 32
LAWS = [("uniform", "uniform_motion_eval", False), ("parabola", "parabola_eval", False), ("collision", "collision_eval", True)]
rng = np.random.default_rng(0)
model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(args.device).eval()


def decode(raw):
    path = f"{OUT}/v.mp4"
    open(path, "wb").write(raw)
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, 256, 256, 3)


@torch.inference_mode()
def encode(frames):
    x = torch.from_numpy((frames.astype(np.float32) / 255.0 - MEAN) / STD).permute(0, 3, 1, 2).to(args.device)
    h = model(pixel_values=x).last_hidden_state.float().cpu().numpy()
    return h[:, 0], h[:, 5:].mean(1)


def is_id(p, col):
    r, v = (p[:2], p[2:]) if col else (p[:1], p[1:])
    return bool(np.all((r >= 0.7) & (r <= 1.5)) and np.all((v >= 1) & (v <= 4)))


def inside(frames):
    obj = frames.min(-1) < 220
    present = obj.sum((1, 2)) >= 20
    edge = obj[:, :2].any((1, 2)) | obj[:, -2:].any((1, 2)) | obj[:, :, :2].any((1, 2)) | obj[:, :, -2:].any((1, 2))
    return present & ~edge


rows = []  # (law index, video id, t, cls stored/clean, mean stored/clean) collected as arrays
feats = {m: {k: {f: [] for f in ("z", "d1", "d2", "z1", "z2")} for k in ("cls", "mean")} for m in ("stored", "clean")}
lab, grp = [], []
nvid = {}
vid_counter = 0
for li, (law, fname, col) in enumerate(LAWS):
    f = h5py.File(f"{SRC}/{fname}.hdf5", "r")
    n = 0
    todo = [(g, j) for g in sorted(f["init_streams"]) for j, p in enumerate(f["init_streams"][g][:]) if is_id(p, col)]
    if args.limit:
        todo = todo[: args.limit]
    for g, j in todo:
        fr = decode(f["video_streams"][g][j].tobytes())
        frc = fr.copy()
        frc[fr.min(-1) >= 235] = 255
        ok = inside(fr)
        ts = [t for t in range(T - 2) if ok[t] and ok[t + 1] and ok[t + 2]]
        if not ts:
            continue
        enc = {"stored": encode(fr), "clean": encode(frc)}
        for m, (c, mp) in enc.items():
            for k, z in (("cls", c), ("mean", mp)):
                zz = z[ts]; z1 = z[[t + 1 for t in ts]]; z2 = z[[t + 2 for t in ts]]
                feats[m][k]["z"].append(zz); feats[m][k]["z1"].append(z1); feats[m][k]["z2"].append(z2)
                feats[m][k]["d1"].append(z1 - zz); feats[m][k]["d2"].append(z2 - 2 * z1 + zz)
        lab += [li] * len(ts)
        grp += [vid_counter] * len(ts)
        vid_counter += 1
        n += 1
    nvid[law] = n
    print(law, "videos", n, flush=True)
lab, grp = np.array(lab), np.array(grp)
for m in feats:
    for k in feats[m]:
        for f_ in feats[m][k]:
            feats[m][k][f_] = np.concatenate(feats[m][k][f_]).astype(np.float32)
np.savez_compressed(f"{OUT}/sep_features.npz", lab=lab, grp=grp,
                    **{f"{m}_{k}_{f_}": feats[m][k][f_] for m in feats for k in feats[m] for f_ in feats[m][k]})

res = {"laws": [l[0] for l in LAWS], "n_videos": nvid, "n_samples": {LAWS[i][0]: int((lab == i).sum()) for i in range(3)}}
gkf = GroupKFold(n_splits=5)
vids = np.unique(grp)
test_vids = set(rng.choice(vids, int(len(vids) * 0.3), replace=False).tolist())
for i in range(3):  # only matters for small --limit runs: every law needs a test video and a training video
    lv = [int(v) for v in np.unique(grp[lab == i])]
    tv = [v for v in lv if v in test_vids]
    if len(lv) >= 2 and not tv:
        test_vids.add(lv[0])
    elif len(lv) >= 2 and len(tv) == len(lv):
        test_vids.discard(lv[0])
is_test = np.array([g in test_vids for g in grp])
for m in ("stored", "clean"):
    res[m] = {}
    for k in ("cls", "mean"):
        out = {"scatter": {}, "probe": {}}
        for f_ in ("z", "d1", "d2"):
            X = feats[m][k][f_]
            p = PCA(2).fit(X)
            xy = p.transform(X)
            pts = {}
            for i in range(3):
                idx = np.where(lab == i)[0]
                idx = rng.choice(idx, min(700, len(idx)), replace=False)
                pts[LAWS[i][0]] = np.round(xy[idx], 3).tolist()
            out["scatter"][f_] = {"var": np.round(p.explained_variance_ratio_, 4).tolist(), "pts": pts}
            acc3, accup = [], []
            for tr, te in gkf.split(X, lab, grp):
                clf = make_pipeline(StandardScaler(), PCA(min(64, len(tr) - 1), random_state=0), LogisticRegression(C=1.0, max_iter=3000))
                clf.fit(X[tr], lab[tr])
                acc3.append(balanced_accuracy_score(lab[te], clf.predict(X[te])))
                trm, tem = tr[lab[tr] < 2], te[lab[te] < 2]
                if len(tem) == 0 or len(np.unique(lab[trm])) < 2:
                    continue
                clf2 = make_pipeline(StandardScaler(), PCA(min(64, len(trm) - 1), random_state=0), LogisticRegression(C=1.0, max_iter=3000))
                clf2.fit(X[trm], lab[trm])
                accup.append(balanced_accuracy_score(lab[tem], clf2.predict(X[tem])))
            out["probe"][f_] = {"acc3": round(float(np.mean(acc3)), 4), "acc3_sd": round(float(np.std(acc3)), 4),
                                "up": round(float(np.mean(accup)), 4), "up_sd": round(float(np.std(accup)), 4)}
            print(m, k, f_, out["probe"][f_], flush=True)
        # cross-law dynamics
        Z, Z1, Z2 = feats[m][k]["z"], feats[m][k]["z1"], feats[m][k]["z2"]
        pca = PCA(64, random_state=0).fit(np.concatenate([Z, Z1, Z2]))
        a, b, c = pca.transform(Z), pca.transform(Z1), pca.transform(Z2)
        X2, Y2 = np.concatenate([b, a], 1), c
        mat = np.zeros((3, 3))
        for i in range(3):
            trm = (lab == i) & ~is_test
            reg = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(X2[trm], Y2[trm])
            for j in range(3):
                tem = (lab == j) & is_test
                pred = reg.predict(X2[tem])
                mat[i, j] = ((pred - Y2[tem]) ** 2).sum(1).mean() / ((b[tem] - Y2[tem]) ** 2).sum(1).mean()
        extrap = []
        for j in range(3):
            tem = (lab == j) & is_test
            extrap.append(float(((2 * b[tem] - a[tem] - Y2[tem]) ** 2).sum(1).mean() / ((b[tem] - Y2[tem]) ** 2).sum(1).mean()))
        out["xlaw"] = {"train_test": np.round(mat, 4).tolist(), "extrap": np.round(extrap, 4).tolist()}
        print(m, k, "xlaw", np.round(mat, 3).tolist(), "extrap", np.round(extrap, 3).tolist(), flush=True)
        res[m][k] = out
json.dump(res, open(f"{OUT}/sep.json", "w"), separators=(",", ":"))
print("done", res["n_samples"])
