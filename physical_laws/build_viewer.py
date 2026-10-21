"""Build the static viewer from the outputs of extract.py and separation.py.

usage: python physical_laws/build_viewer.py out/viewer out/sep/sep.json physical_laws/viewer/site [--font data/fonts/latinmodern-math.otf]

Writes into the site folder:
  index.html               viewer_src.html with the font inlined as a data URI, inside a minimal HTML page
  meta.json, sep.json      copied
  feat/<video><m>.json     base64 of the arrays in feat/<video><m>.bin (PCA RGB, |dz|, int8 coefficients, residual norm)
  frames/<video><m>.png    copied 8x4 frame sprites
<m> is "" for stored frames and "_clean" for cleaned frames.
Open with: python -m http.server -d physical_laws/viewer/site 8000, then http://localhost:8000/
"""
import argparse
import base64
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("viewer_data", help="output folder of extract.py")
ap.add_argument("sep_json", help="sep.json written by separation.py")
ap.add_argument("site")
ap.add_argument("--font", default="data/fonts/latinmodern-math.otf")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
T, NT = 32, 256
meta = json.load(open(f"{args.viewer_data}/meta.json"))
K = meta["k"]
for d in ("feat", "frames"):
    os.makedirs(f"{args.site}/{d}", exist_ok=True)
n = 0
for sc in meta["scenarios"]:
    for v in sc["videos"]:
        for tag in ("", "_clean"):
            b = open(f"{args.viewer_data}/feat/{v['id']}{tag}.bin", "rb").read()
            o, parts = 0, {}
            for name, size in (("rgb", T * NT * 3), ("dz", T * NT), ("coef", T * NT * K), ("res", T * NT)):
                parts[name] = base64.b64encode(b[o : o + size]).decode()
                o += size
            assert o == len(b), v["id"]
            json.dump(parts, open(f"{args.site}/feat/{v['id']}{tag}.json", "w"), separators=(",", ":"))
            shutil.copy(f"{args.viewer_data}/frames/{v['id']}{tag}.png", f"{args.site}/frames/{v['id']}{tag}.png")
            n += 1
shutil.copy(f"{args.viewer_data}/meta.json", f"{args.site}/meta.json")
shutil.copy(args.sep_json, f"{args.site}/sep.json")

src = open(f"{HERE}/viewer/viewer_src.html").read()
font = base64.b64encode(open(args.font, "rb").read()).decode()
assert src.count("/*LMFONT*/") == 1
page = src.replace("/*LMFONT*/", font)
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>:root { color-scheme: light; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); } '
        'body { margin: 0; } img { max-width: 100%; } [hidden] { display: none !important; }</style>\n</head>\n<body>\n')
open(f"{args.site}/index.html", "w").write(head + page + "\n</body>\n</html>\n")
print(f"{n} videos x modes, page {len(page) // 1000} kB -> {args.site}")
