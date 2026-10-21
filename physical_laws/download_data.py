"""Download the PhyWorld eval sets and the Latin Modern Math font used by the viewer.

usage: python physical_laws/download_data.py [--out data]

PhyWorld (Kang et al., ICML 2025) is on Hugging Face as magicr/phyworld (CC-BY-4.0). Only the eval files are needed:
uniform motion, parabola and collision eval sets (in-distribution and out-of-distribution videos together) and the
out-of-template eval set of the combinatorial (PHYRE) scenes. Total about 80 MB.
The font is the unmodified latinmodern-math.otf from CTAN (GUST Font License).
"""
import argparse
import os
import urllib.request

HF = "https://huggingface.co/datasets/magicr/phyworld/resolve/main/"
FILES = {
    "phyworld/uniform_motion_eval.hdf5": HF + "id_ood_data/uniform_motion_eval.hdf5",
    "phyworld/parabola_eval.hdf5": HF + "id_ood_data/parabola_eval.hdf5",
    "phyworld/collision_eval.hdf5": HF + "id_ood_data/collision_eval.hdf5",
    "phyworld/combinatorial_out_of_template_eval_1K.hdf5": HF + "combinatorial_data/combinatorial_out_of_template_eval_1K.hdf5",
    "fonts/latinmodern-math.otf": "https://mirrors.ctan.org/fonts/lm-math/opentype/latinmodern-math.otf",
}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
args = ap.parse_args()
for rel, url in FILES.items():
    path = os.path.join(args.out, rel)
    if os.path.exists(path):
        print("have", path)
        continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("get ", path)
    urllib.request.urlretrieve(url, path + ".part")
    os.replace(path + ".part", path)
