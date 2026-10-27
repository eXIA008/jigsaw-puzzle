import argparse
import json
import os
import random

import cv2
import numpy as np

import matplotlib.pyplot as plt

INPUT_PATH = "image.jpg"
M = 3
HINT_WIDTH = 100
HINT_PLACEMENT = "split"
FIT_MODE = "crop"
OUTPUT_DIR = "hasil"
ROTATE = True
SAVE_FILES = True
SHOW = True
SAVE_TRUTH = True
SEED = None


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def fit_to_grid(img, m, mode=None, k=0):
    mode = mode or FIT_MODE
    h, w = img.shape[:2]
    h2, w2 = 2 * k + ((h - 2 * k) // m) * m, 2 * k + ((w - 2 * k) // m) * m
    if h - 2 * k < m or w - 2 * k < m:
        raise ValueError("Gambar terlalu kecil untuk M / HINT_WIDTH ini")
    if mode == "resize":
        h2, w2 = 2 * k + max(m, round((h - 2 * k) / m) * m), 2 * k + max(m, round((w - 2 * k) / m) * m)
        out = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
    else:
        out = img[:h2, :w2].copy()
    print(f"Ukuran asli {w}x{h} -> dipakai {w2}x{h2} (mode {mode}, selisih {w - w2} x {h - h2} px)")
    return out


def clamp_hint(img, m, k):
    h, w = img.shape[:2]
    kmax = max(1, min(h // m, w // m) // 4)
    if k > kmax:
        print(f"PERINGATAN: HINT_WIDTH={k} terlalu besar untuk M={m}; dipakai {kmax} (maks 1/4 sisi potongan)")
        return kmax
    return k


def extract_ring(img, k):
    if k <= 0:
        return None
    return {
        "top":    img[:k, :].copy(),
        "bottom": img[-k:, :].copy(),
        "left":   img[:, :k].copy(),
        "right":  img[:, -k:].copy(),
    }


def make_frame(img, m, hint_width=2, placement=None, grid_gray=170):
    placement = placement or HINT_PLACEMENT
    h, w = img.shape[:2]
    k = hint_width
    g = (grid_gray,) * 3 if img.ndim == 3 else grid_gray
    pad = k if (k > 0 and placement == "outside") else 0
    frame = np.full((h + 2 * pad, w + 2 * pad) + img.shape[2:], 255, np.uint8)

    split = placement == "split" and k > 0
    ox, oy = (k, k) if split else (pad, pad)
    gw, gh = (w - 2 * k, h - 2 * k) if split else (w, h)
    for i in range(1, m):
        x, y = ox + i * gw // m, oy + i * gh // m
        cv2.line(frame, (x, oy), (x, oy + gh - 1), g, 1)
        cv2.line(frame, (ox, y), (ox + gw - 1, y), g, 1)
    if split:
        cv2.rectangle(frame, (ox, oy), (ox + gw - 1, oy + gh - 1), g, 1)
    if k == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (120,) * 3 if img.ndim == 3 else 120, 1)
        return frame

    if placement == "outside":
        frame[:k, k:k + w] = img[:k, :]
        frame[k + h:, k:k + w] = img[-k:, :]
        frame[k:k + h, :k] = img[:, :k]
        frame[k:k + h, k + w:] = img[:, -k:]
        frame[:k, :k] = img[:k, :k]
        frame[:k, k + w:] = img[:k, -k:]
        frame[k + h:, :k] = img[-k:, :k]
        frame[k + h:, k + w:] = img[-k:, -k:]
    else:
        frame[:k, :] = img[:k, :]
        frame[-k:, :] = img[-k:, :]
        frame[:, :k] = img[:, :k]
        frame[:, -k:] = img[:, -k:]
    return frame


def cut_pieces(img, m, k=0):
    inner = img[k:img.shape[0] - k, k:img.shape[1] - k] if k else img
    h, w = inner.shape[:2]
    ph, pw = h // m, w // m
    pieces = []
    for r in range(m):
        for c in range(m):
            pieces.append(((r, c), inner[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw].copy()))
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_PATH)
    ap.add_argument("--m", type=int, default=M, help="jumlah potongan per sisi (grid M x M)")
    ap.add_argument("--hint_width", type=int, default=HINT_WIDTH, help="0 = bingkai kosong murni; 1-3 = cincin piksel tepi")
    ap.add_argument("--rotate", action="store_true", default=ROTATE, help="(opsional) rotasi acak 0/90/180/270 pada potongan")
    ap.add_argument("--save_truth", action="store_true", default=SAVE_TRUTH, help="simpan ground_truth.json (HANYA untuk evaluasi, jangan dibaca solver / jangan dikumpulkan)")
    ap.add_argument("--save_files", action="store_true", default=SAVE_FILES, help="simpan frame + potongan + ring + hint_meta ke folder")
    ap.add_argument("--no_show", action="store_true", default=not SHOW, help="jangan tampilkan jendela")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUTPUT_DIR)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    raw = load_image(a.input)
    a.hint_width = clamp_hint(raw, a.m, a.hint_width)
    ks = a.hint_width if HINT_PLACEMENT == "split" else 0
    img = fit_to_grid(raw, a.m, k=ks)
    h, w = img.shape[:2]
    frame = make_frame(img, a.m, a.hint_width)

    ring = extract_ring(img, ks)

    if a.save_files or a.save_truth:
        os.makedirs(os.path.join(a.out, "pieces"), exist_ok=True)
    if a.save_files:
        cv2.imwrite(os.path.join(a.out, "frame.png"), frame)
        if ring is not None:
            ring_dir = os.path.join(a.out, "ring")
            os.makedirs(ring_dir, exist_ok=True)
            for side, arr in ring.items():
                cv2.imwrite(os.path.join(ring_dir, f"ring_{side}.png"), arr)

    pieces = cut_pieces(img, a.m, ks)
    rng.shuffle(pieces)
    ids = rng.sample(range(10 ** 4, 10 ** 5), len(pieces))

    ph, pw = pieces[0][1].shape[:2]
    truth = {"image_size": [h, w], "m": a.m, "hint_width": a.hint_width, "placement": HINT_PLACEMENT,
             "piece_size": [ph, pw], "grid_origin_xy": [ks, ks], "pieces": {}}
    shown = []
    for pid, ((r, c), p) in zip(ids, pieces):
        rot = rng.choice([0, 90, 180, 270]) if a.rotate else 0
        if rot:
            p = np.rot90(p, k=rot // 90, axes=(1, 0)).copy()
        name = f"piece_{pid}.png"
        if a.save_files:
            cv2.imwrite(os.path.join(a.out, "pieces", name), p)
        shown.append(p)
        truth["pieces"][name] = {"row": r, "col": c, "rotation_cw": rot}

    if a.save_truth:
        with open(os.path.join(a.out, "ground_truth.json"), "w") as f:
            json.dump(truth, f, indent=2)

    if a.save_files:
        hint_meta = {
            "m": a.m,
            "hint_width": a.hint_width,
            "placement": HINT_PLACEMENT,
            "piece_size": [ph, pw],
            "has_ring": ring is not None,
        }
        with open(os.path.join(a.out, "hint_meta.json"), "w") as f:
            json.dump(hint_meta, f, indent=2)

    print(f"OK: {len(pieces)} potongan" + (f" disimpan -> {a.out}/" if a.save_files else " (tidak disimpan)"))
    if not a.no_show:
        show(img, frame, shown, a.m)


def show(img, frame, pieces, m):

    def rgb(x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if x.ndim == 3 else x
    kw = {} if img.ndim == 3 else {"cmap": "gray", "vmin": 0, "vmax": 255}

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    ax = fig.add_subplot(gs[0]); ax.imshow(rgb(img), **kw); ax.set_title("Gambar asli"); ax.axis("off")
    ax = fig.add_subplot(gs[1]); ax.imshow(rgb(frame), **kw); ax.set_title("Bingkai"); ax.axis("off")
    sub = gs[2].subgridspec(m, m, wspace=0.08, hspace=0.08)
    for i, p in enumerate(pieces):
        ax = fig.add_subplot(sub[i // m, i % m]); ax.imshow(rgb(p), **kw); ax.axis("off")
    fig.text(0.78, 0.93, f"Potongan ({len(pieces)}, diacak)", ha="center")
    plt.show()


if __name__ == "__main__":
    main()