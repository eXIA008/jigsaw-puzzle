import argparse
import json
import os
import time

import cv2
import numpy as np

INPUT_DIR = "hasil"
T_STRIP = 3
TAU = 2500.0
MAX_CANDIDATES = 5
SAVE = True
SHOW = True


def load_hint_meta(path):
    with open(path) as f:
        meta = json.load(f)
    meta["piece_size"] = tuple(meta["piece_size"])
    return meta


def load_ring(input_dir):
    ring = {}
    for side in ("top", "bottom", "left", "right"):
        path = os.path.join(input_dir, "ring", f"ring_{side}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        ring[side] = img
    return ring


def load_pieces(input_dir):
    pieces = {}
    pieces_dir = os.path.join(input_dir, "pieces")
    for name in sorted(os.listdir(pieces_dir)):
        if name.lower().endswith(".png"):
            img = cv2.imread(os.path.join(pieces_dir, name), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(os.path.join(pieces_dir, name))
            pieces[name] = img
    return pieces


def ring_slice(ring, side, idx, meta, t):
    k = meta["hint_width"]
    ph, pw = meta["piece_size"]
    if side == "top":
        return ring["top"][-t:, k + idx * pw:k + (idx + 1) * pw]
    if side == "bottom":
        return ring["bottom"][:t, k + idx * pw:k + (idx + 1) * pw]
    if side == "left":
        return ring["left"][k + idx * ph:k + (idx + 1) * ph, -t:]
    return ring["right"][k + idx * ph:k + (idx + 1) * ph, :t]


def extract_edge(img, side, t=3):
    if side == "top":
        return img[:t, ...]
    if side == "bottom":
        return img[-t:, ...]
    if side == "left":
        return img[:, :t, ...]
    return img[:, -t:, ...]


def edge_similarity(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(d * d))


def total_score(cand, ref_top, ref_left, ref_bottom=None, ref_right=None, t=3):
    s = 0.0
    if ref_top is not None:
        s += edge_similarity(extract_edge(cand, "top", t), ref_top)
    if ref_left is not None:
        s += edge_similarity(extract_edge(cand, "left", t), ref_left)
    if ref_bottom is not None:
        s += edge_similarity(extract_edge(cand, "bottom", t), ref_bottom)
    if ref_right is not None:
        s += edge_similarity(extract_edge(cand, "right", t), ref_right)
    return s


def solve(pieces, ring, meta, t=3, tau=TAU, max_candidates=MAX_CANDIDATES, steps=None):
    m = meta["m"]
    ph, pw = meta["piece_size"]
    n = m * m
    has_ring = meta["has_ring"]
    grid = [[None] * m for _ in range(m)]
    gaps = [None] * n
    confs = [None] * n
    used = set()

    rots = {}
    for pid, img in pieces.items():
        variants = []
        for rot in range(4):
            im = np.rot90(img, rot)
            if im.shape[:2] == (ph, pw):
                variants.append((rot, np.ascontiguousarray(im)))
        rots[pid] = variants

    def refs_for(r, c):
        top = left = bottom = right = None
        if r == 0:
            if has_ring:
                top = ring_slice(ring, "top", c, meta, t)
        else:
            top = extract_edge(grid[r - 1][c][2], "bottom", t)
        if c == 0:
            if has_ring:
                left = ring_slice(ring, "left", r, meta, t)
        else:
            left = extract_edge(grid[r][c - 1][2], "right", t)
        if r == m - 1 and has_ring:
            bottom = ring_slice(ring, "bottom", c, meta, t)
        if c == m - 1 and has_ring:
            right = ring_slice(ring, "right", r, meta, t)
        return top, left, bottom, right

    def backtrack(pos):
        if pos == n:
            return True
        r, c = divmod(pos, m)
        top, left, bottom, right = refs_for(r, c)
        n_refs = sum(x is not None for x in (top, left, bottom, right))
        cands = []
        for pid, variants in rots.items():
            if pid in used:
                continue
            for rot, im in variants:
                s = total_score(im, top, left, bottom, right, t)
                avg = s / n_refs if n_refs else 0.0
                if avg <= tau:
                    cands.append((avg, pid, rot, im))
        cands.sort(key=lambda x: x[0])
        gaps[pos] = cands[1][0] - cands[0][0] if len(cands) >= 2 else None
        if gaps[pos] is not None:
            best = cands[0][0]
            confs[pos] = gaps[pos] / (best + gaps[pos]) if (best + gaps[pos]) > 0 else 0.0
        for avg, pid, rot, im in cands[:max_candidates]:
            grid[r][c] = (pid, rot, im, avg)
            used.add(pid)
            if steps is not None:
                steps.append({"type": "place", "pos": pos, "pid": pid, "rot": rot, "score": avg})
            if backtrack(pos + 1):
                return True
            grid[r][c] = None
            used.remove(pid)
            if steps is not None:
                steps.append({"type": "unplace", "pos": pos, "pid": pid, "rot": rot})
        return False

    ok = backtrack(0)
    return grid, ok, gaps, confs


def reconstruct_image(grid, ring, meta):
    m = meta["m"]
    ph, pw = meta["piece_size"]
    has_ring = meta["has_ring"]
    k = meta["hint_width"] if has_ring else 0
    ref = next((cell[2] for row in grid for cell in row if cell is not None), None)
    if ref is None:
        return None
    h, w = 2 * k + m * ph, 2 * k + m * pw
    canvas = np.zeros((h, w) + ref.shape[2:], dtype=ref.dtype)
    if has_ring:
        canvas[:k, :] = ring["top"]
        canvas[-k:, :] = ring["bottom"]
        canvas[:, :k] = ring["left"]
        canvas[:, -k:] = ring["right"]
    for r in range(m):
        for c in range(m):
            if grid[r][c] is not None:
                canvas[k + r * ph:k + (r + 1) * ph, k + c * pw:k + (c + 1) * pw] = grid[r][c][2]
    return canvas


def show(reconstructed, input_dir):
    import matplotlib.pyplot as plt

    def rgb(x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if x.ndim == 3 else x

    kw = {} if reconstructed.ndim == 3 else {"cmap": "gray", "vmin": 0, "vmax": 255}
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(rgb(reconstructed), **kw)
    ax.set_title("Rekonstruksi solver")
    ax.axis("off")
    frame = cv2.imread(os.path.join(input_dir, "frame.png"), cv2.IMREAD_UNCHANGED)
    if frame is not None:
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(rgb(frame), **kw)
        ax.set_title("Bingkai generator")
        ax.axis("off")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=INPUT_DIR, help="folder hasil dari puzzle_generator.py")
    ap.add_argument("--t", type=int, default=T_STRIP, help="tebal strip pembanding")
    ap.add_argument("--tau", type=float, default=TAU, help="ambang rata-rata MSE per sisi")
    ap.add_argument("--max_candidates", type=int, default=MAX_CANDIDATES)
    ap.add_argument("--save", action=argparse.BooleanOptionalAction, default=SAVE)
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    a = ap.parse_args()

    meta = load_hint_meta(os.path.join(a.input_dir, "hint_meta.json"))
    if meta["placement"] != "split":
        raise SystemExit(f"placement '{meta['placement']}' belum didukung (hanya 'split')")

    m = meta["m"]
    pieces = load_pieces(a.input_dir)
    if len(pieces) != m * m:
        raise SystemExit(f"jumlah potongan {len(pieces)} != m*m = {m * m}")

    ring = load_ring(a.input_dir) if meta["has_ring"] else None
    ph, pw = meta["piece_size"]
    t = max(1, min(a.t, ph, pw))
    if meta["has_ring"]:
        t = min(t, meta["hint_width"])

    start = time.perf_counter()
    grid, ok, gaps, confs = solve(pieces, ring, meta, t=t, tau=a.tau, max_candidates=a.max_candidates)
    elapsed = time.perf_counter() - start

    solution = {}
    for r in range(m):
        for c in range(m):
            cell = grid[r][c]
            if cell is not None:
                pid, rot, im, s = cell
                solution[pid] = [r, c, ((4 - rot) % 4) * 90]

    gap_vals = [g for g in gaps if g is not None]
    conf_vals = [x for x in confs if x is not None]
    solve_meta = {
        "ok": ok,
        "time_s": round(elapsed, 3),
        "t": t,
        "tau": a.tau,
        "max_candidates": a.max_candidates,
        "gaps": gaps,
        "gap_mean": round(sum(gap_vals) / len(gap_vals), 3) if gap_vals else None,
        "confidences": confs,
        "confidence_mean": round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else None,
    }

    if a.save:
        with open(os.path.join(a.input_dir, "solution.json"), "w") as f:
            json.dump(solution, f, indent=2)
        with open(os.path.join(a.input_dir, "solve_meta.json"), "w") as f:
            json.dump(solve_meta, f, indent=2)
        rec = reconstruct_image(grid, ring, meta)
        if rec is not None:
            cv2.imwrite(os.path.join(a.input_dir, "reconstructed.png"), rec)

    print(f"Selesai {elapsed:.2f} s | ok={ok} | terisi {len(solution)}/{m * m} | gap_mean={solve_meta['gap_mean']}")
    if a.show:
        rec = reconstruct_image(grid, ring, meta)
        if rec is not None:
            show(rec, a.input_dir)


if __name__ == "__main__":
    main()
