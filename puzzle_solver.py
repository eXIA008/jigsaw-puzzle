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
MAX_NODES = 200000
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


def _as_rows(strip, side):
    s = strip.astype(np.float32)
    return np.swapaxes(s, 0, 1) if side in ("left", "right") else s


def seam_cost(before, after, side_before):
    a = _as_rows(before, side_before)
    b = _as_rows(after, "top" if side_before == "bottom" else "left")
    d = a - b
    return float(np.mean(d * d))


def seam_parts(cand, r, c, grid, ring, meta, t):
    m = meta["m"]
    tk = min(t, meta["hint_width"]) if ring else t
    parts = []
    if r > 0:
        parts.append(seam_cost(extract_edge(grid[r - 1][c][2], "bottom", t),
                               extract_edge(cand, "top", t), "bottom"))
    elif ring:
        parts.append(seam_cost(ring_slice(ring, "top", c, meta, tk),
                               extract_edge(cand, "top", tk), "bottom"))
    if c > 0:
        parts.append(seam_cost(extract_edge(grid[r][c - 1][2], "right", t),
                               extract_edge(cand, "left", t), "right"))
    elif ring:
        parts.append(seam_cost(ring_slice(ring, "left", r, meta, tk),
                               extract_edge(cand, "left", tk), "right"))
    if r == m - 1 and ring:
        parts.append(seam_cost(extract_edge(cand, "bottom", tk),
                               ring_slice(ring, "bottom", c, meta, tk), "bottom"))
    if c == m - 1 and ring:
        parts.append(seam_cost(extract_edge(cand, "right", tk),
                               ring_slice(ring, "right", r, meta, tk), "right"))
    return parts


def total_score(cand, r, c, grid, ring, meta, t):
    costs = seam_parts(cand, r, c, grid, ring, meta, t)
    return float(np.mean(costs)) if costs else 0.0


def _grid_seam_mean(grid, ring, meta, t):
    costs = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell is not None:
                costs += seam_parts(cell[2], r, c, grid, ring, meta, t)
    return float(np.mean(costs)) if costs else 0.0


class _Budget(Exception):
    pass


def _attempt(pieces, ring, meta, tau, max_candidates, t, max_nodes, steps=None, stop_mean=None):
    m = meta["m"]
    shape0 = tuple(meta["piece_size"])
    grid = [[None] * m for _ in range(m)]
    used = set()
    best = {"depth": -1, "grid": None}
    nodes = [0]
    budget_hit = [False]
    best_completion = [None, None, None]
    gap_map = {}

    def backtrack(pos):
        if pos > best["depth"]:
            best["depth"] = pos
            best["grid"] = [row[:] for row in grid]
        if pos == m * m:
            if ring is not None or stop_mean is None:
                return True
            mean = _grid_seam_mean(grid, ring, meta, t)
            if best_completion[0] is None or mean < best_completion[0]:
                best_completion[0] = mean
                best_completion[1] = [row[:] for row in grid]
                best_completion[2] = dict(gap_map)
            return False
        nodes[0] += 1
        if nodes[0] > max_nodes:
            budget_hit[0] = True
            raise _Budget
        r, c = divmod(pos, m)
        cands = []
        for name, img in pieces.items():
            if name in used:
                continue
            for rot in range(4):
                im = np.rot90(img, rot)
                if im.shape[:2] != shape0:
                    continue
                s = total_score(im, r, c, grid, ring, meta, t)
                if s <= tau:
                    cands.append((s, name, rot, im))
        cands.sort(key=lambda x: x[0])
        window = max_candidates
        if pos == 0 and ring is None:
            window = len(cands)
        for i, (s, name, rot, im) in enumerate(cands[:window]):
            gap_map[pos] = cands[i + 1][0] - s if i + 1 < len(cands) else float("inf")
            grid[r][c] = (name, rot, im, s)
            used.add(name)
            if steps is not None:
                steps.append({"type": "place", "pos": pos, "pid": name,
                              "rot": ((4 - rot) % 4) * 90, "score": s})
            if backtrack(pos + 1):
                return True
            grid[r][c] = None
            used.discard(name)
            if steps is not None:
                steps.append({"type": "unplace", "pos": pos, "pid": name,
                              "rot": ((4 - rot) % 4) * 90})
        return False

    try:
        ok = backtrack(0)
    except _Budget:
        ok = False
    if ring is not None or stop_mean is None:
        return (grid if ok else best["grid"]), ok, gap_map, None, budget_hit[0]
    if best_completion[0] is not None and best_completion[0] <= stop_mean:
        return best_completion[1], True, best_completion[2], best_completion[0], budget_hit[0]
    if best_completion[0] is not None:
        return best_completion[1], False, best_completion[2], best_completion[0], budget_hit[0]
    return (grid if ok else best["grid"]), False, gap_map, None, budget_hit[0]


def solve(pieces, ring, meta, t=T_STRIP, tau=TAU, max_candidates=MAX_CANDIDATES, steps=None):
    taus = [tau * 2 ** i for i in range(5)] + [float("inf")]
    best_partial = None
    best_completed = None
    selected = None
    ok = False
    last_level = 0
    for level, tv in enumerate(taus):
        last_level = level
        if steps:
            steps.clear()
        grid_a, ok_a, gap_map_a, mean_a, budget_a = _attempt(
            pieces, ring, meta, tv, max_candidates, t, MAX_NODES, steps,
            stop_mean=None if ring is not None else tau)
        if ok_a:
            selected = (grid_a, gap_map_a, level)
            ok = True
            print(f"Solusi ditemukan dengan tau={tv}")
            break
        if ring is not None:
            print(f"Buntu pada tau={tv}, melonggarkan ambang...")
            placed = sum(1 for row in grid_a for cell in row if cell is not None)
            if best_partial is None or placed > best_partial[0]:
                best_partial = (placed, grid_a, gap_map_a, level)
            continue
        if mean_a is not None:
            if best_completed is None or mean_a < best_completed[0]:
                best_completed = (mean_a, grid_a, gap_map_a, level)
            if not budget_a:
                break
        placed = sum(1 for row in grid_a for cell in row if cell is not None)
        if best_partial is None or placed > best_partial[0]:
            best_partial = (placed, grid_a, gap_map_a, level)
    if selected is None and best_completed is not None:
        _, grid_a, gap_map_a, level = best_completed
        selected = (grid_a, gap_map_a, level)
        ok = True
        print(f"Solusi tersusun dengan tau={taus[level]}")
    if selected is None and best_partial is not None:
        _, grid_a, gap_map_a, level = best_partial
        selected = (grid_a, gap_map_a, level)
    if selected is not None and steps is not None and selected[2] != last_level:
        steps.clear()
        _attempt(pieces, ring, meta, taus[selected[2]], max_candidates, t, MAX_NODES, steps,
                 stop_mean=None if ring is not None else tau)
    if selected is None:
        return None, False, [], []
    grid, gap_map, _ = selected
    m = meta["m"]
    gaps = [None] * (m * m)
    confs = [None] * (m * m)
    for pos in range(m * m):
        r, c = divmod(pos, m)
        cell = grid[r][c]
        if cell is not None and np.isfinite(gap_map.get(pos, float("inf"))):
            g = gap_map[pos]
            gaps[pos] = g
            confs[pos] = g / (cell[3] + g) if (cell[3] + g) > 0 else 0.0
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

    m = meta["m"]
    pieces = load_pieces(a.input_dir)
    if len(pieces) != m * m:
        raise SystemExit(f"jumlah potongan {len(pieces)} != m*m = {m * m}")

    ring = load_ring(a.input_dir) if meta["has_ring"] else None
    t = a.t
    if meta["has_ring"]:
        t = min(t, meta["hint_width"])
    else:
        print("PERINGATAN: tanpa ring, pencocokan hanya antar potongan (rentan ambigu).")

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
