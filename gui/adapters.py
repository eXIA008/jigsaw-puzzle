import json
import os
import random
import shutil
import time

import cv2
import numpy as np

import puzzle_generator as gen
import puzzle_solver as solver
import evaluate as evaluator


def _clean_out(out_dir):
    for name in ("frame.png", "hint_meta.json", "ground_truth.json",
                 "solution.json", "solve_meta.json", "reconstructed.png"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)
    for sub in ("pieces", "ring"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)


def hint_kmax(m, path=None, uploaded=None):
    img = None
    if uploaded is not None:
        img = cv2.imdecode(np.frombuffer(uploaded.getbuffer(), np.uint8), cv2.IMREAD_UNCHANGED)
    elif path and os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    return max(1, min(h // m, w // m) // 4)


def generate_puzzle(input_path, out_dir, m=3, hint_width=100, placement="split",
                    fit_mode="crop", rotate=False, seed=None, save_files=True,
                    save_truth=True):
    if os.path.isdir(out_dir):
        _clean_out(out_dir)
    rng = random.Random(seed)
    raw = gen.load_image(input_path)
    hint_width = gen.clamp_hint(raw, m, hint_width)
    ks = hint_width if placement == "split" else 0
    img = gen.fit_to_grid(raw, m, fit_mode, k=ks)
    h, w = img.shape[:2]
    frame = gen.make_frame(img, m, hint_width, placement=placement)
    ring = gen.extract_ring(img, ks)

    if save_files or save_truth:
        os.makedirs(os.path.join(out_dir, "pieces"), exist_ok=True)
    if save_files:
        cv2.imwrite(os.path.join(out_dir, "frame.png"), frame)
        if ring is not None:
            ring_dir = os.path.join(out_dir, "ring")
            os.makedirs(ring_dir, exist_ok=True)
            for side, arr in ring.items():
                cv2.imwrite(os.path.join(ring_dir, f"ring_{side}.png"), arr)

    pieces = gen.cut_pieces(img, m, ks)
    rng.shuffle(pieces)
    ids = rng.sample(range(10 ** 4, 10 ** 5), len(pieces))

    ph, pw = pieces[0][1].shape[:2]
    truth = {"image_size": [h, w], "m": m, "hint_width": hint_width, "placement": placement,
             "piece_size": [ph, pw], "grid_origin_xy": [ks, ks], "pieces": {}}
    shown = []
    for pid, ((r, c), p) in zip(ids, pieces):
        rot = rng.choice([0, 90, 180, 270]) if rotate else 0
        if rot:
            p = np.rot90(p, k=rot // 90, axes=(1, 0)).copy()
        name = f"piece_{pid}.png"
        if save_files:
            cv2.imwrite(os.path.join(out_dir, "pieces", name), p)
        shown.append(p)
        truth["pieces"][name] = {"row": r, "col": c, "rotation_cw": rot}

    if save_truth:
        with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
            json.dump(truth, f, indent=2)

    if save_files:
        hint_meta = {
            "m": m,
            "hint_width": hint_width,
            "placement": placement,
            "piece_size": [ph, pw],
            "has_ring": ring is not None,
        }
        with open(os.path.join(out_dir, "hint_meta.json"), "w") as f:
            json.dump(hint_meta, f, indent=2)

    return {
        "raw": raw,
        "image": img,
        "frame": frame,
        "pieces": shown,
        "m": m,
        "hint_width": hint_width,
        "piece_size": (ph, pw),
        "count": len(shown),
        "out_dir": out_dir,
    }


def solve_puzzle(input_dir, t=3, tau=2500.0, max_candidates=5, save=True, record_steps=True):
    meta = solver.load_hint_meta(os.path.join(input_dir, "hint_meta.json"))

    m = meta["m"]
    pieces = solver.load_pieces(input_dir)
    if len(pieces) != m * m:
        raise ValueError(f"jumlah potongan {len(pieces)} != m*m = {m * m}")

    ring = solver.load_ring(input_dir) if meta["has_ring"] else None
    ph, pw = meta["piece_size"]
    t = max(1, min(t, ph, pw))
    if meta["has_ring"]:
        t = min(t, meta["hint_width"])

    steps = [] if record_steps else None
    start = time.perf_counter()
    grid, ok, gaps, confs = solver.solve(
        pieces, ring, meta, t=t, tau=tau, max_candidates=max_candidates, steps=steps)
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
        "tau": tau,
        "max_candidates": max_candidates,
        "gaps": gaps,
        "gap_mean": round(sum(gap_vals) / len(gap_vals), 3) if gap_vals else None,
        "confidences": confs,
        "confidence_mean": round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else None,
    }

    rec = solver.reconstruct_image(grid, ring, meta)
    if save:
        with open(os.path.join(input_dir, "solution.json"), "w") as f:
            json.dump(solution, f, indent=2)
        with open(os.path.join(input_dir, "solve_meta.json"), "w") as f:
            json.dump(solve_meta, f, indent=2)
        if rec is not None:
            cv2.imwrite(os.path.join(input_dir, "reconstructed.png"), rec)

    return {
        "grid": grid,
        "ok": ok,
        "gaps": gaps,
        "confs": confs,
        "solution": solution,
        "solve_meta": solve_meta,
        "steps": steps if steps is not None else [],
        "reconstructed": rec,
        "meta": meta,
        "m": m,
        "pieces": pieces,
    }


def evaluate_solution(input_dir):
    truth_path = os.path.join(input_dir, "ground_truth.json")
    sol_path = os.path.join(input_dir, "solution.json")
    if not os.path.exists(truth_path):
        raise ValueError(f"{truth_path} tidak ada. Generate dengan Simpan ground_truth aktif dulu.")
    if not os.path.exists(sol_path):
        raise ValueError(f"{sol_path} tidak ada. Jalankan Solve dulu.")

    truth = evaluator.load_ground_truth(truth_path)
    with open(sol_path) as f:
        solution = json.load(f)
    solve_meta = None
    meta_path = os.path.join(input_dir, "solve_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            solve_meta = json.load(f)
    metrics = evaluator.compare(solution, truth)
    return {"metrics": metrics, "solve_meta": solve_meta}
