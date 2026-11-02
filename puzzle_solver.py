import json
import os
import cv2
import numpy as np


def load_hint_meta(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File metadata tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pieces(input_dir):
    pieces_dir = os.path.join(input_dir, "pieces")
    if not os.path.isdir(pieces_dir):
        raise FileNotFoundError(f"Folder keping tidak ditemukan: {pieces_dir}")
    pieces = {}
    for fname in sorted(os.listdir(pieces_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            fpath = os.path.join(pieces_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Standardisasi ruang warna: BGRA -> BGR (Operasi Titik)
                if img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                pieces[fname] = img
    return pieces


def load_ring(input_dir):
    ring_dir = os.path.join(input_dir, "ring")
    if not os.path.isdir(ring_dir):
        return None
    ring = {}
    for side in ("top", "bottom", "left", "right"):
        p = os.path.join(ring_dir, f"ring_{side}.png")
        if not os.path.isfile(p):
            return None
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        ring[side] = img
    return ring


def _calc_edge_dissimilarity(strip1, strip2, t=3):
    """
    Operasi Aritmatika Pengurangan Citra g(x,y) = |f(x,y) - h(x,y)|
    dikombinasikan dengan Konvolusi Filter Gradien Spasial Dinamis
    """
    s1 = strip1.astype(np.float32)
    s2 = strip2.astype(np.float32)

    # 1. Operasi Aritmatika Pengurangan Citra pada garis pertemuan terluar
    diff_direct = np.abs(s1[-1] - s2[0])
    l1_cost = float(np.mean(diff_direct))

    # 2. Filter Diferensial Spasial
    actual_depth = min(s1.shape[0], s2.shape[0])
    if actual_depth >= 2:
        grad1 = np.diff(s1, axis=0) 
        grad2 = np.diff(s2, axis=0) 

        # Menghitung rata-rata arah gradien di sepanjang kedalaman pita t
        mean_grad1 = np.mean(grad1, axis=0)
        mean_grad2 = np.mean(grad2, axis=0)

        # Evaluasi deviasi kesinambungan gradien
        grad_continuity_error = np.abs(mean_grad1 - mean_grad2)
        grad_cost = float(np.mean(grad_continuity_error))

        return l1_cost + 0.5 * grad_cost

    return l1_cost


def _score_placement(cand_img, r, c, m, grid, ring, ph, pw, k, t):
    """
    Evaluasi kompatibilitas spasial kandidat keping terhadap batas tetangga
    dan ring pembatas bingkai dengan koordinat pergeseran spasial yang presisi.
    """
    costs = []

    # 1. Batas Atas
    if r > 0 and grid[r - 1][c] is not None:
        neighbor = grid[r - 1][c][2]
        cost = _calc_edge_dissimilarity(neighbor[-t:, :], cand_img[:t, :], t=t)
        costs.append(cost)
    elif r == 0 and ring is not None:
        ring_strip = ring["top"][-t:, k + c * pw : k + (c + 1) * pw]
        cost = _calc_edge_dissimilarity(ring_strip, cand_img[:t, :], t=t)
        costs.append(cost)

    # 2. Batas Bawah
    if r < m - 1 and grid[r + 1][c] is not None:
        neighbor = grid[r + 1][c][2]
        cost = _calc_edge_dissimilarity(cand_img[-t:, :], neighbor[:t, :], t=t)
        costs.append(cost)
    elif r == m - 1 and ring is not None:
        ring_strip = ring["bottom"][:t, k + c * pw : k + (c + 1) * pw]
        cost = _calc_edge_dissimilarity(cand_img[-t:, :], ring_strip, t=t)
        costs.append(cost)

    # 3. Batas Kiri
    if c > 0 and grid[r][c - 1] is not None:
        neighbor = grid[r][c - 1][2]
        s1 = np.transpose(neighbor[:, -t:], (1, 0, 2))
        s2 = np.transpose(cand_img[:, :t], (1, 0, 2))
        cost = _calc_edge_dissimilarity(s1, s2, t=t)
        costs.append(cost)
    elif c == 0 and ring is not None:
        ring_strip = np.transpose(ring["left"][k + r * ph : k + (r + 1) * ph, -t:], (1, 0, 2))
        cand_strip = np.transpose(cand_img[:, :t], (1, 0, 2))
        cost = _calc_edge_dissimilarity(ring_strip, cand_strip, t=t)
        costs.append(cost)

    # 4. Batas Kanan
    if c < m - 1 and grid[r][c + 1] is not None:
        neighbor = grid[r][c + 1][2]
        s1 = np.transpose(cand_img[:, -t:], (1, 0, 2))
        s2 = np.transpose(neighbor[:, :t], (1, 0, 2))
        cost = _calc_edge_dissimilarity(s1, s2, t=t)
        costs.append(cost)
    elif c == m - 1 and ring is not None:
        ring_strip = np.transpose(ring["right"][k + r * ph : k + (r + 1) * ph, :t], (1, 0, 2))
        cand_strip = np.transpose(cand_img[:, -t:], (1, 0, 2))
        cost = _calc_edge_dissimilarity(cand_strip, ring_strip, t=t)
        costs.append(cost)

    if not costs:
        return 0.0, 0

    # Kombinasi mean penalty dan max penalty untuk mendeteksi inkonsistensi sisi
    score = 0.6 * float(np.mean(costs)) + 0.4 * float(np.max(costs))
    return score, len(costs)


def solve(pieces, ring, meta, t=3, max_candidates=5, steps=None):
    """
    Algoritma Penataan Heuristik Berbasis Keyakinan (Confidence-Guided Search)
    dengan Pencatatan Riwayat Eksekusi (Backtracking) untuk GUI.
    """
    m = meta["m"]
    ph, pw = meta["piece_size"]
    has_ring = meta.get("has_ring", False)
    k = int(meta.get("hint_width", 0)) if has_ring else 0
    t = max(1, min(t, ph, pw))
    if has_ring and k > 0:
        t = min(t, k)

    # Pra-komputasi Transformasi Geometri Rotasi Ortogonal (0, 90, 180, 270 derajat CW)
    cached_pieces = {}
    for pid, img in pieces.items():
        rot_list = []
        for rot_k in (0, 1, 2, 3):
            rot_img = np.rot90(img, k=rot_k, axes=(1, 0)).copy()
            if rot_img.shape[0] == ph and rot_img.shape[1] == pw:
                rot_list.append((rot_k, rot_img))
        cached_pieces[pid] = rot_list

    grid = [[None for _ in range(m)] for _ in range(m)]
    used_pids = set()
    gaps = []
    confs = []

    def get_candidate_rankings_for_slot(r, c):
        candidates = []
        for pid, rots in cached_pieces.items():
            if pid in used_pids:
                continue
            for rot_k, rot_img in rots:
                score, num_c = _score_placement(rot_img, r, c, m, grid, ring, ph, pw, k, t)
                if num_c > 0:
                    candidates.append((score, pid, rot_k, rot_img))
        candidates.sort(key=lambda x: x[0])
        return candidates

    def select_best_slot_by_confidence():
        best_slot = None
        best_ranking = None
        max_priority = -float("inf")

        for r in range(m):
            for c in range(m):
                if grid[r][c] is not None:
                    continue

                num_c = 0
                if r > 0 and grid[r - 1][c] is not None: num_c += 1
                elif r == 0 and ring is not None: num_c += 1

                if r < m - 1 and grid[r + 1][c] is not None: num_c += 1
                elif r == m - 1 and ring is not None: num_c += 1

                if c > 0 and grid[r][c - 1] is not None: num_c += 1
                elif c == 0 and ring is not None: num_c += 1

                if c < m - 1 and grid[r][c + 1] is not None: num_c += 1
                elif c == m - 1 and ring is not None: num_c += 1

                if num_c == 0:
                    continue

                ranking = get_candidate_rankings_for_slot(r, c)
                if not ranking:
                    return None, None

                s1 = ranking[0][0]
                s2 = ranking[1][0] if len(ranking) > 1 else (s1 + 50.0)
                gap = s2 - s1

                # Prioritaskan slot dengan batasan aktif banyak dan kepastian perbedaan skor tinggi
                priority = (num_c * 100.0) + gap - (s1 * 0.1)
                if priority > max_priority:
                    max_priority = priority
                    best_slot = (r, c)
                    best_ranking = ranking

        return best_slot, best_ranking

    def backtrack():
        if len(used_pids) == m * m:
            return True

        slot, ranking = select_best_slot_by_confidence()
        if slot is None:
            for r in range(m):
                for c in range(m):
                    if grid[r][c] is None:
                        slot = (r, c)
                        break
                if slot: break
            ranking = []
            for pid, rots in cached_pieces.items():
                if pid not in used_pids:
                    ranking.append((0.0, pid, rots[0][0], rots[0][1]))

        r, c = slot
        pos_idx = r * m + c
        branch_candidates = ranking[:max_candidates]
        if not branch_candidates:
            return False

        for idx, (score, pid, rot_k, rot_img) in enumerate(branch_candidates):
            s1 = score
            s2 = branch_candidates[1][0] if len(branch_candidates) > 1 else (s1 + 10.0)
            cur_gap = s2 - s1
            cur_conf = 1.0 - (s1 / (s2 + 1e-4))

            grid[r][c] = (pid, rot_k, rot_img, score)
            used_pids.add(pid)
            gaps.append(round(float(cur_gap), 2))
            confs.append(round(float(cur_conf), 3))

            if steps is not None:
                steps.append({
                    "type": "place",
                    "pos": pos_idx,
                    "pid": pid,
                    "rot": rot_k * 90,
                    "score": float(score),
                })

            if backtrack():
                return True

            # Mundur (Backtrack):
            grid[r][c] = None
            used_pids.remove(pid)
            if gaps: gaps.pop()
            if confs: confs.pop()

            if steps is not None:
                steps.append({
                    "type": "remove",
                    "pos": pos_idx,
                    "pid": pid,
                    "rot": rot_k * 90,
                    "score": float(score),
                })

        return False

    success = backtrack()
    return grid, success, gaps, confs


def reconstruct_image(grid, ring, meta):
    """
    Operasi Geometri Translasi Spasial Citra: Menyusun mosaik keping
    ke koordinat piksel global pada kanvas citra utuh.
    """
    m = meta["m"]
    ph, pw = meta["piece_size"]
    has_ring = meta.get("has_ring", False)
    k = int(meta.get("hint_width", 0)) if has_ring else 0

    if has_ring and ring is not None and k > 0:
        total_h = m * ph + 2 * k
        total_w = m * pw + 2 * k
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        canvas[:k, :] = ring["top"]
        canvas[-k:, :] = ring["bottom"]
        canvas[:, :k] = ring["left"]
        canvas[:, -k:] = ring["right"]

        for r in range(m):
            for c in range(m):
                cell = grid[r][c]
                if cell is not None:
                    y1 = k + r * ph
                    x1 = k + c * pw
                    canvas[y1 : y1 + ph, x1 : x1 + pw] = cell[2]
        return canvas
    else:
        canvas = np.zeros((m * ph, m * pw, 3), dtype=np.uint8)
        for r in range(m):
            for c in range(m):
                cell = grid[r][c]
                if cell is not None:
                    canvas[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = cell[2]
        return canvas