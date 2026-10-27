"""Puzzle Solver - Tugas Besar 1 Tahap 2 (PCD)

Input : folder hasil generator (pieces/, ring/, hint_meta.json)
Output: solution.json  {nama_piece: [row, col, rotasi_ccw]}
        rekonstruksi.png

Solver HANYA membaca pieces/, ring/, hint_meta.json. Tidak membaca ground_truth.json.
Rotasi dilaporkan sebagai sudut searah jarum jam yang harus dikenakan pada potongan
untuk mengembalikannya, yaitu (360 - rotation_cw) % 360 (internal: np.rot90 k kali, CCW).

Contoh:
    python puzzle_solver.py --input_dir hasil --show
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

# ====== PENGATURAN ======
INPUT_DIR = "hasil"
T = 3                      # tebal strip apakah # ambang MSE per sambungan; dilonggarkan otomatis jika solver buntu
MAX_CANDIDATES = 5         # kandidat terbaik yang dicoba per posisi
MAX_NODES = 200000         # batas langkah backtracking per percobaan
GRAD_WEIGHT = 0.5          # bobot kontinuitas gradien (aktif jika t >= 2)
SAVE = True                # True = simpan solution.json, rekonstruksi.png, figure_solver.png
SHOW = True                # True = tampilkan figure di jendela
EXPLAIN = True             # True = cetak penjelasan tiap penempatan + angka sambungan di figure
# ========================


# ---------- loader ----------
def _read(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = img[:, :, None]
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def load_hint_meta(path):
    with open(path) as f:
        return json.load(f)


def load_ring(d):
    ring = {}
    for side in ("top", "bottom", "left", "right"):
        p = os.path.join(d, f"ring_{side}.png")
        if os.path.exists(p):
            ring[side] = _read(p)
    return ring or None


def load_pieces(d):
    files = sorted(glob.glob(os.path.join(d, "piece_*.png")))
    if not files:
        raise FileNotFoundError(f"tidak ada piece_*.png di {d}")
    return {os.path.basename(f): _read(f) for f in files}


# ---------- geometri ring (dokumentasi bagian 2) ----------
def ring_slice(ring, side, idx, meta, t):
    """Strip ring sejajar potongan ke-idx di sisi `side`, setebal t, dengan urutan
    baris/kolom dari luar ke dalam (top/left) atau dalam ke luar (bottom/right)."""
    k = meta["hint_width"]
    ph, pw = meta["piece_size"]
    if side == "top":
        return ring["top"][-t:, k + idx * pw:k + (idx + 1) * pw]
    if side == "bottom":
        return ring["bottom"][:t, k + idx * pw:k + (idx + 1) * pw]
    if side == "left":
        return ring["left"][k + idx * ph:k + (idx + 1) * ph, -t:]
    return ring["right"][k + idx * ph:k + (idx + 1) * ph, :t]


# ---------- skor ----------
def extract_edge(img, side, t=3):
    """Strip tepi t baris/kolom, warna asli."""
    if side == "top":    return img[:t, :, :]
    if side == "bottom": return img[-t:, :, :]
    if side == "left":   return img[:, :t, :]
    if side == "right":  return img[:, -t:, :]


def _as_rows(strip, side):
    """Ubah strip ke bentuk (t, panjang, C) dengan baris terakhir (bottom/right) atau
    pertama (top/left) berbatasan dengan sambungan. Left/right ditranspos."""
    s = strip.astype(np.float32)
    return np.swapaxes(s, 0, 1) if side in ("left", "right") else s


def seam_cost(before, after, side_before, detail=False):
    """Biaya sambungan antara strip `before` (sisi bottom/right sebuah bagian) dan
    strip `after` (sisi top/left bagian berikutnya). Baris terdekat sambungan
    dibandingkan langsung, ditambah kontinuitas gradien jika t >= 2.
    """
    a = _as_rows(before, side_before)
    b = _as_rows(after, "top" if side_before == "bottom" else "left")
    d = a[-1] - b[0]
    cost = float(np.mean(d * d))
    if a.shape[0] >= 2 and b.shape[0] >= 2:
        pa = 2 * a[-1] - a[-2]               # ekstrapolasi a ke seberang sambungan
        pb = 2 * b[0] - b[1]                 # ekstrapolasi b ke seberang sambungan
        g = (pa - b[0], pb - a[-1])
        cost += GRAD_WEIGHT * 0.5 * float(np.mean(g[0] ** 2) + np.mean(g[1] ** 2))
    if detail:
        return cost, float(np.sqrt(np.mean(d * d)))   # RMSE selisih warna tepat di sambungan
    return cost


def seam_parts(cand, r, c, grid, ring, meta, t, detail=False):
    """Daftar (label, biaya[, rmse]) tiap sambungan yang diketahui untuk `cand` di (r, c)."""
    m = meta["m"]
    parts = []
    tk = min(t, meta["hint_width"]) if ring else t
    # atas
    if r > 0:
        ref = extract_edge(grid[r - 1][c][2], "bottom", t)
        parts.append(("atas <- potongan " + grid[r - 1][c][0][6:11],
                      seam_cost(ref, extract_edge(cand, "top", t), "bottom", detail)))
    elif ring:
        parts.append(("atas <- ring",
                      seam_cost(ring_slice(ring, "top", c, meta, tk), extract_edge(cand, "top", tk), "bottom", detail)))
    # kiri
    if c > 0:
        ref = extract_edge(grid[r][c - 1][2], "right", t)
        parts.append(("kiri <- potongan " + grid[r][c - 1][0][6:11],
                      seam_cost(ref, extract_edge(cand, "left", t), "right", detail)))
    elif ring:
        parts.append(("kiri <- ring",
                      seam_cost(ring_slice(ring, "left", r, meta, tk), extract_edge(cand, "left", tk), "right", detail)))
    # bawah / kanan hanya dari ring (di baris/kolom terakhir)
    if r == m - 1 and ring:
        parts.append(("bawah <- ring",
                      seam_cost(extract_edge(cand, "bottom", tk), ring_slice(ring, "bottom", c, meta, tk), "bottom", detail)))
    if c == m - 1 and ring:
        parts.append(("kanan <- ring",
                      seam_cost(extract_edge(cand, "right", tk), ring_slice(ring, "right", r, meta, tk), "right", detail)))
    return parts


def total_score(cand, r, c, grid, ring, meta, t):
    """Rata-rata biaya semua sambungan yang diketahui untuk `cand` di (r, c)."""
    costs = [p[1] for p in seam_parts(cand, r, c, grid, ring, meta, t)]
    return float(np.mean(costs)) if costs else 0.0


def explain(grid, ring, meta, t):
    """Penjelasan per posisi: error tiap sambungan (RMSE tingkat keabuan 0-255) dan
    keyakinan terhadap kandidat terbaik kedua. Dihitung ulang dari grid akhir."""
    out, tk = [], t
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            seams = [{"sambungan": lab, "rmse": round(rm, 2)}
                     for lab, (_, rm) in seam_parts(cell[2], r, c, grid, ring, meta, tk, detail=True)]
            worst = max((s["rmse"] for s in seams), default=0.0)
            verdict = "yakin" if worst < 12 and cell[4] > 50 else "cukup" if worst < 25 else "ragu"
            gap = None if not np.isfinite(cell[4]) else round(cell[4], 1)
            out.append({"posisi": [r, c], "potongan": cell[0], "rotasi_cw": (360 - 90 * cell[1]) % 360,
                        "skor": round(cell[3], 1), "gap_ke_kandidat_2": gap,
                        "sambungan": seams, "kesimpulan": verdict})
    return out


def print_explain(items):
    print("\nPenjelasan penempatan (RMSE = rata-rata selisih warna di sambungan, skala 0-255; kecil = halus)")
    for it in items:
        r, c = it["posisi"]
        s = "; ".join(f"{x['sambungan']}: {x['rmse']}" for x in it["sambungan"])
        gap = "tanpa pesaing lolos tau" if it["gap_ke_kandidat_2"] is None else it["gap_ke_kandidat_2"]
        print(f"({r},{c}) {it['potongan'][6:11]} rot {it['rotasi_cw']:>3}: {s} | gap {gap} -> {it['kesimpulan']}")


# ---------- solver ----------
class _Budget(Exception):
    pass


def _attempt(pieces, ring, meta, tau, max_candidates, t, allow_rot, max_nodes):
    m = meta["m"]
    shape0 = tuple(meta["piece_size"])
    grid = [[None] * m for _ in range(m)]       # sel: (nama, rot, array, skor, gap)
    used = set()
    best = {"depth": -1, "grid": None}
    nodes = [0]
    rots = range(4) if allow_rot else range(1)

    def backtrack(pos):
        if pos > best["depth"]:
            best["depth"] = pos
            best["grid"] = [row[:] for row in grid]
        if pos == m * m:
            return True
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise _Budget
        r, c = divmod(pos, m)
        cands = []
        for name, img in pieces.items():
            if name in used:
                continue
            for rot in rots:
                im = np.rot90(img, rot)
                if im.shape[:2] != shape0:      # potongan tidak persegi: rotasi 90/270 tak valid
                    continue
                s = total_score(im, r, c, grid, ring, meta, t)
                if s <= tau:
                    cands.append((s, name, rot, im))
        cands.sort(key=lambda x: x[0])
        for i, (s, name, rot, im) in enumerate(cands[:max_candidates]):
            gap = (cands[i + 1][0] - s) if i + 1 < len(cands) else float("inf")
            grid[r][c] = (name, rot, im, s, gap)
            used.add(name)
            if backtrack(pos + 1):
                return True
            grid[r][c] = None
            used.discard(name)
        return False

    try:
        ok = backtrack(0)
    except _Budget:
        ok = False
    return (grid if ok else best["grid"]), ok


def solve(pieces, ring, meta, tau=TAU, max_candidates=MAX_CANDIDATES, t=T, allow_rot=True):
    """Coba tau awal; jika buntu/anggaran habis, longgarkan tau (x2) beberapa kali,
    terakhir tanpa ambang."""
    taus = [tau * 2 ** i for i in range(5)] + [float("inf")]
    grid, ok = None, False
    for tv in taus:
        grid, ok = _attempt(pieces, ring, meta, tv, max_candidates, t, allow_rot, MAX_NODES)
        if ok:
            print(f"Solusi ditemukan dengan tau={tv}")
            break
        print(f"Buntu pada tau={tv}, melonggarkan ambang...")
    return grid, ok


# ---------- rekonstruksi ----------
def reconstruct_image(grid, ring, meta):
    m = meta["m"]
    k = meta["hint_width"] if ring else 0
    ph, pw = meta["piece_size"]
    sample = next(cell[2] for row in grid for cell in row if cell)
    ch = sample.shape[2]
    H, W = 2 * k + m * ph, 2 * k + m * pw
    canvas = np.full((H, W, ch), 255, np.uint8)
    if ring:
        canvas[:k, :] = ring["top"]
        canvas[-k:, :] = ring["bottom"]
        canvas[:, :k] = ring["left"]
        canvas[:, -k:] = ring["right"]
    for r in range(m):
        for c in range(m):
            if grid[r][c]:
                canvas[k + r * ph:k + (r + 1) * ph, k + c * pw:k + (c + 1) * pw] = grid[r][c][2]
    return canvas[:, :, 0] if ch == 1 else canvas


def verify_from_pieces(grid, pieces, ring, meta, canvas):
    """Bukti bahwa rekonstruksi murni dari potongan + ring: tiap sel kanvas harus
    identik byte-per-byte dengan np.rot90(potongan_asli, rot), dan area di luar sel
    harus identik dengan ring. Solver tidak pernah membuka gambar asli (foto.jpg)."""
    m, k = meta["m"], (meta["hint_width"] if ring else 0)
    ph, pw = meta["piece_size"]
    cv = canvas if canvas.ndim == 3 else canvas[:, :, None]
    for r in range(m):
        for c in range(m):
            cell = grid[r][c]
            if cell is None:
                continue
            name, rot = cell[0], cell[1]
            tile = cv[k + r * ph:k + (r + 1) * ph, k + c * pw:k + (c + 1) * pw]
            if not np.array_equal(tile, np.rot90(pieces[name], rot)):
                return False
    if ring:
        ok = (np.array_equal(cv[:k, :], ring["top"]) and np.array_equal(cv[-k:, :], ring["bottom"])
              and np.array_equal(cv[:, :k], ring["left"]) and np.array_equal(cv[:, -k:], ring["right"]))
        return ok
    return True


def draw_seams(ax, items, meta, ring):
    """Tulis nilai RMSE tiap sambungan tepat di garis sambungannya (hijau kecil/halus,
    oranye sedang, merah besar) dan ID + rotasi potongan di tengah selnya."""
    m, k = meta["m"], (meta["hint_width"] if ring else 0)
    ph, pw = meta["piece_size"]
    for r in range(m + 1):
        ax.axhline(k + r * ph - 0.5, color="red", lw=0.8)
    for c in range(m + 1):
        ax.axvline(k + c * pw - 0.5, color="red", lw=0.8)
    for it in items:
        r, c = it["posisi"]
        x, y = k + c * pw, k + r * ph
        ax.text(x + pw / 2, y + ph / 2, f"{it['potongan'][6:11]}\nrot {it['rotasi_cw']}", ha="center", va="center",
                fontsize=7, color="white", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
        pos = {"atas": (x + pw / 2, y + 9), "kiri": (x + 14, y + ph / 2 - 22),
               "bawah": (x + pw / 2, y + ph - 9), "kanan": (x + pw - 14, y + ph / 2 - 22)}
        for s in it["sambungan"]:
            side = s["sambungan"].split()[0]
            col = "#1a9850" if s["rmse"] < 8 else "#e08a00" if s["rmse"] < 20 else "#d73027"
            ax.text(*pos[side], f"{s['rmse']:g}", ha="center", va="center", fontsize=7, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=col, lw=0))


def make_figure(pieces, canvas, grid, meta, ring, items=None):
    """Figure: potongan acak (masukan) -> rekonstruksi + nilai tiap sambungan -> hasil bersih."""
    import matplotlib.pyplot as plt

    def rgb(x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if x.ndim == 3 and x.shape[2] == 3 else x[:, :, 0] if x.ndim == 3 else x
    gray = canvas.ndim == 2 or canvas.shape[2] == 1
    kw = {"cmap": "gray", "vmin": 0, "vmax": 255} if gray else {}
    m = meta["m"]
    fig = plt.figure(figsize=(19, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.6, 1.0])
    sub = gs[0].subgridspec(m, m, wspace=0.08, hspace=0.08)
    for i, name in enumerate(sorted(pieces)):
        ax = fig.add_subplot(sub[i // m, i % m])
        ax.imshow(rgb(pieces[name]), **kw)
        ax.axis("off")
    fig.text(0.13, 0.93, f"Masukan: {len(pieces)} potongan acak", ha="center")
    ax = fig.add_subplot(gs[1]); ax.imshow(rgb(canvas), **kw)
    if items:
        draw_seams(ax, items, meta, ring)
    ax.set_title("Angka = RMSE sambungan (0-255, kecil = halus/cocok)"); ax.axis("off")
    ax = fig.add_subplot(gs[2]); ax.imshow(rgb(canvas), **kw)
    ax.set_title("Hasil rekonstruksi"); ax.axis("off")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=INPUT_DIR)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--t", type=int, default=T)
    ap.add_argument("--max_candidates", type=int, default=MAX_CANDIDATES)
    ap.add_argument("--no_rotate", action="store_true", help="hanya coba orientasi 0 derajat")
    ap.add_argument("--explain", action="store_true", default=EXPLAIN, help="cetak alasan tiap penempatan (error sambungan), simpan explain.json, dan tampilkan angka di figure")
    ap.add_argument("--no_explain", dest="explain", action="store_false")
    ap.add_argument("--show", action="store_true", default=SHOW, help="tampilkan figure")
    ap.add_argument("--no_show", dest="show", action="store_false")
    ap.add_argument("--save", action="store_true", default=SAVE, help="simpan solution.json, rekonstruksi.png, figure_solver.png")
    ap.add_argument("--no_save", dest="save", action="store_false")
    a = ap.parse_args()

    meta = load_hint_meta(os.path.join(a.input_dir, "hint_meta.json"))
    ring = load_ring(os.path.join(a.input_dir, "ring")) if meta.get("has_ring") else None
    pieces = load_pieces(os.path.join(a.input_dir, "pieces"))
    m = meta["m"]
    if len(pieces) != m * m:
        raise ValueError(f"jumlah potongan {len(pieces)} != M*M = {m * m}")
    if ring is None:
        print("PERINGATAN: tanpa ring, pencocokan hanya antar potongan (sangat ambigu).")

    t0 = time.time()
    grid, ok = solve(pieces, ring, meta, a.tau, a.max_candidates, a.t, not a.no_rotate)
    dt = time.time() - t0
    placed = sum(1 for row in grid for cell in row if cell)
    print(f"{'SELESAI' if ok else 'GAGAL (parsial)'}: {placed}/{m * m} potongan, {dt:.2f} detik")

    solution = {cell[0]: [r, c, (360 - 90 * cell[1]) % 360]
                for r, row in enumerate(grid) for c, cell in enumerate(row) if cell}
    gaps = [cell[4] for row in grid for cell in row if cell and np.isfinite(cell[4])]
    if gaps:
        print(f"Confidence gap rata-rata: {np.mean(gaps):.2f}, minimum: {np.min(gaps):.2f}")

    items = None
    if a.explain:
        items = explain(grid, ring, meta, a.t)
        print_explain(items)
        if a.save:
            with open(os.path.join(a.input_dir, "explain.json"), "w") as f:
                json.dump(items, f, indent=2)

    canvas = reconstruct_image(grid, ring, meta)
    print("Verifikasi: " + ("semua sel kanvas identik dengan potongan (setelah rotasi) + ring, "
                            "tidak ada piksel dari sumber lain"
                            if verify_from_pieces(grid, pieces, ring, meta, canvas) else "GAGAL, ada sel yang bukan dari potongan"))
    if a.save:
        with open(os.path.join(a.input_dir, "solution.json"), "w") as f:
            json.dump(solution, f, indent=2)
        cv2.imwrite(os.path.join(a.input_dir, "rekonstruksi.png"), canvas)
        print(f"Disimpan: {a.input_dir}/solution.json, rekonstruksi.png, figure_solver.png")
    if a.save or a.show:
        import matplotlib
        if not a.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = make_figure(pieces, canvas, grid, meta, ring, items)
        if a.save:
            fig.savefig(os.path.join(a.input_dir, "figure_solver.png"), dpi=130, bbox_inches="tight")
        if a.show:
            plt.show()


if __name__ == "__main__":
    main()
