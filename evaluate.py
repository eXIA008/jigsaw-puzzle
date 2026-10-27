import argparse
import json
import os


def load_ground_truth(path):
    with open(path) as f:
        return json.load(f)


def confidence_label(conf):
    if conf is None:
        return "-"
    if conf >= 0.5:
        return "tinggi"
    if conf >= 0.2:
        return "sedang"
    return "rendah"


def compare(solution, truth):
    truth_pieces = truth["pieces"]
    total = len(truth_pieces)
    pos_ok = 0
    rot_ok = 0
    rot_checked = any(t["rotation_cw"] % 360 != 0 for t in truth_pieces.values())
    wrong = []
    pos_status = {}
    for name, t in sorted(truth_pieces.items()):
        sol = solution.get(name)
        placed_ok = sol is not None and (sol[0], sol[1]) == (t["row"], t["col"])
        if sol is not None:
            pos_status[(sol[0], sol[1])] = placed_ok
        if sol is None:
            wrong.append(f"{name}: tidak ada di solution")
            continue
        row, col, rot = sol
        if (row, col) == (t["row"], t["col"]):
            pos_ok += 1
        else:
            wrong.append(f"{name}: posisi ({row},{col}) seharusnya ({t['row']},{t['col']})")
        expected = (360 - t["rotation_cw"]) % 360
        if rot == expected:
            rot_ok += 1
        else:
            wrong.append(f"{name}: rotasi {rot} seharusnya {expected}")
    return {"total": total, "pos_ok": pos_ok, "rot_ok": rot_ok, "rot_checked": rot_checked, "wrong": wrong,
            "m": truth.get("m"), "pos_status": pos_status}


def report_metrics(metrics, solve_meta=None):
    total = metrics["total"]
    acc_pos = 100 * metrics["pos_ok"] / total if total else 0.0
    print("=== Laporan Evaluasi Puzzle Solver ===")
    print(f"Potongan       : {total}")
    print(f"Akurasi posisi : {metrics['pos_ok']}/{total} ({acc_pos:.1f}%)")
    if metrics["rot_checked"]:
        acc_rot = 100 * metrics["rot_ok"] / total
        print(f"Akurasi rotasi : {metrics['rot_ok']}/{total} ({acc_rot:.1f}%)")
    else:
        print("Akurasi rotasi : - (puzzle tanpa rotasi)")
    if solve_meta is not None:
        gap = solve_meta.get("gap_mean")
        print(f"Gap rata-rata  : {'-' if gap is None else gap}")
        conf_mean = solve_meta.get("confidence_mean")
        print(f"Confidence     : {'-' if conf_mean is None else f'{conf_mean} ({confidence_label(conf_mean)})'}")
        print(f"Waktu solve    : {solve_meta.get('time_s', '-')} s")
        confs = solve_meta.get("confidences")
        m = metrics.get("m")
        if confs and m:
            print("Confidence per posisi:")
            for r in range(m):
                row = []
                for c in range(m):
                    x = confs[r * m + c]
                    mark = {True: "benar", False: "salah"}.get(metrics["pos_status"].get((r, c)), "-")
                    row.append(f"({r},{c}) {'-' if x is None else round(x, 2)} {confidence_label(x)}/{mark}")
                print("  " + " | ".join(row))
    for line in metrics["wrong"]:
        print(f"  {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="hasil", help="folder berisi ground_truth.json, solution.json, solve_meta.json")
    a = ap.parse_args()

    truth_path = os.path.join(a.input_dir, "ground_truth.json")
    sol_path = os.path.join(a.input_dir, "solution.json")
    meta_path = os.path.join(a.input_dir, "solve_meta.json")

    if not os.path.exists(truth_path):
        raise SystemExit(f"{truth_path} tidak ada. Jalankan puzzle_generator.py dengan --save_truth dulu.")
    if not os.path.exists(sol_path):
        raise SystemExit(f"{sol_path} tidak ada. Jalankan puzzle_solver.py --save dulu.")

    truth = load_ground_truth(truth_path)
    with open(sol_path) as f:
        solution = json.load(f)
    solve_meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            solve_meta = json.load(f)

    metrics = compare(solution, truth)
    report_metrics(metrics, solve_meta)


if __name__ == "__main__":
    main()
