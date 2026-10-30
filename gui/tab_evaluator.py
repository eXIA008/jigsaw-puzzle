import os

import streamlit as st

import gui.render as rv
from gui import adapters


def _resolve_dir():
    src = st.session_state.get("ev_src", "Hasil sesi ini")
    if src == "Hasil sesi ini":
        d = st.session_state.get("last_gen_dir") or os.path.join(rv.session_workdir(), "hasil")
        if not os.path.exists(os.path.join(d, "ground_truth.json")):
            raise ValueError("Belum ada ground_truth.json pada sesi ini. Generate dengan Simpan ground_truth aktif.")
        return d
    if src == "Unggah file":
        dest = os.path.join(rv.session_workdir(), "eval_upload")
        sol = st.session_state.get("ev_sol")
        truth = st.session_state.get("ev_truth")
        if sol is None or truth is None:
            raise ValueError("Unggah solution.json dan ground_truth.json keduanya.")
        rv.save_upload(sol, dest, name="solution.json")
        rv.save_upload(truth, dest, name="ground_truth.json")
        return dest
    d = (st.session_state.get("ev_dir") or "").strip()
    if not d or not os.path.isdir(d):
        raise ValueError(f"Folder '{d}' tidak ada.")
    return d


def render():
    with st.sidebar:
        st.markdown("**Puzzle Evaluator**")
        st.caption("Mode untuk menilai kualitas hasil solve. " \
        "Pengguna memilih sumber data hasil solve dan ground truth (misalnya dari hasil sesi ini), lalu menekan tombol Evaluasi.")
        sources = ["Hasil sesi ini", "Unggah file"]
        st.selectbox("Sumber data", sources, key="ev_src")
        if st.session_state.get("ev_src") == "Unggah file":
            st.file_uploader("solution.json", type=["json"], key="ev_sol")
            st.file_uploader("ground_truth.json", type=["json"], key="ev_truth")
        elif st.session_state.get("ev_src") == "Folder lokal":
            st.text_input("Folder hasil", value="hasil", key="ev_dir")
        run = st.button("Evaluasi", type="primary", width="stretch", key="ev_run")

    if run:
        try:
            result = adapters.evaluate_solution(_resolve_dir())
            st.session_state.eval_result = result
            st.success("Evaluasi selesai.")
        except (ValueError, FileNotFoundError, OSError) as exc:
            st.error(str(exc))

    result = st.session_state.get("eval_result")
    if result is None:
        st.info("Belum ada hasil. Pilih sumber data lalu tekan Evaluasi.")
        return

    metrics = result["metrics"]
    total = metrics["total"] or 0
    acc_pos = 100 * metrics["pos_ok"] / total if total else 0.0
    acc_rot = 100 * metrics["rot_ok"] / total if metrics["rot_checked"] else None
    solve_meta = result["solve_meta"] or {}
    gap = solve_meta.get("gap_mean")
    conf = solve_meta.get("confidence_mean")

    st.markdown("## Laporan evaluasi")
    st.markdown(f"**Akurasi posisi:** `{acc_pos:.1f}%`")
    st.markdown(f"**Akurasi rotasi:** `{'n/a' if acc_rot is None else f'{acc_rot:.1f}%'}`")
    st.markdown(f"**Potongan:** `{total}`")
    st.markdown(f"**Waktu solve:** `{solve_meta.get('time_s', '-')} s`")
    st.markdown(f"**Gap mean:** `{'-' if gap is None else gap}`")
    st.markdown(f"**Confidence mean:** `{'-' if conf is None else conf}`")

    with st.expander("Detail per potongan"):
        if metrics["wrong"]:
            for line in metrics["wrong"]:
                st.write(line)
        else:
            st.write("Semua potongan benar posisi dan rotasinya.")
