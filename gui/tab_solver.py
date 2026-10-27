import os
import time

import streamlit as st

import gui.render as rv
from gui import adapters


def _resolve_dir():
    src = st.session_state.get("sol_src", "Hasil sesi ini")
    if src == "Hasil sesi ini":
        d = st.session_state.get("last_gen_dir") or os.path.join(rv.session_workdir(), "hasil")
        if not os.path.exists(os.path.join(d, "hint_meta.json")):
            raise ValueError("Belum ada hasil generate pada sesi ini. Generate dulu atau ubah sumber data.")
        return d
    if src == "Unggah ZIP":
        uploaded = st.session_state.get("sol_zip")
        if uploaded is None:
            raise ValueError("Belum ada file ZIP yang diunggah.")
        return rv.extract_zip(uploaded, os.path.join(rv.session_workdir(), "unpacked"))
    d = (st.session_state.get("sol_dir") or "").strip()
    if not d or not os.path.isdir(d):
        raise ValueError(f"Folder '{d}' tidak ada.")
    return d


def render():
    with st.sidebar:
        st.markdown("**Puzzle Solver**")
        st.caption("Mode untuk menyusun kembali puzzle secara otomatis.")
        sources = ["Hasil sesi ini", "Unggah ZIP"]
        if not rv.is_cloud():
            sources.append("Folder lokal")
        st.selectbox("Sumber data", sources, key="sol_src")
        if st.session_state.get("sol_src") == "Unggah ZIP":
            st.file_uploader("ZIP hasil generate", type=["zip"], key="sol_zip")
        elif st.session_state.get("sol_src") == "Folder lokal":
            st.text_input("Folder hasil", value="hasil", key="sol_dir")

        st.markdown("**Parameter solve**")
        st.slider(
            "t (tebal strip)", 1, 16, 3, key="sol_t",
            help="Tebal band tepi yang dicocokkan, dalam piksel. Solver membandingkan band ini antar potongan "
                 "bersebelahan dan terhadap ring; makin tebal, makin banyak baris yang ikut dalam rata-rata MSE sambungan.",
        )
        st.slider(
            "tau (ambang MSE)", 0.0, 10000.0, 2500.0, step=100.0, key="sol_tau",
            help="Skor sambungan di atas nilai ini tidak dijadikan kandidat. Bila solver buntu, ambang dilonggarkan "
                 "otomatis sampai lima kali sebelum akhirnya diabaikan.",
        )
        st.slider(
            "Max kandidat", 1, 20, 5, key="sol_maxc",
            help="Banyak kandidat terbaik yang dicoba per sel saat backtracking. Nilai besar aman untuk potongan yang "
                 "mirip, tetapi pencarian makin lambat.",
        )
        run = st.button("Solve", type="primary", width="stretch", key="sol_run")

    if run:
        try:
            input_dir = _resolve_dir()
            with st.spinner("Menyusun potongan..."):
                result = adapters.solve_puzzle(
                    input_dir,
                    t=int(st.session_state["sol_t"]),
                    tau=float(st.session_state["sol_tau"]),
                    max_candidates=int(st.session_state["sol_maxc"]),
                )
            result["input_dir"] = input_dir
            st.session_state.solve_result = result
            st.session_state["sol_step"] = len(result["steps"])
            meta = result["solve_meta"]
            filled = len(result["solution"])
            total = result["m"] * result["m"]
            status = "berhasil" if result["ok"] else "gagal"
            st.success(f"Solve {status}: {filled}/{total} potongan terisi dalam {meta['time_s']} s.")
        except (ValueError, FileNotFoundError, OSError) as exc:
            st.error(str(exc))

    result = st.session_state.get("solve_result")
    if result is None:
        st.info("Belum ada hasil. Pilih sumber data, atur parameter, lalu tekan Solve.")
        return

    meta = result["solve_meta"]
    m = result["m"]
    total = m * m
    gap = meta.get("gap_mean")
    conf = meta.get("confidence_mean")

    st.markdown("## Hasil solve")
    st.markdown(f"**Status:** `{'ok' if result['ok'] else 'gagal'}`")
    st.markdown(f"**Terisi:** `{len(result['solution'])}/{total}`")
    st.markdown(f"**Waktu solve:** `{meta['time_s']} s`")
    st.markdown(f"**Gap mean:** `{'-' if gap is None else gap}`")
    st.markdown(f"**Confidence mean:** `{'-' if conf is None else conf}`")

    if result["reconstructed"] is not None:
        st.markdown("### Rekonstruksi")
        st.image(rv.to_rgb(result["reconstructed"]), width="stretch")
        st.download_button(
            "Unduh Hasil Rekonstruksi",
            data=_png_bytes(result["reconstructed"]),
            file_name="reconstructed.png",
            key="sol_png",
        )
        st.download_button(
            "Unduh solution.json",
            data=_json_bytes(result["solution"]),
            file_name="solution.json",
            key="sol_json",
        )

    st.markdown("### Band strip yang dibandingkan")
    piece_name = sorted(result["pieces"])[0]
    st.image(rv.strip_band_image(result["pieces"][piece_name], meta["t"]), width=256)
    st.caption(
        f"Band merah setebal {meta['t']} piksel pada tepi potongan ini yang dicocokkan: sisi atas potongan dibandingkan "
        "dengan sisi bawah potongan tetangganya (atau ring), dihitung sebagai rata-rata MSE sepanjang band. "
        "Makin tebal t, makin banyak baris yang ikut dihitung dan waktu solve naik."
    )

    st.markdown("### Animasi langkah")
    steps = result["steps"]
    if not steps:
        st.info("Tidak ada langkah backtracking terekam (tidak ada kandidat yang lolos tau).")
        return
    st.caption(f"{len(steps)} langkah backtracking.")
    step_i = st.slider("Langkah ke-", 0, len(steps), key="sol_step")
    board_slot = st.empty()
    if st.button("Putar animasi", key="sol_play"):
        for i in range(1, len(steps) + 1):
            board_slot.image(
                rv.board_state_image(result["pieces"], steps, i, m),
                width="stretch",
            )
            time.sleep(0.12)
    board_slot.image(
        rv.board_state_image(result["pieces"], steps, step_i, m),
        width="stretch",
    )
    if step_i > 0:
        ev = steps[step_i - 1]
        act = "pasang" if ev["type"] == "place" else "lepas"
        st.caption(f"Langkah {step_i}: {act} {ev['pid']} di posisi {ev['pos']} (rotasi {ev['rot']} derajat).")


def _png_bytes(img):
    import cv2

    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def _json_bytes(obj):
    import json

    return json.dumps(obj, indent=2).encode("utf-8")
