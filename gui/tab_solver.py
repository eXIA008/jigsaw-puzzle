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
        st.markdown("### Solver")
        sources = ["Hasil sesi ini", "Unggah ZIP"]
        if not rv.is_cloud():
            sources.append("Folder lokal")
        st.selectbox("Sumber data", sources, key="sol_src")
        if st.session_state.get("sol_src") == "Unggah ZIP":
            st.file_uploader("ZIP hasil generate", type=["zip"], key="sol_zip")
        elif st.session_state.get("sol_src") == "Folder lokal":
            st.text_input("Folder hasil", value="hasil", key="sol_dir")
        st.number_input("t (tebal strip)", min_value=1, max_value=16, value=3, step=1, key="sol_t")
        st.number_input("tau (ambang MSE)", min_value=0.0, value=2500.0, step=100.0,
                        format="%.1f", key="sol_tau")
        st.number_input("Max kandidat", min_value=1, max_value=20, value=5, step=1, key="sol_maxc")
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
            meta = result["solve_meta"]
            filled = len(result["solution"])
            total = result["m"] * result["m"]
            status = "berhasil" if result["ok"] else "gagal"
            st.success(
                f"Solve {status}: {filled}/{total} potongan terisi dalam {meta['time_s']} s."
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            st.error(str(exc))

    result = st.session_state.get("solve_result")
    if result is None:
        st.info("Belum ada hasil. Pilih sumber data, atur parameter, lalu tekan Solve.")
        return

    meta = result["solve_meta"]
    m = result["m"]
    total = m * m
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "ok" if result["ok"] else "gagal")
    c2.metric("Terisi", f"{len(result['solution'])}/{total}")
    c3.metric("Waktu solve", f"{meta['time_s']} s")
    gap = meta.get("gap_mean")
    c4.metric("Gap mean", "-" if gap is None else str(gap))

    if result["reconstructed"] is not None:
        st.image(rv.to_rgb(result["reconstructed"]), caption="Rekonstruksi solver",
                 width="stretch")
        st.download_button(
            "Unduh reconstructed.png",
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

    st.markdown("#### Animasi langkah")
    steps = result["steps"]
    st.caption(f"{len(steps)} langkah backtracking (place dan unplace).")
    n = max(1, len(steps))
    step_i = st.slider("Langkah ke-", 0, len(steps), len(steps), key="sol_step",
                       disabled=not steps)
    board_slot = st.empty()
    if st.button("Putar animasi", key="sol_play") and steps:
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
    if steps and step_i > 0:
        ev = steps[step_i - 1]
        act = "pasang" if ev["type"] == "place" else "lepas"
        st.caption(f"Langkah {step_i}: {act} {ev['pid']} di posisi {ev['pos']} (rotasi {ev['rot']}°).")


def _png_bytes(img):
    import cv2

    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def _json_bytes(obj):
    import json

    return json.dumps(obj, indent=2).encode("utf-8")
