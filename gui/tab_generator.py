import os

import streamlit as st

import gui.render as rv
from gui import adapters


def _resolve_image():
    uploaded = st.session_state.get("gen_up")
    if uploaded is not None:
        dest = os.path.join(rv.session_workdir(), "uploads")
        return rv.save_upload(uploaded, dest)
    path = (st.session_state.get("gen_path") or "").strip()
    if path:
        return path
    raise ValueError("Pilih gambar dulu (unggah atau isi path lokal).")


def render():
    with st.sidebar:
        st.markdown("### Generator")
        st.file_uploader(
            "Gambar", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
            key="gen_up",
        )
        if not rv.is_cloud():
            st.text_input("Atau path gambar (lokal)", value="", key="gen_path",
                          placeholder="foto.jpg")
        col1, col2 = st.columns(2)
        col1.number_input("M (grid)", min_value=2, max_value=8, value=3, step=1, key="gen_m")
        col2.number_input("Hint width", min_value=0, max_value=64, value=40, key="gen_hint")
        st.selectbox("Placement hint", ["split", "inside", "outside"], key="gen_place")
        st.selectbox("Mode ukur gambar", ["crop", "resize"], key="gen_fit")
        st.checkbox("Rotasi acak potongan", value=True, key="gen_rotate")
        st.text_input("Seed (kosong = acak)", value="", key="gen_seed")
        st.checkbox("Simpan ground_truth.json", value=True, key="gen_truth")
        if not rv.is_cloud():
            st.text_input("Folder keluaran (kosong = folder sesi)", value="", key="gen_out")
        run = st.button("Generate", type="primary", width="stretch", key="gen_run")

    if run:
        try:
            image_path = _resolve_image()
            seed_txt = (st.session_state.get("gen_seed") or "").strip()
            seed = int(seed_txt) if seed_txt else None
            out_dir = (st.session_state.get("gen_out") or "").strip()
            if not out_dir:
                out_dir = os.path.join(rv.session_workdir(), "hasil")
            with st.spinner("Memotong puzzle..."):
                result = adapters.generate_puzzle(
                    image_path, out_dir,
                    m=int(st.session_state["gen_m"]),
                    hint_width=int(st.session_state["gen_hint"]),
                    placement=st.session_state["gen_place"],
                    fit_mode=st.session_state["gen_fit"],
                    rotate=bool(st.session_state["gen_rotate"]),
                    seed=seed,
                    save_truth=bool(st.session_state["gen_truth"]),
                )
            st.session_state.gen_result = result
            st.session_state.last_gen_dir = out_dir
            st.session_state.pop("solve_result", None)
            st.success(f"Selesai: {result['count']} potongan tersimpan di {out_dir}")
        except (ValueError, FileNotFoundError, OSError) as exc:
            st.error(str(exc))

    result = st.session_state.get("gen_result")
    if result is None:
        st.info("Belum ada hasil. Pilih gambar, atur parameter, lalu tekan Generate.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(rv.to_rgb(result["raw"]), caption="Gambar asli", width="stretch")
    with c2:
        st.image(rv.to_rgb(result["frame"]), caption="Bingkai", width="stretch")
    with c3:
        montage = rv.piece_montage(result["pieces"])
        if montage is not None:
            st.image(montage, caption=f"Potongan ({result['count']}, diacak)", width="stretch")

    st.caption(
        f"Ukuran papan {result['image'].shape[1]}x{result['image'].shape[0]} px, "
        f"grid {result['m']}x{result['m']}, potongan {result['piece_size'][1]}x{result['piece_size'][0]} px, "
        f"hint width {result['hint_width']}"
    )

    out_dir = st.session_state.get("last_gen_dir")
    if out_dir and os.path.isdir(out_dir):
        st.download_button(
            "Unduh ZIP hasil",
            data=rv.zip_dir_bytes(out_dir),
            file_name="hasil_puzzle.zip",
            key="gen_zip",
        )
