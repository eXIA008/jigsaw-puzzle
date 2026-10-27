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
    raise ValueError("Pilih gambar terlebih dahulu.")


def render():
    with st.sidebar:
        st.markdown("**Puzzle Generator**")
        st.caption("Mode untuk membuat puzzle dari gambar yang diunggah pengguna. ")
        st.file_uploader(
            "Pilih Gambar", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
            key="gen_up")

        st.markdown("**Pengaturan Puzzle**")
        st.slider(
            "Ukuran Grid Puzzle", 2, 10, 3, key="gen_m",
            help="Jumlah potongan per sisi. Nilai m berarti puzzle terdiri dari m x m potongan dengan ukuran kira-kira lebar gambar dibagi m. Semakin besar m, makin kecil tiap potongan.",
        )
        st.slider(
            "Ukuran Bingkai", 0, 100, 40, key="gen_hint",
            help="Tebal ring hint di tepi luar papan, dalam piksel. Ring dibaca solver sebagai petunjuk tepi luar, nilai 0 berarti tanpa petunjuk sehingga solver hanya mengandalkan sambungan antar potongan.",
        )
        st.selectbox(
            "Placement Hint", ["split", "inside", "outside"], key="gen_place",
            help="Menentukan penempatan ring hint pada bingkai. "
                 "split: ring dipisah dari gambar, disimpan sebagai petunjuk, dan dipakai solver sebagai tepi luar papan. "
                 "inside: salinan tepi menempel menutupi tepi papan yang ukurannya sama dengan foto. "
                 "outside: salinan tepi ditaruh di luar area grid pada papan yang lebih besar dari foto. "
                 "inside dan outside tidak menghasilkan ring, jadi solver menyusun hanya dari sambungan antar potongan.",
        )
        st.selectbox("Mode Ukur Gambar", ["crop", "resize"], key="gen_fit",
                     help="Mode untuk menentukan ukuran gambar sebelum di proses puzzle generator. 'crop'= Gambar hanya dipotong (lossless pixel)" \
                     " 'resize'= Gambar dibesarkan/dikecilkan agar tetap terlihat.")
        st.checkbox(
            "Rotasi acak potongan", value=True, key="gen_rotate")

        st.markdown("**Pengaturan lain**")
        st.text_input("Seed", value="", key="gen_seed", 
                      help="Seed untuk Reproducibility dari puzzle generator. Kosong = hasil selalu acak")

        st.checkbox(
            "Simpan ground_truth.json", value=True, key="gen_truth")
        if not rv.is_cloud():
            st.text_input("Output Path", value="", key="gen_out", 
            help="Directory atau Folder output dari puzzle generator. Kosong = hasil generator disimpan di folder sesi")

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

    st.markdown("## Hasil generate")
    st.markdown(f"**Grid:** `{result['m']} x {result['m']}`")
    st.markdown(f"**Potongan:** `{result['count']}`")
    st.markdown(f"**Ukuran papan:** `{result['image'].shape[1]} x {result['image'].shape[0]} px`")
    st.markdown(f"**Ukuran potongan:** `{result['piece_size'][1]} x {result['piece_size'][0]} px`")
    st.markdown(f"**Hint width:** `{result['hint_width']}`")

    st.markdown("### Gambar asli")
    st.image(rv.to_rgb(result["raw"]), width="stretch")

    st.markdown("### Bingkai")
    st.image(rv.to_rgb(result["frame"]), width="stretch")

    st.markdown("### Potongan (diacak)")
    montage = rv.piece_montage(result["pieces"])
    if montage is not None:
        st.image(montage, width="stretch")

    out_dir = st.session_state.get("last_gen_dir")
    if out_dir and os.path.isdir(out_dir):
        st.download_button(
            "Unduh ZIP Hasil",
            data=rv.zip_dir_bytes(out_dir),
            file_name="hasil_puzzle.zip",
            key="gen_zip",
        )
