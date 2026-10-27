import io
import os
import shutil
import tempfile
import zipfile

import cv2
import numpy as np


def is_cloud():
    return os.path.isdir("/mount/src")


def session_workdir():
    import streamlit as st

    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="puzzle_studio_")
    return st.session_state.workdir


def to_rgb(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def piece_montage(pieces, cols=None, cell=140, pad=8, bg=(250, 250, 250)):
    n = len(pieces)
    if not n:
        return None
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    size = rows * (cell + pad) + pad
    canvas = np.full((size, cols * (cell + pad) + pad, 3), bg, np.uint8)
    for i, p in enumerate(pieces):
        r, c = divmod(i, cols)
        thumb = cv2.resize(to_rgb(p), (cell, cell), interpolation=cv2.INTER_AREA)
        y, x = pad + r * (cell + pad), pad + c * (cell + pad)
        canvas[y:y + cell, x:x + cell] = thumb
    return canvas


def strip_band_image(piece, t, cell=256):
    img = cv2.resize(to_rgb(piece), (cell, cell), interpolation=cv2.INTER_AREA)
    band = max(2, int(round(t * cell / piece.shape[0])))
    band = min(band, cell // 3)
    out = img.copy()
    fill = out.copy()
    cv2.rectangle(fill, (0, 0), (cell - 1, band - 1), (255, 75, 75), -1)
    out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
    cv2.rectangle(out, (0, 0), (cell - 1, band - 1), (194, 31, 31), max(2, cell // 110))
    return out


def board_state_image(pieces_by_name, steps, upto, m, cell=120, pad=6):
    state = {}
    for ev in steps[:upto]:
        if ev["type"] == "place":
            state[ev["pos"]] = (ev["pid"], ev["rot"])
        else:
            state.pop(ev["pos"], None)

    grid_px = m * (cell + pad) + pad
    canvas = np.full((grid_px, grid_px, 3), (238, 240, 243), np.uint8)
    for pos, (pid, rot) in state.items():
        r, c = divmod(pos, m)
        y, x = pad + r * (cell + pad), pad + c * (cell + pad)
        img = np.rot90(to_rgb(pieces_by_name[pid]), (4 - rot // 90) % 4)
        thumb = cv2.resize(np.ascontiguousarray(img), (cell, cell), interpolation=cv2.INTER_AREA)
        canvas[y:y + cell, x:x + cell] = thumb

    if upto > 0:
        active = steps[upto - 1]["pos"]
        r, c = divmod(active, m)
        y, x = pad + r * (cell + pad), pad + c * (cell + pad)
        cv2.rectangle(canvas, (x, y), (x + cell - 1, y + cell - 1), (17, 24, 39), 3)
    return canvas


def zip_dir_bytes(path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for base, _dirs, files in os.walk(path):
            for name in files:
                full = os.path.join(base, name)
                rel = os.path.relpath(full, path).replace(os.sep, "/")
                zf.write(full, rel)
    return buf.getvalue()


def save_upload(uploaded, dest_dir, name=None):
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, name or uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def extract_zip(uploaded, dest_dir):
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(uploaded.getbuffer())) as zf:
        for member in zf.namelist():
            target = os.path.normpath(os.path.join(dest_dir, member))
            if not target.startswith(os.path.abspath(dest_dir)):
                raise ValueError("File ZIP tidak sah.")
        zf.extractall(dest_dir)
    return dest_dir
