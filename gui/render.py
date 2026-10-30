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


def resolve_cell_dimensions(m, piece_size=None, pad=6, max_w=940, max_h=680):
    if piece_size is not None and len(piece_size) >= 2 and piece_size[0] > 0 and piece_size[1] > 0:
        d1, d2 = int(piece_size[0]), int(piece_size[1])
    else:
        d1, d2 = 100, 100

    pw = max(d1, d2)
    ph = min(d1, d2)
    aspect = pw / ph 

    cell_w = max(70, min(360, int((max_w - (m + 1) * pad) / m)))
    cell_h = max(50, int(round(cell_w / aspect)))
    grid_h = m * (cell_h + pad) + pad
    if grid_h > max_h:
        cell_h = max(50, int((max_h - (m + 1) * pad) / m))
        cell_w = max(50, int(round(cell_h * aspect)))

    grid_w = m * (cell_w + pad) + pad
    grid_h = m * (cell_h + pad) + pad
    return cell_w, cell_h, grid_w, grid_h


def board_state_image(pieces_by_name, steps, upto, m, cell=120, pad=6, piece_size=None):
    state = {}
    for ev in steps[:upto]:
        if ev["type"] == "place":
            state[ev["pos"]] = (ev["pid"], ev["rot"])
        else:
            state.pop(ev["pos"], None)

    if piece_size is None and pieces_by_name:
        sample = next(iter(pieces_by_name.values()))
        piece_size = sample.shape[:2]

    if piece_size is not None and len(piece_size) >= 2 and piece_size[0] > 0 and piece_size[1] > 0:
        d1, d2 = int(piece_size[0]), int(piece_size[1])
        pw = max(d1, d2)
        ph = min(d1, d2)
        aspect = pw / ph
        cell_w = cell
        cell_h = max(1, int(round(cell_w / aspect)))
    else:
        cell_w = cell
        cell_h = cell

    grid_w = m * (cell_w + pad) + pad
    grid_h = m * (cell_h + pad) + pad
    canvas = np.full((grid_h, grid_w, 3), (238, 240, 243), np.uint8)
    for pos, (pid, rot) in state.items():
        r, c = divmod(pos, m)
        y, x = pad + r * (cell_h + pad), pad + c * (cell_w + pad)
        img = np.rot90(to_rgb(pieces_by_name[pid]), (4 - rot // 90) % 4)
        thumb = cv2.resize(np.ascontiguousarray(img), (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[y:y + cell_h, x:x + cell_w] = thumb

    if upto > 0:
        active = steps[upto - 1]["pos"]
        r, c = divmod(active, m)
        y, x = pad + r * (cell_h + pad), pad + c * (cell_w + pad)
        cv2.rectangle(canvas, (x, y), (x + cell_w - 1, y + cell_h - 1), (17, 24, 39), 3)
    return canvas


def pieces_to_base64_dict(pieces, cell_w=160, cell_h=160):
    import base64

    b64_dict = {}
    for pid, img in pieces.items():
        rgb = to_rgb(img)
        ih, iw = rgb.shape[:2]
        if iw >= ih:
            tw, th = cell_w, cell_h
        else:
            tw, th = cell_h, cell_w
        thumb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        if bgr.ndim == 3 and bgr.shape[2] == 4:
            ok, buf = cv2.imencode(".png", bgr)
            mime = "image/png"
        else:
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            mime = "image/jpeg"
        if ok:
            b64_dict[pid] = f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return b64_dict


def steps_to_gif_bytes(pieces_by_name, steps, m, cell=120, pad=6, duration_ms=120, piece_size=None):
    from PIL import Image

    frames = []
    f0 = board_state_image(pieces_by_name, steps, 0, m, cell=cell, pad=pad, piece_size=piece_size)
    frames.append(Image.fromarray(f0))
    for i in range(1, len(steps) + 1):
        f = board_state_image(pieces_by_name, steps, i, m, cell=cell, pad=pad, piece_size=piece_size)
        frames.append(Image.fromarray(f))
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return buf.getvalue()


def build_player_html(pieces_b64, steps, m, cell_w=160, cell_h=160, pad=6):
    import json

    grid_w = m * (cell_w + pad) + pad
    grid_h = m * (cell_h + pad) + pad
    max_card_w = max(640, min(1040, grid_w + 40))
    clean_steps = []
    for s in steps:
        clean_steps.append({
            "type": s["type"],
            "pos": int(s["pos"]),
            "pid": str(s["pid"]),
            "rot": int(s["rot"]),
            "score": round(float(s["score"]), 1) if "score" in s and s["score"] is not None else None,
        })
    steps_json = json.dumps(clean_steps)
    pieces_json = json.dumps(pieces_b64)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #31333F;
    background: transparent;
    padding: 6px;
    display: flex;
    justify-content: center;
    user-select: none;
  }}
  .player-card {{
    background: #FFFFFF;
    border: 1px solid #E6E9EF;
    border-radius: 8px;
    padding: 16px;
    width: 100%;
    max-width: {max_card_w}px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }}
  .status-bar {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 13px;
  }}
  .step-badge {{
    background: #F0F2F6;
    color: #31333F;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
    border: 1px solid #DCE1E8;
  }}
  .action-text {{
    flex: 1;
    text-align: right;
    font-size: 12.5px;
    color: #31333F;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .canvas-wrapper {{
    position: relative;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #DCE1E8;
    background: #EEF0F3;
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);
  }}
  canvas {{
    display: block;
    max-width: 100%;
    height: auto;
  }}
  .scrubber-container {{
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .step-label {{
    font-size: 11px;
    color: #6C727F;
    font-family: ui-monospace, monospace;
  }}
  .scrubber {{
    flex: 1;
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    border-radius: 3px;
    background: #DCE1E8;
    outline: none;
    cursor: pointer;
  }}
  .scrubber::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #FF4B4B;
    cursor: pointer;
    border: 2px solid #FFFFFF;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    transition: transform 0.1s;
  }}
  .scrubber::-webkit-slider-thumb:hover {{
    transform: scale(1.18);
  }}
  .controls-row {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    flex-wrap: wrap;
  }}
  .btn-group {{
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  button {{
    font-family: inherit;
    font-size: 12.5px;
    padding: 5px 10px;
    border: 1px solid #DCE1E8;
    border-radius: 4px;
    background: #FFFFFF;
    color: #31333F;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    transition: all 0.15s ease;
  }}
  button:hover {{
    background: #F0F2F6;
    border-color: #B4B9C2;
  }}
  button:active {{
    background: #E4E7EB;
  }}
  button.primary-btn {{
    background: #FF4B4B;
    border-color: #FF4B4B;
    color: #FFFFFF;
    font-weight: 600;
    min-width: 80px;
  }}
  button.primary-btn:hover {{
    background: #E03E3E;
    border-color: #E03E3E;
  }}
  button.primary-btn.playing {{
    background: #111827;
    border-color: #111827;
  }}
  .speed-select {{
    font-family: inherit;
    font-size: 12px;
    padding: 4px 6px;
    border: 1px solid #DCE1E8;
    border-radius: 4px;
    background: #FFFFFF;
    color: #31333F;
    cursor: pointer;
    outline: none;
  }}
  .speed-select:focus {{
    border-color: #FF4B4B;
  }}
  .loop-label {{
    font-size: 12px;
    color: #31333F;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
  }}
</style>
</head>
<body>
<div class="player-card">
  <div class="status-bar">
    <span id="stepBadge" class="step-badge">Langkah 0 / {len(clean_steps)}</span>
    <div id="statusText" class="action-text">Papan awal</div>
  </div>

  <div class="canvas-wrapper">
    <canvas id="puzzleCanvas" width="{grid_w}" height="{grid_h}"></canvas>
  </div>

  <div class="scrubber-container">
    <span class="step-label">0</span>
    <input type="range" id="scrubber" class="scrubber" min="0" max="{len(clean_steps)}" value="0">
    <span class="step-label">{len(clean_steps)}</span>
  </div>

  <div class="controls-row">
    <div class="btn-group">
      <button id="btnStart" title="Ke Awal">⏮</button>
      <button id="btnPrev" title="Mundur 1 Langkah">◀</button>
      <button id="btnPlay" class="primary-btn" title="Putar / Jeda Animasi">▶ Putar</button>
      <button id="btnNext" title="Maju 1 Langkah">▶</button>
      <button id="btnEnd" title="Ke Akhir">⏭</button>
    </div>
    <div class="btn-group">
      <select id="speedSelect" class="speed-select" title="Kecepatan Animasi">
        <option value="350">0.5x</option>
        <option value="180" selected>1.0x</option>
        <option value="90">2.0x</option>
        <option value="40">4.0x</option>
        <option value="15">10x</option>
      </select>
      <label class="loop-label" title="Ulang animasi otomatis saat mencapai akhir">
        <input type="checkbox" id="loopCheck"> Loop
      </label>
      <button id="btnCapture" title="Unduh gambar frame saat ini sebagai PNG">📸 Simpan</button>
    </div>
  </div>
</div>

<script>
(function() {{
  const steps = {steps_json};
  const piecesData = {pieces_json};
  const m = {m};
  const cellW = {cell_w};
  const cellH = {cell_h};
  const pad = {pad};
  const totalSteps = steps.length;

  const canvas = document.getElementById("puzzleCanvas");
  const ctx = canvas.getContext("2d");
  const scrubber = document.getElementById("scrubber");
  const stepBadge = document.getElementById("stepBadge");
  const statusText = document.getElementById("statusText");
  const btnPlay = document.getElementById("btnPlay");
  const speedSelect = document.getElementById("speedSelect");
  const loopCheck = document.getElementById("loopCheck");

  // Precompute board state for every step
  const states = [];
  let curr = {{}};
  states.push(Object.assign({{}}, curr));
  for (let i = 0; i < steps.length; i++) {{
    const ev = steps[i];
    curr = Object.assign({{}}, curr);
    if (ev.type === "place") {{
      curr[ev.pos] = {{ pid: ev.pid, rot: ev.rot }};
    }} else {{
      delete curr[ev.pos];
    }}
    states.push(curr);
  }}

  // Preload pieces
  const images = {{}};
  let loadedCount = 0;
  const pieceKeys = Object.keys(piecesData);
  const totalPieces = pieceKeys.length;

  function onImgLoad() {{
    loadedCount++;
    if (loadedCount >= totalPieces) {{
      drawStep(currentStep);
    }}
  }}

  for (let k of pieceKeys) {{
    const im = new Image();
    im.onload = onImgLoad;
    im.src = piecesData[k];
    images[k] = im;
  }}

  let currentStep = 0;
  let isPlaying = false;
  let timer = null;

  function drawStep(stepIndex) {{
    currentStep = Math.max(0, Math.min(stepIndex, totalSteps));
    scrubber.value = currentStep;
    stepBadge.innerText = "Langkah " + currentStep + " / " + totalSteps;

    if (currentStep === 0) {{
      statusText.innerHTML = "<span>🏁 Papan awal (kosong)</span>";
    }} else {{
      const ev = steps[currentStep - 1];
      const r = Math.floor(ev.pos / m) + 1;
      const c = (ev.pos % m) + 1;
      if (ev.type === "place") {{
        const scoreStr = ev.score !== null ? (" (MSE: " + ev.score + ")") : "";
        statusText.innerHTML = "<span style='color:#0E7039;font-weight:600;'>🟢 Pasang</span> <b>" + ev.pid + "</b> di (" + r + "," + c + ") rotasi " + ev.rot + "°" + scoreStr;
      }} else {{
        statusText.innerHTML = "<span style='color:#D32F2F;font-weight:600;'>🔴 Lepas [Backtrack]</span> <b>" + ev.pid + "</b> dari (" + r + "," + c + ")";
      }}
    }}

    // Background
    ctx.fillStyle = "#EEF0F3";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const state = states[currentStep];

    // Draw cells
    for (let r = 0; r < m; r++) {{
      for (let c = 0; c < m; c++) {{
        const pos = r * m + c;
        const x = pad + c * (cellW + pad);
        const y = pad + r * (cellH + pad);

        ctx.fillStyle = "#E4E7EB";
        ctx.fillRect(x, y, cellW, cellH);

        if (state && state[pos]) {{
          const item = state[pos];
          const im = images[item.pid];
          if (im && im.complete) {{
            ctx.save();
            ctx.translate(x + cellW / 2, y + cellH / 2);
            ctx.rotate((item.rot * Math.PI) / 180);
            if (item.rot === 90 || item.rot === 270) {{
              ctx.drawImage(im, -cellH / 2, -cellW / 2, cellH, cellW);
            }} else {{
              ctx.drawImage(im, -cellW / 2, -cellH / 2, cellW, cellH);
            }}
            ctx.restore();
          }}
        }}
      }}
    }}

    // Highlight active cell
    if (currentStep > 0) {{
      const ev = steps[currentStep - 1];
      const r = Math.floor(ev.pos / m);
      const c = ev.pos % m;
      const x = pad + c * (cellW + pad);
      const y = pad + r * (cellH + pad);

      if (ev.type === "place") {{
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 3;
        ctx.strokeRect(x + 1.5, y + 1.5, cellW - 3, cellH - 3);
      }} else {{
        ctx.strokeStyle = "#FF4B4B";
        ctx.lineWidth = 3;
        ctx.strokeRect(x + 1.5, y + 1.5, cellW - 3, cellH - 3);

        ctx.beginPath();
        ctx.moveTo(x + cellW * 0.35, y + cellH * 0.35);
        ctx.lineTo(x + cellW * 0.65, y + cellH * 0.65);
        ctx.moveTo(x + cellW * 0.65, y + cellH * 0.35);
        ctx.lineTo(x + cellW * 0.35, y + cellH * 0.65);
        ctx.stroke();
      }}
    }}
  }}

  function pause() {{
    if (timer) {{
      clearInterval(timer);
      timer = null;
    }}
    isPlaying = false;
    btnPlay.innerText = "▶ Putar";
    btnPlay.classList.remove("playing");
  }}

  function play() {{
    if (currentStep >= totalSteps) {{
      currentStep = 0;
    }}
    isPlaying = true;
    btnPlay.innerText = "⏸ Jeda";
    btnPlay.classList.add("playing");
    const delay = parseInt(speedSelect.value, 10) || 180;
    timer = setInterval(() => {{
      if (currentStep < totalSteps) {{
        drawStep(currentStep + 1);
      }} else {{
        if (loopCheck.checked) {{
          drawStep(0);
        }} else {{
          pause();
        }}
      }}
    }}, delay);
  }}

  btnPlay.addEventListener("click", () => {{
    if (isPlaying) pause(); else play();
  }});

  document.getElementById("btnStart").addEventListener("click", () => {{ pause(); drawStep(0); }});
  document.getElementById("btnEnd").addEventListener("click", () => {{ pause(); drawStep(totalSteps); }});
  document.getElementById("btnPrev").addEventListener("click", () => {{ pause(); drawStep(currentStep - 1); }});
  document.getElementById("btnNext").addEventListener("click", () => {{ pause(); drawStep(currentStep + 1); }});

  scrubber.addEventListener("input", (e) => {{
    pause();
    drawStep(parseInt(e.target.value, 10));
  }});

  speedSelect.addEventListener("change", () => {{
    if (isPlaying) {{
      pause();
      play();
    }}
  }});

  document.getElementById("btnCapture").addEventListener("click", () => {{
    const a = document.createElement("a");
    a.download = "langkah_" + currentStep + ".png";
    a.href = canvas.toDataURL("image/png");
    a.click();
  }});

  drawStep(0);
}})();
</script>
</body>
</html>"""


def render_solver_player(pieces, steps, m, cell_w=None, cell_h=None, pad=6, piece_size=None):
    import streamlit.components.v1 as components

    if piece_size is None and pieces:
        sample = next(iter(pieces.values()))
        piece_size = sample.shape[:2]

    if cell_w is None or cell_h is None:
        cell_w, cell_h, grid_w, grid_h = resolve_cell_dimensions(
            m, piece_size, pad=pad, max_w=940, max_h=680
        )
    else:
        grid_w = m * (cell_w + pad) + pad
        grid_h = m * (cell_h + pad) + pad

    b64_dict = pieces_to_base64_dict(pieces, cell_w=cell_w, cell_h=cell_h)
    html = build_player_html(b64_dict, steps, m, cell_w=cell_w, cell_h=cell_h, pad=pad)
    components.html(html, height=grid_h + 195, scrolling=False)



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
