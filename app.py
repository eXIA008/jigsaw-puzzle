import streamlit as st

from gui import tab_evaluator, tab_generator, tab_solver
from gui.theme import apply_theme

st.set_page_config(page_title="Puzzle Studio", layout="wide", initial_sidebar_state="expanded")
apply_theme()

st.markdown("# Jigsaw Puzzle")
st.markdown(
    "Jigsaw Puzzle adalah aplikasi web berbasis Streamlit untuk membuat puzzle jigsaw dari gambar yang diunggah pengguna. " \
    "Aplikasi ini punya tiga mode: Puzzle Generator, Puzzle Solver, dan Evaluate.\n"
)

section = st.radio(
    "Bagian", ["Puzzle Generator", "Puzzle Solver", "Evaluate"],
    horizontal=True, key="nav", label_visibility="collapsed",
)

if section == "Puzzle Generator":
    tab_generator.render()
elif section == "Puzzle Solver":
    tab_solver.render()
else:
    tab_evaluator.render()
