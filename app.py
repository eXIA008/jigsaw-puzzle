import streamlit as st

from gui import tab_evaluator, tab_generator, tab_solver
from gui.theme import apply_theme

st.set_page_config(page_title="Puzzle Studio", layout="wide")
apply_theme()

st.markdown("## Puzzle Studio")
st.caption("Generator, solver, dan evaluator puzzle gambar")

section = st.radio(
    "Bagian", ["Generator", "Solver", "Evaluate"],
    horizontal=True, key="nav", label_visibility="collapsed",
)

if section == "Generator":
    tab_generator.render()
elif section == "Solver":
    tab_solver.render()
else:
    tab_evaluator.render()
