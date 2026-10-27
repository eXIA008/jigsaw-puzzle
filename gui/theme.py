CSS = """
<style>
#MainMenu, footer { visibility: hidden; }
.stApp { background: #FFFFFF; color: #31333F; }
section[data-testid="stSidebar"] { background: #F0F2F6; }
</style>
"""


def apply_theme():
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)
