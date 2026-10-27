PALETTE = {
    "bg": "#FAFAFA",
    "card": "#FFFFFF",
    "line": "#E5E7EB",
    "ink": "#111827",
    "ink_soft": "#6B7280",
    "accent": "#1F2937",
    "sidebar": "#F3F4F6",
}

CSS = """
<style>
.stApp { background: #FAFAFA; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
    background: #F3F4F6;
    border-right: 1px solid #E5E7EB;
}
#MainMenu, footer { visibility: hidden; }

h1, h2, h3, h4 { color: #111827; font-weight: 600; letter-spacing: -0.01em; }
p, label, span, li { color: #111827; }
.stCaption, .stMarkdown p small { color: #6B7280; }

.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    border: 1px solid #E5E7EB;
    background: #FFFFFF;
    color: #111827;
}
.stButton > button[kind="primary"] {
    background: #1F2937;
    border: 1px solid #1F2937;
    color: #FFFFFF;
}
.stButton > button:hover { filter: brightness(0.96); }
.stButton > button[kind="primary"]:hover { filter: brightness(1.25); }
.stButton > button:focus-visible { outline: 2px solid #1F2937; outline-offset: 2px; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px;
    background: #FFFFFF;
}
[data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px;
    background: #FFFFFF;
}
[data-testid="stMetric"] {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    background: #FFFFFF;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #6B7280; }
[data-testid="stMetricValue"] { color: #111827; }

.st-key-nav [data-baseweb="radio-group"] {
    gap: 0;
    border-bottom: 1px solid #E5E7EB;
}
.st-key-nav [data-baseweb="radio"] > div:first-child { display: none; }
.st-key-nav label {
    margin: 0 28px 0 0;
    padding: 10px 2px 8px;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    color: #6B7280;
    cursor: pointer;
}
.st-key-nav label:has(input:checked) {
    color: #111827;
    font-weight: 600;
    border-bottom: 2px solid #111827;
}
.st-key-nav label:hover { color: #111827; }

[data-testid="stFileUploader"] {
    border: 1px dashed #E5E7EB;
    border-radius: 8px;
    background: #FFFFFF;
}
</style>
"""


def apply_theme():
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)
