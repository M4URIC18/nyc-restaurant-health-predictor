"""
Shared UI components for CleanKitchen NYC Streamlit app.
"""

import streamlit as st
import os


def load_css():
    """Load the external CSS file."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".streamlit", "style.css")

    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        # Fallback: minimal styles if CSS file not found
        st.markdown("""
            <style>
            .stApp { background-color: #FFFFFF; }
            </style>
        """, unsafe_allow_html=True)


def setup_page(title: str = "CleanKitchen NYC", layout: str = "wide"):
    """
    Configure page settings and load CSS.
    Call this at the top of each page.
    """
    st.set_page_config(
        page_title=title,
        layout=layout,
        initial_sidebar_state="collapsed"
    )
    load_css()


def render_top_nav():
    """
    Render the top navigation bar with brand and page links.
    Uses st.page_link() for stateless navigation.
    """
    # Navigation row: Brand | Nav Links
    brand_col, nav_col = st.columns([1, 2])

    with brand_col:
        st.page_link("app.py", label="CleanKitchen NYC", icon=None)

    with nav_col:
        # Create columns for nav buttons
        cols = st.columns([1, 1, 1, 2])

        with cols[0]:
            st.page_link("pages/1_Filter.py", label="Filter")

        with cols[1]:
            st.page_link("pages/2_About.py", label="About")

        with cols[2]:
            st.page_link("pages/3_Creators.py", label="Creators")


def render_header_divider():
    """Render the horizontal divider after navbar."""
    st.markdown('<div class="header-separator"></div>', unsafe_allow_html=True)
