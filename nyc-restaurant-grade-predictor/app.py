"""
CleanKitchen NYC - Landing Page

A Streamlit application for predicting NYC restaurant health inspection grades.
This is the main entry point and landing page.
"""

import streamlit as st
from src.components import load_css, render_top_nav, render_header_divider

# --- Page Configuration ---
st.set_page_config(
    page_title="CleanKitchen NYC",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
load_css()

# --- Top Navigation ---
render_top_nav()
render_header_divider()

# --- Hero Section ---
st.markdown('<div class="hero-sub">Health Predicting/Filtering - NYC</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Healthier Food<br>for NYC</div>', unsafe_allow_html=True)

# Get Started button
col1, col2 = st.columns([2, 8])
with col1:
    st.page_link("pages/1_Filter.py", label="Get Started →", icon="🔍")

st.markdown("<br><br>", unsafe_allow_html=True)

# --- Image Marquee ---
image_links = [
    "https://picsum.photos/id/63/900/600",
    "https://picsum.photos/id/292/900/600",
    "https://picsum.photos/id/425/900/600",
    "https://picsum.photos/id/429/900/600",
    "https://picsum.photos/id/488/900/600",
    "https://picsum.photos/id/493/900/600"
]

# Create the HTML structure for marquee items
content_items_html = ""
for url in image_links:
    content_items_html += f"""
    <div class="marquee-item">
        <img src="{url}" alt="Restaurant Image">
    </div>
    """

# Duplicate for seamless loop
looped_content = content_items_html * 2

st.markdown(
    f"""
    <div class="marquee-wrapper">
        <div class="marquee-container">
            <div class="marquee-content">
                {looped_content}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Footer Navigation ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Explore")
    st.page_link("pages/1_Filter.py", label="Filter & Predict", icon="🗺️")

with col2:
    st.markdown("### Learn")
    st.page_link("pages/2_About.py", label="About the Project", icon="📖")

with col3:
    st.markdown("### Connect")
    st.page_link("pages/3_Creators.py", label="Meet the Creators", icon="👥")
