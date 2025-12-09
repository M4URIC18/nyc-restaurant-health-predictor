"""
Creators Page - Team information for CleanKitchen NYC.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.components import load_css, render_top_nav, render_header_divider

# --- Page Configuration ---
st.set_page_config(
    page_title="Creators - CleanKitchen NYC",
    layout="wide",
    initial_sidebar_state="collapsed"
)
load_css()

# --- Top Navigation ---
render_top_nav()
render_header_divider()

# --- Helper Function ---
def render_member_card(name, college, major, linkedin_url, github_url, image_url):
    """Generates the HTML for a single team member card."""
    linkedin_icon_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"
    linkedin_icon_html = f'<img class="linkedin-icon" src="{linkedin_icon_url}" alt="LinkedIn">'

    github_icon_url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
    github_icon_html = f'<img class="github-icon" src="{github_icon_url}" alt="GitHub">'

    # Handle empty URLs gracefully
    linkedin_link = f'<a href="{linkedin_url}" target="_blank">{name}</a>' if linkedin_url else "N/A"
    github_link = f'<a href="{github_url}" target="_blank">{name}</a>' if github_url else "N/A"

    html = f"""
    <div class="team-member-container">
        <div>
            <img class="profile-img" src="{image_url}" alt="{name}'s Photo">
        </div>
        <div class="member-details">
            <strong>{name}</strong>
            <ul>
                <li>College: {college}</li>
                <li>Major: {major}</li>
                <li>LinkedIn: {linkedin_icon_html} {linkedin_link}</li>
                <li>GitHub: {github_icon_html} {github_link}</li>
            </ul>
        </div>
    </div>
    <div class="divider"></div>
    """
    st.markdown(html, unsafe_allow_html=True)


# --- Page Content ---
# --- Team Members ---
render_member_card(
    name="Jack Kaplan",
    college="CUNY - Brooklyn College",
    major="Computer Science",
    linkedin_url="https://www.linkedin.com/in/jackkaplan1",
    github_url="https://github.com/Jack-Kaplan",
    image_url="https://ca.slack-edge.com/T094PKG3ASD-U095B5FJP1P-d40df1de134c-512"
)

render_member_card(
    name="Dominik Kasza",
    college="CUNY - Queens College",
    major="Computer Science",
    linkedin_url="https://www.linkedin.com/in/dominik-kasza-",
    github_url="https://github.com/Guciosk",
    image_url="https://ca.slack-edge.com/T094PKG3ASD-U09628KG754-788280e58034-512"
)

render_member_card(
    name="Mauricio Embus Perez",
    college="John Jay College",
    major="Computer Science",
    linkedin_url="https://www.linkedin.com/in/ma-us-pez08/",
    github_url="https://github.com/M4URIC18",
    image_url="https://ca.slack-edge.com/T094PKG3ASD-U096M8URFAR-dfe4d20b491f-512"
)
