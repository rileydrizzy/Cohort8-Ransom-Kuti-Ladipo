"""doc
"""

import streamlit as st
from components.configs import PROJECT_LOGO

# Set page configuration
st.set_page_config(page_title="Upload", page_icon="🧏", initial_sidebar_state="auto")

st.title("_Welcome to NSL-2-AUDIO_ 🧏")

st.divider()

# Display project logo in the sidebar
st.sidebar.image(PROJECT_LOGO, width=300)

# Display project title and description in the sidebar
with st.sidebar:
    st.title("_Welcome to NSL-2-AUDIO_")
    st.markdown(
        "NSL-2-AUDIO is an open-source Automatic Sign Language Translation system, "
        "specifically designed to translate Nigerian Sign Language (NSL) into one of "
        "the Low-Resource Languages (LRLs) spoken in Nigeria."
    )
