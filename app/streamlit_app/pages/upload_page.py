"""upload page
"""

import time
import streamlit as st
from streamlit_lottie import st_lottie, st_lottie_spinner
from components.configs import PROJECT_LOGO, PROCESSING_ANIMATION, RESULT_PAGE
# from components.utils import load_lottiefile, #extract_landmarks_features

# Set page configuration
st.set_page_config(page_title="Upload", page_icon="🧏", initial_sidebar_state="auto")

# Display project logo and title
st.image(PROJECT_LOGO, width=300)

st.title("_Welcome to NSL-2-AUDIO_ 🧏")

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

st.divider()

st.markdown(
    "<div style='text-align: justify;'><p style='text-indent: 2em;'>"
    "Please "
    "</p></div>",
    unsafe_allow_html=True,
)

sign_lang_video = st.file_uploader(
    label=":red[Upload Video]", type=["MP4", "PNG"], on_change=None
)

# processing_animation_ = load_lottiefile(PROCESSING_ANIMATION)

if sign_lang_video is not None:
    st.video(sign_lang_video)
    st.success("Your file has been uploaded successfully!")
    st.write("Click the button to perform translation.")
    if st.button(label=""):
        # st_lottie(processing_animation_)
        # TODO Extract feature landmarks
        # data = extract_landmarks_features(sign_lang_video)

        # TODO Send data to model inference

        # st.switch_page(page=RESULT_PAGE)
        pass

# st.page_link("app.py", label="Go Back", icon="🏠")
