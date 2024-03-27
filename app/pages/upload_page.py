"""
Upload Page

This module defines the Streamlit upload page for the NSL-2-AUDIO project. Users can upload
videos containing Nigerian Sign Language (NSL) signs for translation into spoken audio.

The upload process involves selecting a supported video file format (MP4, AVI, MKV) with a 
front-facing view of a single individual signing. The uploaded video is then processed for 
translation into spoken audio.

"""

import time
import streamlit as st
from streamlit_lottie import st_lottie
from components.configs import PROJECT_LOGO, PROCESSING_ANIMATION
from components.utils import load_lottiefile, extract_landmarks_features

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

st.subheader(
    "Hear your sign language come alive! Ready to translate your ASL video into spoken audio?"
)
st.markdown(
    "We bridge the communication gap, making your sign language understandable for everyone. "
    "Here's how:"
)
st.write("""**1. Click "Choose File".**""")

st.write(
    "**2. Select your ASL video. Ensure it's in a supported format (MP4, AVI, MKV) and features:**"
)
st.write(
    "- **A single individual signing:** For optimal accuracy, only one person should be signing in the video."
)
st.write("- **Front-facing view:** The signer should be facing the camera directly.")

st.markdown(
    """**3. Click "Upload". Lean back and relax! Our system will analyze your video and translate the signs to spoken audio.**"""
)

st.write("**Please note:**")
st.write(
    "- This technology is still under development, and translation accuracy may vary."
)
st.write("- Currently, we only support Yoruba language for now.")
st.write(
    "- Videos with multiple signers or those not facing the camera might experience decreased accuracy."
)

# Allow users to upload sign language video
sign_lang_video = st.file_uploader(
    label="Choose File", type=["MP4", "AVI", "MKV", "PNG"], on_change=None
)

processing_animation_ = load_lottiefile(PROCESSING_ANIMATION)

if sign_lang_video is not None:
    st.video(sign_lang_video)
    st.success("Your file has been uploaded successfully!")
    st.write("Click the button to perform translation.")
    if st.button(label="Start Translation"):
        st_lottie(processing_animation_)
        # TODO Extract feature landmarks
        # data = extract_landmarks_features(sign_lang_video)

        # TODO Send data to model inference
        pass

st.page_link(
    "app.py", label="Go Back to Home Page", icon="🏠", use_container_width=True
)
