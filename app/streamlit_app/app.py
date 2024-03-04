"""
NSL-2-AUDIO Web Application

This module contains the main code for the NSL-2-AUDIO web application, \
an open-source Automatic Sign Language Translation system designed to \
translate Nigerian Sign Language (NSL) into Low-Resource Languages (LRLs) spoken in Nigeria.

Features:
- Integration with Streamlit for the user interface.
- Utilizes Lottie animations for visual enhancements.

Usage:
Run this module to launch the NSL-2-AUDIO application.

Dependencies:
- streamlit
- streamlit-lottie

Example:
    $ python -m streamlit run app.py

"""

import time
import streamlit as st
from streamlit_lottie import st_lottie
from components.configs import (
    UPLOAD_PAGE,
    PROJECT_LOGO,
    NSL_SIGN_GIF,
    GOOGLE_ASL_URL,
    IROYIN_SPEECH_URL,
    LOADING_ANIMATION,
    FOOTER,
    FOOTER_STYLE,
    SLIDES_HTML_CODE,
)
from components.utils import load_lottiefile

# Set page configuration
st.set_page_config(
    page_title="Home", page_icon=PROJECT_LOGO, initial_sidebar_state="auto"
)

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

# Display project logo and title in the main page
st.image(PROJECT_LOGO, width=300)
st.divider()
st.markdown(
    "<h3 style='text-align: center; color: black;'>"
    "Transcending Barriers: A Comprehensive Framework for Sign Language Video Translation"
    "and Speech Generation into Low-Resource Languages (LRLs) in Nigeria </h1>",
    unsafe_allow_html=True,
)

st.markdown("")
st.image(NSL_SIGN_GIF)
st.markdown("")

# Overview section
st.markdown("**<u> OVERVIEW </u>**", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align: justify;">'
    "This project introduces a groundbreaking application designed to bridge the "
    "communication gap between the Nigerian Sign Language (NSL) community and the "
    "broader Nigerian society. The application leverages the power of artificial "
    "intelligence to translate NSL signs into spoken Yoruba, fostering inclusivity "
    "and promoting opportunities in various aspects of life, including: </div>",
    unsafe_allow_html=True,
)

# Display bullet points for overview
st.write(
    "* **Education:** Empowering deaf individuals to access educational resources "
    "and participate actively in classrooms"
)
st.write(
    "* **Employment:** Enabling equal participation in the job market by facilitating "
    "communication between employers and NSL users."
)
st.write(
    "* **Healthcare:** Enhancing access to healthcare services by eliminating communication "
    "barriers between medical professionals and NSL patients."
)
st.write(
    "* **Social Interaction:** Encouraging broader social integration and fostering "
    "stronger connections within the community."
)

st.divider()

# Introduction section
st.subheader("intro")
st.markdown("**<u> INTRODUCTION </u>**", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: justify;'><p style='text-indent: 2em;'>"
    "Imagine a world where communication transcends spoken words, where understanding blooms not "
    "just from sound, but from the graceful movement of hands. This world, though often unseen, "
    "exists within the vibrant tapestry of the Nigerian Sign Language (NSL) community. Yet, for many, "
    "the bridge connecting their world to the broader society remains frustratingly out of reach. "
    "This project seeks to dismantle those barriers, brick by digital brick. We present a revolutionary "
    "Automatic Sign Language Translation (ASLT) application, a testament to the transformative power of technology. "
    "This application is not just software; its a bridge, meticulously crafted to empower the NSL community "
    "and foster a more inclusive Nigeria. Within its framework lies the potential to unlock doors that were once "
    "firmly shut. Imagine classrooms where deaf students actively participate, workplaces where communication "
    "flows freely, and hospitals where healthcare is accessible to all. This, and much more, is the future "
    "we strive to build a future where NSL is not just a language, but a key that unlocks a world of opportunity. "
    "</p></div>",
    unsafe_allow_html=True,
)

# Render slideshow HTML component
st.components.v1.html(SLIDES_HTML_CODE, height=600)

# Data collection and training section
st.markdown("**<u> DATA COLLECTION AND TRAINING </u>**", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: justify;'><p style='text-indent: 2em;'>"
    "The cornerstone of any successful AI application is data. For our Automatic Sign Language "
    "Translation (ASLT) application, this data is the lifeblood that allows it to learn, "
    "interpret, and translate NSL signs accurately. However, collecting data for sign language "
    "translation presents unique challenges, requiring a multifaceted approach: </p></div>",
    unsafe_allow_html=True,
)

# Display bullet points for data collection and training
st.markdown(
    "**1. Leveraging Existing Resources:** "
    "We begin by harnessing the power of existing, publicly available datasets, "
    f"such as the [Google American Sign Language]({GOOGLE_ASL_URL})(ASL) and How2Sign datasets. "
    "While these datasets focus on American Sign Language (ASL), they serve as a valuable "
    "foundation due to the close linguistic relationship between ASL and NSL. Additionally, "
    f"we incorporate the [**IròyìnSpeech**]({IROYIN_SPEECH_URL}) dataset, a rich collection of spoken Yoruba speech, "
    "to enhance the naturalness of the final translation output.",
    unsafe_allow_html=True,
)

st.markdown(
    "**2. Community Collaboration:** "
    "Recognizing the importance of inclusivity and the unique nuances of NSL, "
    "we actively engage with the NSL community in Nigeria. Through partnerships with "
    "NSL organizations and educational institutions, we will gather data directly from "
    "native NSL signers. This collaboration ensures that our application captures the authentic "
    "expressions and variations present within the NSL community."
)

st.markdown(
    "**3. Continuous Learning and Refinement:** "
    "Data collection is not a one-time event. As the application evolves and encounters "
    "new signs and contexts, we remain committed to continuous learning. We will establish "
    "mechanisms for ongoing data collection and feedback from the NSL community, allowing the "
    "application to continuously improve its accuracy and adaptability."
)

st.write(
    "By combining these diverse approaches, we aim to build a robust and comprehensive data "
    "foundation that empowers our ASLT application to become a valuable tool for the NSL "
    "community in Nigeria."
)

# Display loading animation and button to proceed to the next page
loading_animation_ = load_lottiefile(LOADING_ANIMATION)
if st.button(
    ":blue[CLICK HERE TO PROCEED]", use_container_width=True, type="secondary"
):
    st_lottie(loading_animation_, height=300, width=300)
    time.sleep(5)  # Artificial delay for demonstration purposes
    st.switch_page(page=UPLOAD_PAGE)

# Footer
st.markdown(FOOTER_STYLE, unsafe_allow_html=True)
st.markdown(FOOTER, unsafe_allow_html=True)
