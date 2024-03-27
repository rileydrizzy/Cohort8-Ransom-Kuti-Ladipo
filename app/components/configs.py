"""
This module contains configuration settings for the NSL-2-AUDIO application.

Variables:
- GITHUB_REPO_URL (str): The URL of the GitHub repository for the project.
- GOOGLE_ASL_URL (str): The URL of the Google American Sign Language dataset.
- IROYIN_SPEECH_URL (str): The URL of the IròyìnSpeech dataset.
- PAGES_DIR (str): The directory containing page modules.
- UPLOAD_PAGE (str): The path to the upload page module.
- RESULT_PAGE (str): The path to the result page module.
- PROJECT_LOGO (str): The path to the project logo image file.
- NSL_SIGN_GIF (str): The path to the NSL sign GIF image file.
- SLIDES_IMAGES (list): List of URLs for slideshow images.
- LOADING_ANIMATION (str): The path to the loading animation Lottie file.
- SLIDES_HTML_CODE (str): HTML code for the slideshow component.
- FOOTER_STYLE (str): CSS style for hiding Streamlit footer.
- FOOTER (str): HTML code for the footer section.

Usage:
Import this module to access configuration variables and URLs used in the NSL-2-AUDIO application.

"""

# URLs
GITHUB_REPO_URL = "https://github.com/rileydrizzy/NSL_2_AUDIO"
GOOGLE_ASL_URL = "https://www.kaggle.com/competitions/asl-fingerspelling/data"
IROYIN_SPEECH_URL = "https://www.kaggle.com/datasets/rileydrizzy/iroyinspeech"

# Directories and paths
PAGES_DIR = "pages/"
UPLOAD_PAGE = PAGES_DIR + "upload_page.py"
RESULT_PAGE = PAGES_DIR + "result_page.py"
PROJECT_LOGO = "assets/images/logos/logo.png"
NSL_SIGN_GIF = "assets/sign_lang.gif"
LOADING_ANIMATION = "assets/lottie/Animation_1708486960708.json"
PROCESSING_ANIMATION = "assets/lottie/Animation_1708486960708.json"

# Slideshow images
SLIDES_IMAGES = [
    "https://unsplash.com/photos/GJ8ZQV7eGmU/download?force=true&w=1920",
    "app/streamlit_app/assets/images/sign_image_1.jpg",
    "assets/images/sign_image_2.jpg",
]

# HTML code for slideshow component
SLIDES_HTML_CODE = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {{
    box-sizing: border-box;
}}
body {{
    font-family: Verdana, sans-serif;
}}
.mySlides {{
    display: none;
}}
img {{
    vertical-align: middle;
}}
/* Slideshow container */
.slideshow-container {{
    max-width: 1000px;
    position: relative;
    margin: auto;
}}
/* Caption text */
.text {{
    color: #f2f2f2;
    font-size: 15px;
    padding: 8px 12px;
    position: absolute;
    bottom: 8px;
    width: 100%;
    text-align: center;
}}
/* Number text (1/3 etc) */
.numbertext {{
    color: #f2f2f2;
    font-size: 12px;
    padding: 8px 12px;
    position: absolute;
    top: 0;
}}
/* The dots/bullets/indicators */
.dot {{
    height: 15px;
    width: 15px;
    margin: 0 2px;
    background-color: #bbb;
    border-radius: 50%;
    display: inline-block;
    transition: background-color 0.6s ease;
}}
.active {{
    background-color: #717171;
}}
/* Fading animation */
.fade {{
    animation-name: fade;
    animation-duration: 1.5s;
}}
@keyframes fade {{
    from {{opacity: .4}} 
    to {{opacity: 1}}
}}
/* On smaller screens, decrease text size */
@media only screen and (max-width: 300px) {{
    .text {{font-size: 11px}}
}}
</style>
</head>
<body>
<h2>Automatic Slideshow</h2>
<p>Change image every 2 seconds:</p>
<div class="slideshow-container">
"""

# Add slides for each image path
for i, path in enumerate(SLIDES_IMAGES, 1):
    SLIDES_HTML_CODE += f"""
<div class="mySlides fade">
    <div class="numbertext">{i} / {len(SLIDES_IMAGES)}</div>
    <img src="{path}" style="width:100%">
    <div class="text">Caption Text</div>
</div

>
"""

# Add dots for navigation
SLIDES_HTML_CODE += """
</div>
<br>
<div style="text-align:center">
"""

for i in range(len(SLIDES_IMAGES)):
    SLIDES_HTML_CODE += "<span class='dot'></span>"

SLIDES_HTML_CODE += """
</div>
<script>
let slideIndex = 0;
showSlides();

function showSlides() {{
    let i;
    let slides = document.getElementsByClassName("mySlides");
    let dots = document.getElementsByClassName("dot");
    for (i = 0; i < slides.length; i++) {{
        slides[i].style.display = "none";  
    }}
    slideIndex++;
    if (slideIndex > slides.length) {{slideIndex = 1}}   
    for (i = 0; i < dots.length; i++) {{
        dots[i].className = dots[i].className.replace(" active", "");
    }}
    slides[slideIndex-1].style.display = "block";  
    dots[slideIndex-1].className += " active";
    setTimeout(showSlides, 2000); // Change image every 2 seconds
}}
</script>
</body>
</html>
"""

# CSS style for hiding Streamlit footer
FOOTER_STYLE = """
<style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after { 
        visibility: visible;
        display: block;
        position: relative;
        # background-color: red;
        padding: 5px;
        top: 2px;
    }
</style>
"""

# HTML code for the footer section
FOOTER = """
<style>
img {
    max-width: 100%;
}

.headerStyle {
    transition: transform .2s;
}

.headerStyle:hover {
    transform: scale(1.5);
    transition: 0.2s;
}
.image1 { 
    padding: 10px;
    transition: transform .2s;
}
.image2 { 
    padding: 10px;
    transition: transform .2s;
}
.image1:hover {  
    transform: scale(1.5);
    transition: 0.2s;
}
.image2:hover {  
    transform: scale(1.5);
    transition: 0.2s;
}

a:link,
a:visited {
    color: blue;
    background-color: transparent;
    text-decoration: underline;
}

a2:hover {
    border-style: solid;
},
a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
}

.footer {
    position: relative;
    width: 100%;
    left: 0;
    bottom: 0;
    background-color: transparent;
    margin-top: auto;
    color: #CD5C5C;
    padding: 24px;
    text-align: center;
}
</style>
<div class="footer">
    <p style="font-size: 13px">Copyright (c) 2024 Ladipo Ezekiel Ipadeola.</p>
    <p style="font-size: 13px">This software is distributed under an M!T license. Please consult the LICENSE file for more details.</p>
    <a href="https://github.com/rileydrizzy/NSL_2_AUDIO"><img class="image2" src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="github" width="70" height="70"></a>
</div>
"""
