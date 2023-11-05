"""
NSL2AUDIO

A Machine Learning System for Sign Language Video Translation and Speech Generation 
into Low-Resource Languages (LRLs) in Nigeria
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="UTF-8") as f:
    long_description = f.read()


__version__ = "0.1.0"

REPO_NAME = "NSL2AUDIO"
AUTHOR_USER_NAME = "Ipadeola Ladipo"
AUTHOR_EMAIL = "ipadeolaoladipo@outlook.com"
PROJECT_URL = "https://github.com/AISaturdaysLagos/Cohort8-Ransom-Kuti-Ladipo"


setup(
    name=REPO_NAME,
    version=__version__,
    license="MIT",
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A Machine Learning System for Sign Language Video Translation\
        and Speech Generation into Low-Resource Languages (LRLs) in Nigeria",
    long_description=long_description,
    long_description_content="text/markdown",
    url=PROJECT_URL,
    project_urls={"Bug Tracker": f"{PROJECT_URL}/issues"},
    packages=find_packages(exclude=[]),
)
