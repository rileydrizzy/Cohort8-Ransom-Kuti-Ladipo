"""doc
"""

import setuptools

with open("README.md", "r", encoding="UTF-8") as f:
    long_description = f.read()


__version__ = "0.1.0"

REPO_NAME = "NSL_2_Audio"
AUTHOR_USER_NAME = "rileydrizzy"
SRC_REPO = "src"
AUTHOR_EMAIL = "ipadeolaoladipo@outlook.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    license="MIT",
    author=AUTHOR_USER_NAME,
    description="A Machine Learning System for Sign Language Video Translation\
        and Speech Generation into Low-Resource Languages (LRLs) in Nigeria",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
