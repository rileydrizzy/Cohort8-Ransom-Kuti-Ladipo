"""
This module provides utility functions for the NSL-2-AUDIO  web application.

Functions:
- `load_lottiefile(filepath: str) -> dict`: Loads a Lottie animation file (.json) \
    and returns its parsed contents as a dictionary.

Dependencies:
- json

Usage:
Import this module and use the provided functions as needed.

"""

import json


def load_lottiefile(filepath: str) -> dict:
    """Loads a Lottie animation file (.json) and returns its parsed contents as a dictionary.

    Parameters
    ----------
    filepath : str
        Path to the Lottie animation file.

    Returns
    -------
    dict
        The parsed contents of the Lottie animation file.

    Raises
    -------
    FileNotFoundError
        If the specified file is not found.
    """

    with open(filepath, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Lottie animation file not found: {filepath}"
            ) from error
