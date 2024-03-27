"""
Ray serve for deployment
"""

from typing import Any
import wandb
from ray import serve


@serve.deployment
class SIGN2TEXT:
    """_summary_"""

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


@serve.deployment
class YB2AUDIO:
    """_summary_"""

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
