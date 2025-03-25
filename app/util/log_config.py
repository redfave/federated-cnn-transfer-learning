import io
import logging
import sys
from typing import Literal


def configure_logging(mode: Literal["append", "overwrite"]) -> logging.Logger:
    if hasattr(__builtins__, "__IPYTHON__"):  # execution in Jupyter
        sys.stdout = io.TextIOWrapper(sys.stdout, encoding="utf-8")
    else:  # execution in CPython
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(threadName)s %(processName)s %(message)s",
        handlers=[
            logging.FileHandler(
                filename=f"veggie-net.log",
                mode="w" if mode == "overwrite" else "a",
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)
