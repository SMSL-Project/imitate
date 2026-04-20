from __future__ import annotations

import json
import os
from typing import Any


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def write_json(path: str, content: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2)
