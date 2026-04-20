from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


def extract_fenced_blocks(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    return [(lang.strip().lower(), body.strip()) for lang, body in pattern.findall(text)]


def extract_json_from_text(text: str) -> Dict[str, Any]:
    for _, block in extract_fenced_blocks(text):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise ValueError("No JSON object found in the model response.")


def extract_single_python_block(text: str) -> str:
    blocks = extract_fenced_blocks(text)
    if len(blocks) != 1:
        raise ValueError("Expected exactly one fenced python block, but found {} fenced blocks.".format(len(blocks)))

    language, body = blocks[0]
    if language not in {"", "python", "py"}:
        raise ValueError("Expected a python fenced block, but found `{}`.".format(language))
    return body
