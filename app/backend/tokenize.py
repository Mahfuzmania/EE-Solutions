from __future__ import annotations

import re
from typing import List

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]
