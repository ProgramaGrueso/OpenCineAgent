"""Deterministic seeds. Python's hash() is salted per process, so it cannot make renders reproducible."""

from __future__ import annotations

import hashlib


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)
