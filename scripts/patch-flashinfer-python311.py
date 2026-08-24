#!/usr/bin/env python3
"""Apply the upstream FlashInfer Python 3.10/3.11 annotation compatibility fix."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


if sys.version_info >= (3, 12):
    print("FlashInfer compatibility patch not needed on Python 3.12+.")
    raise SystemExit(0)

try:
    distribution = importlib.metadata.distribution("flashinfer-python")
except importlib.metadata.PackageNotFoundError:
    print("flashinfer-python is not installed; nothing to patch.")
    raise SystemExit(0)

target = Path(distribution.locate_file("flashinfer/comm/fd_exchange.py"))
if not target.is_file():
    raise SystemExit(f"FlashInfer compatibility target not found: {target}")

source = target.read_text(encoding="utf-8")
future_import = "from __future__ import annotations"
if future_import in source:
    print(f"FlashInfer Python 3.11 compatibility already present: {target}")
    raise SystemExit(0)

marker = "\"\"\"\n\nimport array"
replacement = "\"\"\"\n\nfrom __future__ import annotations\n\nimport array"
if marker not in source:
    raise SystemExit(
        "FlashInfer fd_exchange.py has an unexpected layout; refusing to patch it."
    )

target.write_text(source.replace(marker, replacement, 1), encoding="utf-8")
print(f"Applied FlashInfer Python 3.11 compatibility patch: {target}")
