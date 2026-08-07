"""Backward-compatible imports for the Qwen3-based ONCE-DIRE model."""

from models.ONCE_DIRE_QWEN3_NAML import (
    ONCEDIREQwen3NewsEncoder,
    ONCE_DIRE_QWEN3_NAML,
)


ONCEDIRELlama1NewsEncoder = ONCEDIREQwen3NewsEncoder
ONCE_DIRE_LLAMA1_NAML = ONCE_DIRE_QWEN3_NAML

__all__ = [
    "ONCEDIRELlama1NewsEncoder",
    "ONCEDIREQwen3NewsEncoder",
    "ONCE_DIRE_LLAMA1_NAML",
    "ONCE_DIRE_QWEN3_NAML",
]
