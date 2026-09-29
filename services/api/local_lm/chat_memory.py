from __future__ import annotations

DEFAULT_CHAT_CONTEXT = 8192
MINIMUM_CHAT_CONTEXT_MEMORY = 512 * 1024**2
CHAT_CONTEXT_BYTES_PER_TOKEN = 128 * 1024
# The longest context the minimum memory already covers: every shorter one
# is estimated at the same cost.
CHAT_CONTEXT_AT_MINIMUM_MEMORY = MINIMUM_CHAT_CONTEXT_MEMORY // CHAT_CONTEXT_BYTES_PER_TOKEN


def estimated_chat_memory(model_bytes: int, context_length: int = DEFAULT_CHAT_CONTEXT) -> int:
    """Estimate model and context memory without claiming a measured peak."""

    return model_bytes + max(
        MINIMUM_CHAT_CONTEXT_MEMORY, context_length * CHAT_CONTEXT_BYTES_PER_TOKEN
    )
