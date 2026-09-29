"""Derive LoRA intent from bounded descriptive provider metadata."""

from .profile_use_cases import derive_profile_use_case, normalize_provider_use_case_metadata

_FORMAT_LABELS = frozenset(
    {"lora", "locon", "lycoris", "dora", "safetensors", "texttoimage", "diffusers"}
)


def derive_lora_use_case(metadata: object) -> str:
    """Keep intent separate from the architecture used for compatibility checks."""
    normalized = normalize_provider_use_case_metadata(metadata)
    descriptive = {key: words for key, words in normalized.items() if key != "base_model"}
    for key in ("tags", "category"):
        descriptive[key] = [
            word
            for word in descriptive.get(key, [])
            if "".join(character for character in word.casefold() if character.isalnum())
            not in _FORMAT_LABELS
        ]
    return derive_profile_use_case(descriptive)
