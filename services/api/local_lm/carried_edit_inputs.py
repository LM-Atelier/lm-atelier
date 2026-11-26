"""What an edit made again on the same pictures keeps from the turn it starts from.

A regenerate, a branch and an edited turn rebuild the source turn's settings
against the fields its workflow offers, and a key with no field behind it is
dropped. A selection and a relight request are not workflow fields, so the
rebuild drops both, and these routes give them back to a turn on the same
pictures. Made again without its selection, an edit changes the whole
picture instead of the part that was selected; without its relight request, it
stores the raw relit result instead of the one finished against its source.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .studio_masks import MASK_SETTING_KEY
from .studio_relight import RELIGHT_SETTING_KEY


def carried_edit_inputs(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a turn's selection and relight request for a later turn on the same pictures.

    Both belong to the pictures they were made for, so a turn given other
    pictures takes neither: a selection is drawn on its picture, and a light
    map is drawn to match it.
    """
    return {
        key: copy.deepcopy(settings[key])
        for key in (MASK_SETTING_KEY, RELIGHT_SETTING_KEY)
        if key in settings
    }
