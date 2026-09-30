"""The two ways a source picture is fitted to its canvas, as one recipe type.

An extension keeps the whole source and has the workflow make the rest of the
canvas; a crop keeps the middle of the source that has the canvas's shape and
uploads it at the canvas size. Both are recorded in the accepted context and
told apart there by their mode.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .source_crop_recipe import SourceCropRecipe
from .source_fit_recipe import SourceExtensionRecipe

SourceFitRecipe = SourceExtensionRecipe | SourceCropRecipe

#: The recorded form, read back by its mode.
RecordedSourceFitRecipe = Annotated[SourceFitRecipe, Field(discriminator="mode")]
