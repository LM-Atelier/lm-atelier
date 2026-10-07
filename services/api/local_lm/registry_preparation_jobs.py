"""Read the accepted inputs of a standalone Registry preparation job."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator

from .schemas import ApiModel

NodeName = Annotated[str, Field(min_length=1, max_length=200)]


class RegistryWorkflowContext(ApiModel):
    workflow_revision_id: str = Field(min_length=1, max_length=40)
    required_node_types: list[NodeName] = Field(min_length=1)


class RegistryPreparationInputs(ApiModel):
    package_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)
    node_types: list[NodeName] = Field(min_length=1)
    renew_install_id: str | None = Field(default=None, min_length=1, max_length=64)
    authorized_workflow: RegistryWorkflowContext | None = None

    @field_validator("node_types")
    @classmethod
    def validate_node_types(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values) or any(
            value != value.strip()
            or any(character < " " or character == "\x7f" for character in value)
            for value in values
        ):
            raise ValueError("Accepted Registry node types are invalid.")
        return values
