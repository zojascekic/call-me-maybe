from typing import Literal

from pydantic import BaseModel


class ParameterSpec(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]


class ReturnSpec(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec


class PromptItem(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, float | int | str | bool]
