from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """The type of one function parameter.

    Attributes:
        type: The JSON type of the parameter. ``number``, ``integer``
            and ``boolean`` are generated as such; any other type is
            generated as a string.
    """

    type: str


class ReturnSpec(BaseModel):
    """The return type of a function.

    Attributes:
        type: The JSON type of the return value.
    """

    type: str


class FunctionDefinition(BaseModel):
    """A function the model can call.

    Attributes:
        name: The function name.
        description: What the function does.
        parameters: Each parameter name mapped to its type.
        returns: The return type.
    """

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec


class PromptItem(BaseModel):
    """One request to translate into a function call.

    Attributes:
        prompt: The natural-language request.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """The function call generated for one prompt.

    Attributes:
        prompt: The original request.
        name: The name of the chosen function.
        parameters: Each parameter name mapped to its generated value.
    """

    prompt: str
    name: str
    parameters: dict[str, float | int | str | bool]
