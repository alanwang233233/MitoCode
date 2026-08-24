"""Tool declaration, registration, and schema generation for MitoCode.

Converts Python type annotations into JSON Schema tool definitions and
keeps separate human/agent-facing documentation per registered tool.
"""

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    Literal,
)

try:
    import types

    UnionType = getattr(types, "UnionType", None)
except ImportError:
    types = None
    UnionType = Union

from mitocode.backend.exception.mitocode_exception import MitoCodeToolDefinitionError


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment.

    Returns a **new** dict on every call so that callers may safely
    augment the result (e.g. with ``default`` or ``enum``) without
    mutating a shared cache entry.

    Args:
        annotation: A Python type annotation (e.g. ``str``, ``int``,
            ``Optional[str]``, ``List[int]``).  Use ``inspect._empty``
            to represent an un-annotated parameter.

    Returns:
        A JSON Schema fragment dict describing the annotated type.
    """
    if annotation is inspect._empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union or origin is UnionType:
        raw = [a for a in args]
        if len(raw) == 1:
            return _annotation_to_schema(raw[0])
        return {"anyOf": [_annotation_to_schema(a) for a in raw]}

    if origin is Literal:
        enum_values = list(args)
        if enum_values:
            sample = enum_values[0]
            schema = _annotation_to_schema(type(sample))
            schema["enum"] = enum_values
            return schema
        return {"type": "string"}

    if origin in (list, tuple, set):
        schema: Dict[str, Any] = {
            "type": "array",
        }
        if args:
            schema["items"] = _annotation_to_schema(args[0])
        return schema

    if origin is dict:
        schema: Dict[str, Any] = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _annotation_to_schema(args[1])
        return schema

    _BASE_TYPE_MAP = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
        None: {"type": "null"},
        type(None): {"type": "null"},
    }
    try:
        schema = _BASE_TYPE_MAP[annotation]
    except KeyError as e:
        raise MitoCodeToolDefinitionError(
            f"Unsupported type annotation: {annotation}. Supported types: str, int, float, bool, dict, list"
        ) from e
    return dict(schema)


def _build_parameters_schema(func: Callable) -> Dict[str, Any]:
    """Build a JSON Schema ``object`` describing *func*'s keyword arguments.

    Positional-only parameters are rejected because LLMs always invoke
    tools via keyword arguments.  ``*args`` and ``**kwargs`` are silently
    skipped.

    Args:
        func: The callable whose parameters will be introspected.

    Returns:
        A JSON Schema ``{"type": "object", "properties": ..., "required": ...}``
        dict describing the accepted keyword arguments.

    Raises:
        MitoCodeToolDefinitionError: If *func* contains positional-only
            parameters or an unsupported type annotation.
    """
    signature = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            func_name = getattr(func, "__name__", type(func).__name__)
            raise MitoCodeToolDefinitionError(
                f"Tool function '{func_name}' contains positional-only parameter '{name}'. "
                "Positional-only parameters are not supported because LLMs invoke tools using keyword arguments."
            )
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        param_schema = _annotation_to_schema(parameter.annotation)
        if parameter.default is not inspect._empty:
            param_schema["default"] = parameter.default
        else:
            required.append(name)
        properties[name] = param_schema

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def declare_tool(
    name: Optional[str] = None,
    *,
    human_doc: Optional[str] = None,
    agent_doc: Optional[str] = None,
):
    """Attach MitoCode tool metadata to a function without registering it.

    The decorated function can later be registered with
    :meth:`ToolManager.register`, which reads the attached metadata as
    defaults. ``name`` defaults to the function's own name.
    """

    def _decorator(func: Callable):
        setattr(
            func,
            "__mitocode_tool_meta__",
            {
                "name": name or func.__name__,
                "human_doc": human_doc,
                "agent_doc": agent_doc,
            },
        )
        return func

    return _decorator


@dataclass
class ToolSpec:
    """Registered tool metadata: callable, docs, JSON schema, display hints.

    Attributes:
        name: Unique tool name used in schemas and dispatch.
        func: The callable implementing the tool.
        human_doc: Description targeted at human-facing UI.
        agent_doc: Description sent to the LLM in tool schemas.
        parameters_schema: JSON Schema of accepted keyword arguments.
        display_action: Optional action key used by the TUI for styling.
        display_param: Parameter whose value is highlighted in the TUI.
    """

    name: str
    func: Callable
    human_doc: str
    agent_doc: str
    parameters_schema: Dict[str, Any]
    display_action: str = ""
    display_param: str = ""


class ToolManager:
    """Registry that stores tool specs and exposes schemas and docs.

    Tools are registered via the :meth:`tool` decorator (or
    :meth:`register` directly). The manager derives each tool's JSON
    Schema from its type annotations and keeps separate human/agent
    descriptions for schema generation.
    """

    def __init__(self):
        """Initialise an empty tool registry.

        Creates an internal mapping from tool names to :class:`ToolSpec`
        instances.  Use :meth:`tool` (decorator) or :meth:`register`
        to populate the registry.
        """
        self._tools: Dict[str, ToolSpec] = {}

    def tool(
        self,
        name: Optional[str] = None,
        *,
        human_doc: Optional[str] = None,
        agent_doc: Optional[str] = None,
        display_action: Optional[str] = None,
        display_param: Optional[str] = None,
    ):
        """Decorator that registers the wrapped function as a tool.

        Args:
            name: Tool name used in schemas and dispatch.  Defaults to
                the wrapped function's ``__name__``.
            human_doc: Human-readable description for UI purposes.
            agent_doc: Description sent to the LLM in tool schemas.
                Defaults to *human_doc* if not provided.
            display_action: Optional action key used by the TUI for
                styling the tool invocation.
            display_param: Parameter whose value should be highlighted
                in the TUI.

        Returns:
            A decorator that registers the function and returns it unchanged.
        """

        def _decorator(func: Callable):
            self.register(
                func,
                name=name,
                human_doc=human_doc,
                agent_doc=agent_doc,
                display_action=display_action,
                display_param=display_param,
            )
            return func

        return _decorator

    def register(
        self,
        func: Callable,
        *,
        name: Optional[str] = None,
        human_doc: Optional[str] = None,
        agent_doc: Optional[str] = None,
        display_action: Optional[str] = None,
        display_param: Optional[str] = None,
    ) -> ToolSpec:
        """Build a :class:`ToolSpec` from *func* and store it under its name.

        Explicit arguments take precedence over metadata attached by
        :func:`declare_tool`, which in turn takes precedence over the
        function's own name and docstring.

        Returns:
            The registered :class:`ToolSpec`.
        """
        meta = getattr(func, "__mitocode_tool_meta__", {})

        tool_name = name or meta.get("name") or func.__name__
        human_description = (
            human_doc or meta.get("human_doc") or inspect.getdoc(func) or ""
        )
        agent_description = agent_doc or meta.get("agent_doc") or human_description
        display = display_action or meta.get("display_action") or ""
        param = display_param or meta.get("display_param") or ""

        spec = ToolSpec(
            name=tool_name,
            func=func,
            human_doc=human_description,
            agent_doc=agent_description,
            parameters_schema=_build_parameters_schema(func),
            display_action=display,
            display_param=param,
        )
        self._tools[tool_name] = spec
        return spec

    def tools_map(self) -> Dict[str, Callable]:
        """Return a name-to-callable mapping of all registered tools."""
        return {name: spec.func for name, spec in self._tools.items()}

    def get_spec(self, name: str) -> ToolSpec | None:
        """Return the spec registered under *name*, or ``None`` if absent."""
        return self._tools.get(name)

    def tool_schemas(self, audience: str = "agent") -> List[Dict[str, Any]]:
        """Build OpenAI function-calling schemas for all registered tools.

        Args:
            audience: ``"agent"`` uses agent-facing descriptions (default);
                any other value uses human-facing descriptions.
        """
        use_agent_doc = audience != "human"
        schemas = []
        for spec in self._tools.values():
            description = spec.agent_doc if use_agent_doc else spec.human_doc

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": description,
                        "parameters": spec.parameters_schema,
                    },
                }
            )
        return schemas

    def human_docs(self) -> Dict[str, str]:
        """Return a mapping of tool names to human-readable documentation."""
        return {name: spec.human_doc for name, spec in self._tools.items()}

    def agent_docs(self) -> Dict[str, str]:
        """Return a mapping of tool names to agent-facing documentation."""
        return {name: spec.agent_doc for name, spec in self._tools.items()}
