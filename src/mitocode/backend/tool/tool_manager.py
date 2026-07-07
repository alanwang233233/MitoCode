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
    name: str
    func: Callable
    human_doc: str
    agent_doc: str
    parameters_schema: Dict[str, Any]
    display_action: str = ""
    display_param: str = ""


class ToolManager:
    def __init__(self):
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
        return {name: spec.func for name, spec in self._tools.items()}

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def tool_schemas(self, audience: str = "agent") -> List[Dict[str, Any]]:
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
        return {name: spec.human_doc for name, spec in self._tools.items()}

    def agent_docs(self) -> Dict[str, str]:
        return {name: spec.agent_doc for name, spec in self._tools.items()}
