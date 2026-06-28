import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin, Literal


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    if annotation is inspect._empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"anyOf": [_annotation_to_schema(a) for a in non_none]}

    if origin is Literal:
        enum_values = list(args)
        if enum_values:
            sample = enum_values[0]
            schema = _annotation_to_schema(type(sample))
            schema["enum"] = enum_values
            return schema
        return {"type": "string"}

    if origin in (list, tuple, set):
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}

    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
    }
    return mapping.get(annotation, {"type": "string"})


def _build_parameters_schema(func: Callable) -> Dict[str, Any]:
    signature = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
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


class ToolManager:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def tool(
            self,
            name: Optional[str] = None,
            *,
            human_doc: Optional[str] = None,
            agent_doc: Optional[str] = None,
    ):
        def _decorator(func: Callable):
            self.register(func, name=name, human_doc=human_doc, agent_doc=agent_doc)
            return func

        return _decorator

    def register(
            self,
            func: Callable,
            *,
            name: Optional[str] = None,
            human_doc: Optional[str] = None,
            agent_doc: Optional[str] = None,
    ) -> ToolSpec:
        meta = getattr(func, "__mitocode_tool_meta__", {})

        tool_name = name or meta.get("name") or func.__name__
        human_description = human_doc or meta.get("human_doc") or inspect.getdoc(func) or ""
        agent_description = agent_doc or meta.get("agent_doc") or human_description

        spec = ToolSpec(
            name=tool_name,
            func=func,
            human_doc=human_description,
            agent_doc=agent_description,
            parameters_schema=_build_parameters_schema(func),
        )
        self._tools[tool_name] = spec
        return spec

    def tools_map(self) -> Dict[str, Callable]:
        return {name: spec.func for name, spec in self._tools.items()}

    def tool_schemas(self, audience: str = "agent") -> List[Dict[str, Any]]:
        use_agent_doc = audience != "human"
        schemas = []
        for spec in self._tools.values():
            description = spec.agent_doc if use_agent_doc else spec.human_doc
            schemas.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": description,
                    "parameters": spec.parameters_schema,
                },
            })
        return schemas

    def human_docs(self) -> Dict[str, str]:
        return {name: spec.human_doc for name, spec in self._tools.items()}

    def agent_docs(self) -> Dict[str, str]:
        return {name: spec.agent_doc for name, spec in self._tools.items()}
