"""Tool declaration, registration, and schema generation for MitoCode.

Converts Python type annotations into JSON Schema tool definitions and
keeps separate human/agent-facing documentation per registered tool.

A caller may also supply the schema outright, which is what tools
discovered on an MCP server need: their arguments are described by the
server, not by anything introspectable in this process.
"""

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Union,
    get_args,
    get_origin,
    Literal,
)

from loguru import logger as loguru_logger

try:
    import types

    UnionType = getattr(types, "UnionType", None)
except ImportError:
    types = None
    UnionType = Union

from mitocode.backend.exception.mitocode_exception import MitoCodeToolDefinitionError

if TYPE_CHECKING:
    # Imported for type checking only.  The runtime stays on duck typing so
    # this module keeps importing in an environment without fastmcp installed.
    from mitocode.backend.mcp.mcp_client import McpClient

logger = loguru_logger.bind(name=__name__)

#: Prefix every namespaced MCP tool name carries, so a remote tool is
#: recognisably remote and cannot be confused with a built-in one.
MCP_TOOL_PREFIX = "mcp__"

#: Separator between the server name and the tool name inside that prefix.
MCP_TOOL_SEPARATOR = "__"



def _normalize_input_schema(raw: Any) -> Dict[str, Any]:
    """Coerce an MCP tool's advertised input schema into a usable JSON Schema.

    Servers are not required to send a complete schema.  This fills in the
    pieces :meth:`ToolManager.tool_schemas` always needs so a malformed
    schema cannot poison an entire LLM request.

    Args:
        raw: The ``input_schema`` value advertised by the MCP server.

    Returns:
        A dict with at least ``type`` and ``properties`` keys.
    """
    if not isinstance(raw, dict):
        return {"type": "object", "properties": {}}
    normalized = dict(raw)
    normalized.setdefault("type", "object")
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    return normalized


def _make_invoke(client: "McpClient", upstream_name: str) -> Callable:
    """Build the async callable that dispatches one MCP tool.

    The client and the upstream tool name are captured in the closure, so the
    generated callable takes ``**kwargs`` and nothing else.  Two earlier shapes
    were wrong and must not come back:

    * Binding them as default arguments — ``async def _invoke(_client=client,
      _tool_name=tool.name, **kwargs)`` — puts them in the *same* namespace as
      the model's keyword arguments.  A server tool advertising an argument
      literally named ``_client`` or ``_tool_name`` then binds over the captured
      values, and ``_tool_name`` in particular silently redirects the call to a
      different tool on that server, bypassing the name-collision guard that
      ``register_mcp_client`` relies on.
    * Closing over the loop variables directly would route every tool registered
      from one catalogue to whichever tool happened to come last.

    A closure factory gives the isolation the default arguments were reaching
    for without exposing either value as a keyword.

    Args:
        client: The connected MCP client to dispatch through.
        upstream_name: The tool's name *on the server*, which is not the
            namespaced name MitoCode registers it under.

    Returns:
        An async callable forwarding its keyword arguments to that one tool.
    """

    async def _invoke(**kwargs):
        """Forward the model's keyword arguments to the server tool."""
        return await client.call_tool(upstream_name, kwargs)

    return _invoke


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

    def tool(  # pylint: disable=too-many-arguments
        self,
        name: Optional[str] = None,
        *,
        human_doc: Optional[str] = None,
        agent_doc: Optional[str] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
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
            parameters_schema: JSON Schema of accepted arguments.
                Defaults to one derived from the function's type
                annotations.
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
                parameters_schema=parameters_schema,
                display_action=display_action,
                display_param=display_param,
            )
            return func

        return _decorator

    def register(  # pylint: disable=too-many-arguments
        self,
        func: Callable,
        *,
        name: Optional[str] = None,
        human_doc: Optional[str] = None,
        agent_doc: Optional[str] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
        display_action: Optional[str] = None,
        display_param: Optional[str] = None,
    ) -> ToolSpec:
        """Build a :class:`ToolSpec` from *func* and store it under its name.

        Explicit arguments take precedence over metadata attached by
        :func:`declare_tool`, which in turn takes precedence over the
        function's own name and docstring.

        ``parameters_schema`` is needed for arguments that cannot be read off
        a signature — an MCP server advertises its own JSON Schema — so when it
        is supplied it is used verbatim instead of being derived.  Annotation
        inference is the last resort.  Between the two sits a third source, a
        ``parameters_schema`` key in the ``declare_tool`` metadata, which
        :func:`declare_tool` publishes no argument for: only a caller writing
        ``func.__mitocode_tool_meta__["parameters_schema"]`` directly can set
        it.  It is honoured when present and non-``None``, like the explicit
        argument — an empty dict is a schema that declares no arguments, not an
        absent one — but it is not part of the public surface.

        Returns:
            The registered :class:`ToolSpec`.
        """
        meta = getattr(func, "__mitocode_tool_meta__", {})
        if parameters_schema is None:
            # An empty dict is a schema, not an absence of one, in this tier too.
            parameters_schema = meta.get("parameters_schema")
            if parameters_schema is None:
                parameters_schema = _build_parameters_schema(func)

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
            parameters_schema=parameters_schema,
            display_action=display,
            display_param=param,
        )
        self._tools[tool_name] = spec
        return spec

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            name: The tool name to remove.

        Returns:
            True if a tool was registered under *name* and has been removed,
            False if no such tool existed.
        """
        return self._tools.pop(name, None) is not None

    @staticmethod
    def build_mcp_tool_name(server_name: str, tool_name: str) -> str:
        """Compose the namespaced MitoCode name for an MCP tool.

        Args:
            server_name: The MCP server's configured name.
            tool_name: The tool's name as advertised by that server.

        Returns:
            ``mcp__<server_name>__<tool_name>``.
        """
        return f"{MCP_TOOL_PREFIX}{server_name}{MCP_TOOL_SEPARATOR}{tool_name}"

    def register_mcp_client(self, client: "McpClient") -> List[ToolSpec]:
        """Register every tool an MCP client has discovered.

        The client must already be connected so that its cached tool catalogue
        is populated.  A tool whose composed name is already taken is skipped
        with a warning rather than overwriting the existing registration.

        Registration is all-or-nothing.  Every tool is read, normalised and
        composed first, and ``self._tools`` is not touched until the whole
        catalogue has survived that pass, because a caller can only ever
        reclaim what it was told it registered: a catalogue whose later entry
        is malformed would otherwise leave its healthy predecessors published,
        unreported, and therefore removable by nobody.

        Publishing afterwards cannot itself fail for these inputs -- the name
        is an f-string result, so it is always hashable, and the docs and
        schema were normalised in the first pass -- which is what makes the
        single pass sufficient rather than a prefix of a wider one.  If
        :meth:`register` ever grows a step that can raise, this needs a
        rollback as well.

        Args:
            client: A connected MCP client exposing ``server_name`` and ``tools``.

        Returns:
            The specs that were successfully registered, in discovery order.
        """
        prepared: List[Dict[str, Any]] = []
        claimed: Set[str] = set()
        for tool in client.tools:
            tool_name = self.build_mcp_tool_name(client.server_name, tool.name)
            if tool_name in claimed or self.get_spec(tool_name) is not None:
                logger.warning(
                    f"Skipping MCP tool '{tool_name}': name already registered"
                )
                continue
            # Claimed as well as taken, so two tools in one catalogue composing
            # the same name resolve exactly as they did when each was published
            # as it was read: the first wins and the second is skipped.
            claimed.add(tool_name)


            description = (tool.description or "").strip()
            if not description:
                description = (
                    f"Tool '{tool.name}' provided by MCP server "
                    f"'{client.server_name}'."
                )

            prepared.append(
                {
                    "func": _make_invoke(client, tool.name),
                    "name": tool_name,
                    "human_doc": f"[{client.server_name}] {description}",
                    "agent_doc": f"[{client.server_name}] {description}",
                    "parameters_schema": _normalize_input_schema(tool.input_schema),
                    "display_action": f"mcp:{client.server_name}",
                }
            )

        registered: List[ToolSpec] = []
        for pending in prepared:
            registered.append(self.register(**pending))
        return registered

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
