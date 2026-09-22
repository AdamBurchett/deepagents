"""Prompt/rendering helpers for REPL and PTC system prompts."""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from typing import TYPE_CHECKING, Any, Literal, cast, get_type_hints

from langchain_core.utils.json_schema import dereference_refs
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

_CAMEL_SEP = re.compile(r"[-_]([a-z])")
_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_RESERVED_TYPE_NAMES = frozenset(
    {"Array", "Map", "Object", "Promise", "Record", "RegExp", "Set", "Symbol"}
)
_REPL_SYSTEM_PROMPT_TEMPLATE = (
    "### Interpreter\n\n"
    "{repl_intro_line}\n\n"
    "{state_persistence_line}\n"
    "- Top-level `await` works; Promises resolve before the call returns.\n"
    "- Runtime sandbox: no built-in filesystem, network, stdlib, or wall-clock "
    "APIs (`fetch`, `require`, `fs`, `process`, real `Date.now()` are "
    "unavailable or stubbed).\n"
    "{side_effects_line}\n"
    "- Timeout: {timeout}s per call. Memory: {memory_limit_mb} MB total.\n"
    "- `console.log` output is captured and returned alongside the result."
)
_SUBAGENT_SYSTEM_PROMPT_TEMPLATE = """

### Dispatching Subagents with `task`

`task` is your primitive for running configured subagents from inside the
JavaScript REPL. Your job here is to DISTRIBUTE work, not to do it yourself:
write JavaScript that fans work out to subagents and assembles their results.
You handle the orchestration - fan-out, filtering, deduplication, multi-stage
flow, and synthesis - in plain JavaScript.

#### The primitive

```javascript
await task({
  description,      // full autonomous task prompt
  subagentType,     // configured subagent name
  label,            // optional short UI label for this dispatch
  responseSchema,   // optional JSON Schema for structured output
}); // -> Promise<unknown>
```

`task` runs a full agentic loop for the selected configured subagent. The
subagent can use whatever tools it was configured with, iterate, inspect
context, and return one final result. `subagentType` is required; use one of
the configured subagent names.

`description` is the only prompt the subagent receives for this dispatch. Make
it complete: the goal, the constraints, what to inspect, and the exact shape
or level of detail you expect back. Give context as locators — file paths and
symbol names — not as pasted file contents. If you already read a file while
exploring, still pass its path and let the subagent read it; do not paste back
what you read. Each dispatch is stateless from the caller's perspective; you
cannot send follow-up messages to the same subagent run.

`label` is optional: when provided, it is shown in the live progress UI
instead of the default description-derived fallback. It is not sent to the
subagent and does not affect execution.

`responseSchema` is optional, but set it on any dispatch whose result feeds
later code. A deterministic, typed shape is what lets you compose the next
stage reliably — index it, sort it, compare fields, branch on it, merge it —
instead of parsing free-form text. This is what makes a whole workflow
composable as one script. When provided, the resolved value is already a typed
JavaScript value matching the schema; do not call `JSON.parse` unless the
subagent intentionally returned a JSON string. Dynamic schemas work for
declarative subagents; runnable-backed subagents reject dynamic schemas because
their runnable is already compiled.

#### Approval model

`task` dispatches from inside the already-running `{tool_name}` call. It
does not route through the parent agent's `ToolNode`-managed `task` tool and
does not trigger parent-level `interrupt_on` / HITL approval for each dispatch.
Declarative subagents still honor approval middleware configured inside their
own spec. If you need approval before launching a subagent from the parent, use
the normal `task` tool outside JavaScript or ensure the `{tool_name}` call
itself is approval-gated.

#### Mental model

Hold your work in JS: an array of items in, an array of results out. Merge each
dispatch result back onto its item. Multi-stage analysis means: run a pass,
filter or regroup the array in JS, then run another pass over the survivors.

You can run the whole workflow in one `{tool_name}` call or split it across
several — both are fine. A single end-to-end script (generate, compare, pick a
winner; or review every item, then synthesize) is clean when you can write it
in one go; splitting is also fine when you want to inspect results between
stages. Either way, don't redo work across calls — reuse what is already in
scope (see "Reuse what earlier evals left in scope" below).

#### Fan out with bounded concurrency

Dispatch independent work in parallel with `Promise.all`, but in explicit
batches around 10 so you do not launch hundreds of subagents at once. The bridge
enforces a hard per-REPL cap of 32 concurrent subagent calls.

```javascript
const files = ["/src/a.ts", "/src/b.ts", "/src/c.ts"]; // found while exploring
const batchSize = 10;
const reviewed = [];
for (let i = 0; i < files.length; i += batchSize) {
  const batch = files.slice(i, i + batchSize);
  reviewed.push(...(await Promise.all(batch.map(async (file) => {
    const result = await task({
      description: "Read " + file + " and review it for SQL injection. " +
        "Cite line numbers.",
      subagentType: "reviewer",
      responseSchema: {
        type: "object",
        properties: {
          vulnerabilities: {
            type: "array",
            items: {
              type: "object",
              properties: {
                type: { type: "string" },
                line: { type: "number" },
                evidence: { type: "string" },
              },
              required: ["type", "line", "evidence"],
            },
          },
        },
        required: ["vulnerabilities"],
      },
    });
    return { file, ...result };
  }))));
}
```

#### Explore with your own tools first, then distribute

You already have your normal tools for reading, listing, globbing, and
grepping files. Use them to explore and understand the task BEFORE you write
the orchestration script. These are ordinary tool calls, separate from the
`{tool_name}` tool: read the data file, list or glob the directory, grep for
what matters, then decide how to split the work.

Never write `{tool_name}` code that spawns a subagent just to read or parse a
file or list a directory. That is a deterministic step you do yourself with a
direct tool call; spending a whole agent loop on it is wasteful.

Once you understand the shape of the work, you have creative freedom in how
you split it:

- One dispatch per file or per record, when the items are already separate.
- Chunk a large input yourself — read it, split it, optionally write a small
  input file per chunk — and dispatch one subagent per chunk.
- A cheap classification pass first, then deeper dispatches only for the items
  that warrant them.

Then write JavaScript in the `{tool_name}` tool that distributes the heavy,
agentic work to subagents with `task()`: analyzing file contents, exploring a
codebase, making judgment calls, rewriting code, or synthesizing a report.

Hand each subagent a locator, not a payload. Subagents have their own file
tools, so for anything that lives in a file — a file to review, rewrite, or
audit — pass the path and let the subagent read it. Do NOT read a whole file
just to paste its contents into the description; that bloats every dispatch
and duplicates the file across them. Reserve inline content for small or
derived data that has no path of its own: a single parsed record, or a chunk
you split out of a larger input (write the chunk to its own file and pass that
path if it is large). Assemble the results in JS.

#### Compose multiple stages

Filter the array in JS between passes. For example: first ask subagents for a
cheap classification, filter to the risky items, then dispatch deeper reviews
only for those items.

```javascript
const tagged = await Promise.all(files.map((file) =>
  task({
    description: "Read " + file + " and classify it as handler, util, " +
      "test, or config.",
    subagentType: "reviewer",
    responseSchema: {
      type: "object",
      properties: { kind: { type: "string" }, risky: { type: "boolean" } },
      required: ["kind", "risky"],
    },
  }).then((tag) => ({ file, ...tag }))
));

const riskyHandlers = tagged.filter((it) => it.kind === "handler" && it.risky);
const deepReviews = await Promise.all(riskyHandlers.map((it) =>
  task({
    description: "Deep security review of " + it.file + ". Cite line numbers.",
    subagentType: "reviewer",
  }).then((review) => ({ ...it, review }))
));
```

#### Return results via the last expression, not `console.log`

The value of the last expression in an `{tool_name}` call (or a resolved
top-level `await`) is returned to you as the result. Make that final
expression the variable holding your result and read it from there.
`console.log` is only for incidental debugging: its output is capped and
truncated, while the returned value is not, so never `console.log` your
actual results.

Keep large intermediate sets in JS variables and return only a compact
summary or a small slice, not the entire dataset. To persist full output,
have a subagent write it, or write it with your own file tool outside the
`{tool_name}` call.

#### Reuse what earlier evals left in scope

The REPL is persistent within a turn: every top-level variable, function, and
class you declare is kept and is available in your next `{tool_name}` call
(each is hoisted to global scope). So if a later step needs something an
earlier eval produced or bound, **reference that variable by name** — do not
write a new literal that re-types data a previous eval already returned or
computed.

If you catch yourself pasting a big array or object of values you produced in
an earlier call, that is the tell: the variable is still in scope, so use it.
Re-typing prior results as a fresh literal wastes tokens and drifts from what
actually ran.

```javascript
// An earlier eval bound this:
//   const auditResults = await Promise.all(files.map(/* ...audit... */));

// A later eval — reference it; do NOT paste the findings back in as a literal:
const findings = auditResults.flatMap((r) =>
  r.findings.map((f) => ({ ...f, file: r.file }))
);
const verified = await Promise.all(findings.map((f) =>
  task({
    description: "Verify this finding: " + f.evidence,
    subagentType: "verifier",
  }).then((v) => ({ ...f, ...v }))
));
```

#### When the user asks for a "workflow"

If the user's request mentions running a "workflow" (or otherwise uses the
word "workflow"), fan the work out to subagents rather than doing it all
yourself. Explore with your own tools first as needed, then write JavaScript
in the `{tool_name}` tool that dispatches subagents with `task()` and
assembles their results. The point is to distribute the heavy work in
parallel, not to grind through it one tool call at a time.
"""


def render_repl_system_prompt(
    *,
    tool_name: str,
    timeout: float,
    memory_limit_mb: int,
    mode: Literal["thread", "turn", "call"],
    ptc_attached: bool = False,
) -> str:
    """Render the base REPL system prompt text for `CodeInterpreterMiddleware`.

    `ptc_attached` controls the "external side effects" bullet: when host
    tools are exposed as the `tools.*` namespace it points the model at the
    API reference; otherwise it states the REPL is pure computation.
    """
    if ptc_attached:
        side_effects_line = (
            "- External side effects from inside the REPL are only reachable "
            "via the `tools.*` namespace documented in the API reference below."
        )
    else:
        side_effects_line = (
            "- The REPL has no access to host tools, files, or the network: it "
            "is pure computation. Return values to communicate results."
        )
    if mode == "call":
        repl_intro_line = (
            f"An `{tool_name}` tool is available. It runs JavaScript in a fresh "
            "sandboxed REPL for each invocation."
        )
        state_persistence_line = (
            "- State (variables, functions) does not persist across tool calls. "
            "Each invocation starts from a blank environment."
        )
    elif mode == "thread":
        repl_intro_line = (
            f"An `{tool_name}` tool is available. It runs JavaScript in a persistent "
            "REPL."
        )
        state_persistence_line = (
            "- State (variables, functions) persists across tool calls and across "
            "multiple turns for this conversation thread."
        )
    else:
        repl_intro_line = (
            f"An `{tool_name}` tool is available. It runs JavaScript in a persistent "
            "REPL."
        )
        state_persistence_line = (
            "- State (variables, functions) persists across tool calls within "
            "a single turn of conversation. They DO NOT persist across multiple turns."
        )
    return _REPL_SYSTEM_PROMPT_TEMPLATE.format(
        repl_intro_line=repl_intro_line,
        state_persistence_line=state_persistence_line,
        side_effects_line=side_effects_line,
        timeout=timeout,
        memory_limit_mb=memory_limit_mb,
    )


def render_subagent_system_prompt(*, tool_name: str = "eval") -> str:
    """Render guidance for the top-level QuickJS `task` global."""
    return _SUBAGENT_SYSTEM_PROMPT_TEMPLATE.replace("{tool_name}", tool_name)


def render_eval_tool_code_doc(*, mode: Literal["thread", "turn", "call"]) -> str:
    """Render the eval tool's `code` argument description."""
    if mode == "call":
        persistence = (
            "Each call runs in a fresh REPL environment (no cross-call state)."
        )
    elif mode == "thread":
        persistence = (
            "State persists across calls and across turns in this conversation."
        )
    else:
        persistence = (
            "State persists across calls within a turn, but resets between turns."
        )
    return (
        "JavaScript expression or statement(s) to evaluate in the sandboxed REPL. "
        f"{persistence}"
    )


def render_eval_tool_description(*, mode: Literal["thread", "turn", "call"]) -> str:
    """Render the public eval tool description."""
    if mode == "call":
        state_line = (
            "Each call runs in a fresh sandboxed REPL with no state carried over."
        )
    elif mode == "thread":
        state_line = (
            "Persistent state is enabled: variables and functions defined in one "
            "call are visible to subsequent calls in this conversation."
        )
    else:
        state_line = (
            "Persistent state is enabled within a single turn: variables and "
            "functions defined in one call are visible to later calls within "
            "the same turn, but reset between turns."
        )
    return (
        "Execute JavaScript in a sandboxed REPL. "
        f"{state_line} No filesystem, network, or real clock. "
        "Top-level `await` is supported; a final-expression Promise resolves "
        "before the call returns."
    )


def to_camel_case(name: str) -> str:
    """Convert `snake_case` / `kebab-case` → `camelCase`."""
    return _CAMEL_SEP.sub(lambda m: m.group(1).upper(), name)


def is_valid_js_identifier(name: str) -> bool:
    """Return whether `name` is a valid JavaScript identifier."""
    return _JS_IDENTIFIER.fullmatch(name) is not None


def is_valid_ptc_tool_name(name: str) -> bool:
    """Return whether a tool can be exposed as `tools.<camelCaseName>`."""
    return is_valid_js_identifier(to_camel_case(name))


def render_ptc_prompt(tools: Sequence[BaseTool], *, tool_name: str = "eval") -> str:
    """Build the `tools` namespace section of the system prompt."""
    if not tools:
        return ""
    schemas, shared_schema, shared_keys = _collect_tool_schemas(tools)
    shared_refs, shared_definitions = _render_schema_definitions(shared_schema)
    blocks: list[str] = []
    for index, tool in enumerate(tools):
        camel = to_camel_case(tool.name)
        input_type = _render_tool_schema(
            schemas.get((index, "input")),
            refs=shared_refs if (index, "input") in shared_keys else None,
            default_type="Record<string, unknown>",
        )
        return_type = _render_tool_schema(
            schemas.get((index, "output")),
            refs=shared_refs if (index, "output") in shared_keys else None,
            default_type="unknown",
        )
        signature = _render_signature(
            camel,
            input_type=input_type,
            return_type=return_type,
        )
        description = (
            (tool.description or "").strip().splitlines()[0] if tool.description else ""
        )
        blocks.append(f"/** {description} */\n{signature}")
    body = "\n\n".join([*shared_definitions, *blocks])
    return (
        "\n\n"
        "### API Reference — `tools` namespace\n\n"
        "The agent tools listed below are exposed on the global object at "
        "`globalThis.tools` (also reachable as `tools`). Each takes a single "
        "object argument and returns a Promise that resolves to the tool's "
        "native value: strings as strings, numbers as numbers, lists as "
        "arrays, dicts as objects, and `None` as `null`. You do NOT need to "
        "`JSON.parse` results — they are already typed.\n\n"
        "Invocation pattern: `await tools.<name>({ ... })`.\n\n"
        "- Use `await` to get tool results; combine with `Promise.all` for "
        "independent calls so they run concurrently.\n"
        f"- If the task needs multiple tool calls, prefer one `{tool_name}` "
        "invocation that performs all of them rather than splitting the work "
        f"across multiple `{tool_name}` calls — each round-trip costs a model "
        "turn.\n"
        "- Pipeline dependent calls within a single program. If a result from "
        "one tool is needed as input to a later tool, chain them in one "
        "program instead of returning the intermediate value to the model.\n"
        "- If a tool returns an ID or other value that can be passed directly "
        "into the next tool, trust it and chain the calls instead of stopping "
        "to double-check it.\n"
        "- To inspect an intermediate value, `console.log` it inside the same "
        "program; otherwise, fetch as much information as possible in one "
        "call.\n"
        f"- Only split work across multiple `{tool_name}` invocations when "
        "you genuinely cannot determine what to do next without additional "
        "model reasoning or user input.\n\n"
        "Example shape — substitute real tool names:\n\n"
        "```typescript\n"
        'const users = await tools.findUsers({ name: "Ada" });\n'
        "const userId = users[0].id;\n"
        "const [city, normalized] = await Promise.all([\n"
        "  tools.cityForUser({ user_id: userId }),\n"
        '  tools.normalize({ name: "Ada" }),\n'
        "]);\n"
        "console.log({ city, normalized });\n"
        "```\n\n"
        "```typescript\n"
        f"{body}\n"
        "```"
    )


def _render_signature(
    fn_name: str,
    *,
    input_type: str,
    return_type: str,
) -> str:
    return f"tools.{fn_name}(input: {input_type}): Promise<{return_type}>"


_SchemaKey = tuple[int, Literal["input", "output"]]
_SchemaAdapter = tuple[_SchemaKey, Literal["validation"], TypeAdapter[Any]]
_JsonSchema = dict[str, Any] | bool


def _collect_tool_schemas(
    tools: Sequence[BaseTool],
) -> tuple[dict[_SchemaKey, dict[str, Any]], dict[str, Any], set[_SchemaKey]]:
    schemas: dict[_SchemaKey, dict[str, Any]] = {}
    adapters: list[_SchemaAdapter] = []
    for index, tool in enumerate(tools):
        input_key: _SchemaKey = (index, "input")
        args_schema = tool.args_schema
        if isinstance(args_schema, dict):
            schemas[input_key] = args_schema
        elif args_schema is not None:
            _add_input_schema_adapter(input_key, args_schema, schemas, adapters)

        output_key: _SchemaKey = (index, "output")
        annotation = _return_annotation(tool)
        if annotation is not None:
            _add_output_schema_adapter(
                output_key,
                annotation,
                schemas,
                adapters,
            )

    if not adapters:
        return schemas, {}, set()
    try:
        generated_roots, shared_schema = TypeAdapter.json_schemas(adapters)
    except Exception:  # noqa: BLE001 — retain per-tool best-effort fallbacks
        return schemas, {}, set()
    shared_roots = {key: schema for (key, _mode), schema in generated_roots.items()}
    schemas.update(shared_roots)
    return schemas, shared_schema, set(shared_roots)


def _add_input_schema_adapter(
    key: _SchemaKey,
    args_schema: Any,
    schemas: dict[_SchemaKey, dict[str, Any]],
    adapters: list[_SchemaAdapter],
) -> None:
    model_json_schema = getattr(args_schema, "model_json_schema", None)
    if not callable(model_json_schema):
        return
    try:
        declared_schema = model_json_schema()
        adapter = TypeAdapter(args_schema)
        generated_schema = adapter.json_schema()
    except Exception:  # noqa: BLE001 — preserve the existing best-effort behavior
        return
    schemas[key] = declared_schema
    if declared_schema == generated_schema:
        adapters.append((key, "validation", adapter))
        return
    # A malformed custom reference must not break prompt construction.
    with contextlib.suppress(Exception):
        schemas[key] = dereference_refs(declared_schema)


def _add_output_schema_adapter(
    key: _SchemaKey,
    annotation: Any,
    schemas: dict[_SchemaKey, dict[str, Any]],
    adapters: list[_SchemaAdapter],
) -> None:
    try:
        adapter = TypeAdapter(annotation)
        validation_schema = adapter.json_schema(mode="validation", by_alias=False)
        serialization_schema = adapter.json_schema(
            mode="serialization",
            by_alias=False,
        )
        aliased_schema = adapter.json_schema(mode="validation", by_alias=True)
    except Exception:  # noqa: BLE001 — one invalid tool must not break the prompt
        return
    if validation_schema != serialization_schema or validation_schema != aliased_schema:
        return
    schemas[key] = validation_schema
    adapters.append((key, "validation", adapter))


def _return_annotation(tool: BaseTool) -> Any | None:
    target = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    if target is None:
        return None
    annotation = inspect.Signature.empty
    with contextlib.suppress(TypeError, ValueError, NameError):
        signature = inspect.signature(target)
        resolved = get_type_hints(target)
        annotation = resolved.get("return", signature.return_annotation)
    if annotation is inspect.Signature.empty or annotation is Any:
        return None
    return annotation


def _render_tool_schema(
    schema: dict[str, Any] | None,
    *,
    refs: dict[str, str] | None,
    default_type: str,
) -> str:
    if schema is None:
        return default_type
    rendered = _json_schema_to_ts(schema, refs=refs)
    return (
        default_type
        if rendered == "unknown" and default_type != "unknown"
        else rendered
    )


def _json_schema_to_ts(
    prop: _JsonSchema,
    *,
    refs: dict[str, str] | None = None,
) -> str:
    """Render a JSON Schema node as a TypeScript type."""
    refs = refs or {}
    direct = _render_direct_schema(prop, refs=refs)
    if direct is not None:
        return direct
    prop = cast("dict[str, Any]", prop)
    composed = _render_composed(prop, refs=refs)
    if composed is not None:
        return composed
    t = prop.get("type")
    if isinstance(t, list):
        parts = [_json_schema_to_ts({**prop, "type": item}, refs=refs) for item in t]
        return " | ".join(dict.fromkeys(parts))
    scalar = _render_scalar(t)
    if scalar is not None:
        return scalar
    if t == "array":
        prefix_items = prop.get("prefixItems")
        if isinstance(prefix_items, list):
            return _render_tuple(prop, prefix_items, refs=refs)
        items = prop.get("items")
        inner = (
            _json_schema_to_ts(items, refs=refs)
            if isinstance(items, (dict, bool))
            else "unknown"
        )
        return _render_array(inner)
    if t == "object" or "properties" in prop:
        return _render_object(prop, refs=refs)
    return "unknown"


def _render_direct_schema(
    prop: _JsonSchema,
    *,
    refs: dict[str, str],
) -> str | None:
    if isinstance(prop, bool):
        return "unknown" if prop else "never"
    ref = prop.get("$ref")
    if isinstance(ref, str):
        return refs.get(ref, "unknown")
    if "const" in prop:
        return json.dumps(prop["const"])
    if "enum" in prop:
        return " | ".join(json.dumps(v) for v in prop["enum"])
    return None


def _render_composed(prop: dict[str, Any], *, refs: dict[str, str]) -> str | None:
    for key, separator in (("anyOf", " | "), ("oneOf", " | "), ("allOf", " & ")):
        branches = prop.get(key)
        if not isinstance(branches, list):
            continue
        parts = [
            _json_schema_to_ts(branch, refs=refs)
            for branch in branches
            if isinstance(branch, (dict, bool))
        ]
        return separator.join(dict.fromkeys(parts))
    return None


def _render_scalar(schema_type: Any) -> str | None:
    if schema_type == "string":
        return "string"
    if schema_type in {"integer", "number"}:
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "null":
        return "null"
    return None


def _render_object(prop: dict[str, Any], *, refs: dict[str, str]) -> str:
    sub_props = prop.get("properties")
    additional = prop.get("additionalProperties")
    if isinstance(sub_props, dict) and sub_props:
        required = set(prop.get("required", []))
        rendered_properties = [
            (
                key,
                value,
                _json_schema_to_ts(value, refs=refs),
            )
            for key, value in sub_props.items()
            if isinstance(value, (dict, bool))
        ]
        fields = [
            f"{_property_description(value)}"
            f"{_typescript_property(key)}{'' if key in required else '?'}: "
            f"{value_type}"
            for key, value, value_type in rendered_properties
        ]
        object_type = "{ " + "; ".join(fields) + " }"
        if isinstance(additional, dict) or additional is True:
            # TypeScript index signatures also constrain explicitly declared fields.
            index_types = [_json_schema_to_ts(additional, refs=refs)]
            index_types.extend(
                value_type if key in required else f"{value_type} | undefined"
                for key, _value, value_type in rendered_properties
            )
            index_type = " | ".join(dict.fromkeys(index_types))
            return f"{object_type} & Record<string, {index_type}>"
        return object_type
    if isinstance(additional, (dict, bool)):
        value_type = _json_schema_to_ts(additional, refs=refs)
        return f"Record<string, {value_type}>"
    return "Record<string, unknown>"


def _property_description(prop: _JsonSchema) -> str:
    if isinstance(prop, bool):
        return ""
    description = prop.get("description")
    return f"/** {description} */ " if isinstance(description, str) else ""


def _render_tuple(
    prop: dict[str, Any],
    prefix_items: list[Any],
    *,
    refs: dict[str, str],
) -> str:
    items = [
        _json_schema_to_ts(item, refs=refs)
        for item in prefix_items
        if isinstance(item, (dict, bool))
    ]
    additional = prop.get("items")
    if isinstance(additional, (dict, bool)):
        items.append(f"...{_render_array(_json_schema_to_ts(additional, refs=refs))}")
    return "[" + ", ".join(items) + "]"


def _render_array(inner: str) -> str:
    if " | " in inner or " & " in inner:
        return f"({inner})[]"
    return f"{inner}[]"


def _render_schema_definitions(
    schema: dict[str, Any] | None,
) -> tuple[dict[str, str], list[str]]:
    if not schema:
        return {}, []
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return {}, []
    refs = _schema_ref_names(definitions)
    rendered = [
        f"type {refs[_definition_ref(name)]} = "
        f"{_json_schema_to_ts(definition, refs=refs)};"
        for name, definition in definitions.items()
        if isinstance(name, str) and isinstance(definition, (dict, bool))
    ]
    return refs, rendered


def _schema_ref_names(
    definitions: dict[str, Any],
) -> dict[str, str]:
    refs: dict[str, str] = {}
    used: set[str] = set()
    for name in definitions:
        if not isinstance(name, str):
            continue
        alias = _generate_type_name(name, used)
        refs[_definition_ref(name)] = alias
    return refs


def _generate_type_name(value: str, used: set[str]) -> str:
    base = _to_type_name(value)
    if base not in used and base not in _RESERVED_TYPE_NAMES:
        used.add(base)
        return base
    suffix = 1
    while f"{base}{suffix}" in used:
        suffix += 1
    alias = f"{base}{suffix}"
    used.add(alias)
    return alias


def _definition_ref(name: str) -> str:
    escaped = name.replace("~", "~0").replace("/", "~1")
    return f"#/$defs/{escaped}"


def _to_type_name(value: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
    name = "".join(part[:1].upper() + part[1:] for part in parts)
    return re.sub(r"^\d+", "", name) or "NoName"


def _typescript_property(name: str) -> str:
    if is_valid_js_identifier(name):
        return name
    return json.dumps(name)
