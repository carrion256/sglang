"""Request-local Responses identities and wire translation for the pinned SDK."""

import copy
import json

from pydantic import TypeAdapter

from sglang.srt.entrypoints.openai.protocol import ResponsesResponse, ResponseTool

_OUTPUT_ADAPTER = TypeAdapter(ResponsesResponse.model_fields["output"].annotation)


def as_dict(value):
    return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else value


def qualified_name(name, namespace=None):
    for value in (name,) if namespace is None else (name, namespace):
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError("Tool names and namespaces must be nonempty strings")
    return name if namespace is None else f"{namespace}.{name}"


def custom_input(arguments):
    try:
        value = json.loads(arguments)
    except (ValueError, TypeError) as error:
        raise ValueError("Custom tool arguments must encode an input string") from error
    if not isinstance(value, dict) or set(value) != {"input"} or not isinstance(value["input"], str):
        raise ValueError("Custom tool arguments must encode exactly one input string")
    return value["input"]


def validated_json_calls(content, names):
    calls = json.loads(content)
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list) or not calls:
        raise ValueError("Required tool output must contain calls")
    for call in calls:
        if not isinstance(call, dict) or call.get("name") not in names:
            raise ValueError("Unknown generated tool identity in required output")
        arguments = call.get("parameters", call.get("arguments", {}))
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        yield call["name"], json.dumps(arguments, ensure_ascii=False)


class ToolRegistry:
    def __init__(self, tools, *, reasoning=None, tool_choice=None):
        self.reasoning = copy.deepcopy(as_dict(reasoning))
        self.tool_choice = copy.deepcopy(tool_choice)
        self.selected = None
        self.original = [copy.deepcopy(as_dict(tool)) for tool in tools or []]
        self.identities = {}
        self.functions = []
        self.builtins = []
        for tool in self.original:
            if tool.get("type") == "namespace":
                namespace = tool.get("name")
                qualified_name(namespace)
                members = tool.get("tools")
                if not isinstance(members, list) or not members:
                    raise ValueError("Namespaces must contain function or custom tools")
                for member in members:
                    self._add(as_dict(member), namespace, tool.get("description"))
            elif tool.get("type") in ("function", "custom"):
                self._add(tool)
            else:
                self.builtins.append(tool)
        self.replayed_identities = dict(self.identities)
        self.history_identities = {}

    def _add(self, tool, namespace=None, description=None):
        if not isinstance(tool, dict) or tool.get("type") not in ("function", "custom"):
            raise ValueError("Unsupported namespace member; expected function or custom")
        if any(value is not None and not isinstance(value, str)
               for value in (description, tool.get("description"))):
            raise ValueError("Tool descriptions must be strings")
        name = tool.get("name")
        qualified = qualified_name(name, namespace)
        if qualified in self.identities:
            raise ValueError(f"Colliding tool identity: {qualified}")
        kind = tool["type"]
        self.identities[qualified] = (name, namespace, kind)
        function = {"type": "function", "name": qualified,
                    "description": "\n\n".join(
                        text for text in (
                            f"Namespace description:\n{description}" if description else "",
                            tool.get("description") or "",
                        ) if text)}
        if kind == "custom":
            tool_format = tool.get("format")
            if tool_format is None:
                tool_format = {"type": "text"}
            if not isinstance(tool_format, dict) or tool_format.get("type") not in ("text", "grammar"):
                raise ValueError("Unsupported custom tool format")
            if tool_format["type"] == "grammar":
                if tool_format.get("syntax") not in ("lark", "regex") or not isinstance(tool_format.get("definition"), str):
                    raise ValueError("Malformed custom grammar")
                function["description"] += f"\nInput grammar (descriptive, not enforced; syntax: {tool_format['syntax']}):\n" + tool_format["definition"]
            function["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}},
                                      "required": ["input"], "additionalProperties": False}
            function["strict"] = True
        else:
            parameters = tool.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                raise ValueError("Function parameters must be an object")
            if tool.get("strict") is not None and not isinstance(tool["strict"], bool):
                raise ValueError("Function strict must be boolean")
            function.update(parameters=parameters, strict=tool.get("strict") or False)
        self.functions.append(function)

    def identity(self, qualified):
        if qualified not in self.identities:
            raise ValueError(f"Unknown generated tool identity: {qualified}")
        return self.identities[qualified]

    def output_identity(self, qualified):
        if self.selected is not None and qualified != self.selected:
            raise ValueError("Generated tool call does not match forced tool choice")
        return self.identity(qualified)

    def validate_completion(self, output, status):
        if status != "completed":
            return
        calls = [as_dict(item) for item in output
                 if as_dict(item).get("type") == "function_call"]
        if self.selected is not None and len(calls) != 1:
            raise ValueError("Forced tool choice requires exactly one call")
        if self.tool_choice == "required" and not calls:
            raise ValueError("Required tool choice requires at least one call")
        for call in calls:
            self.output_item(call)

    def choice(self, choice):
        if isinstance(choice, str):
            if choice == "required" and not self.functions:
                raise ValueError("Required choice needs a declared function or custom tool")
            return choice
        if not isinstance(choice, dict) or choice.get("type") not in ("function", "custom"):
            raise ValueError("Unsupported forced tool choice")
        selected = choice.get("function", choice)
        if not isinstance(selected, dict):
            raise ValueError("Malformed forced tool choice")
        if "function" in choice:
            qualified_name(selected.get("name"), selected.get("namespace"))
            if "name" in choice:
                qualified_name(choice["name"])
            if choice.get("namespace") is not None:
                qualified_name(choice["namespace"])
            for field in ("name", "namespace", "type"):
                if field in choice and field in selected and choice[field] != selected[field]:
                    raise ValueError("Conflicting forced tool choice representations")
        name, namespace = selected.get("name"), selected.get("namespace", choice.get("namespace"))
        qualified = qualified_name(name, namespace)
        if self.identity(qualified) != (name, namespace, choice["type"]):
            raise ValueError("Forced tool choice does not match a declared identity")
        self.selected = qualified
        return {"type": "function", "name": qualified}

    def remember_identity(self, identity):
        name, namespace, kind = identity
        qualified = qualified_name(name, namespace)
        if self.replayed_identities.get(qualified, identity) != identity:
            raise ValueError("Replayed call does not match a declared identity")
        self.replayed_identities[qualified] = identity
        self.history_identities[qualified] = identity

    def inherit_history(self, previous, output):
        for identity in previous.history_identities.values():
            self.remember_identity(identity)
        for item in output:
            item = as_dict(item)
            if item.get("type") == "function_call":
                self.remember_identity(previous.identity(item["name"]))

    def replay(self, item):
        item = copy.deepcopy(as_dict(item))
        if "function_call" in item:
            raise ValueError("Legacy embedded function_call is unsupported; use typed replay")
        if "tool_calls" in item:
            if item.get("role") != "assistant" or item.get("type") not in (None, "message"):
                raise ValueError("Embedded tool_calls require an assistant message")
            calls = item["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError("Embedded tool_calls must be a list")
            normalized = []
            for call in calls:
                if (not isinstance(call, dict) or call.get("type") != "function"
                        or set(call) - {"id", "type", "function"}
                        or not isinstance(call.get("id"), str) or not call["id"]):
                    raise ValueError("Unsupported embedded call; use typed function/custom replay")
                function = call.get("function")
                if (not isinstance(function, dict)
                        or set(function) - {"name", "namespace", "arguments"}
                        or not isinstance(function.get("arguments"), str)):
                    raise ValueError("Malformed embedded function call")
                replayed = self.replay({"type": "function_call", "call_id": call["id"], **function})
                normalized.append({"id": call["id"], "type": "function", "function": {
                    "name": replayed["name"], "arguments": replayed["arguments"]}})
            item["tool_calls"] = normalized
        if item.get("type") not in ("function_call", "custom_tool_call"):
            if item.get("type") == "custom_tool_call_output":
                item["type"] = "function_call_output"
            return item
        name, namespace = item.get("name"), item.get("namespace")
        qualified = qualified_name(name, namespace)
        kind = "custom" if item["type"] == "custom_tool_call" else "function"
        identity = (name, namespace, kind)
        self.remember_identity(identity)
        item.pop("namespace", None)
        item["name"] = qualified
        if kind == "custom":
            if not isinstance(item.get("input"), str):
                raise ValueError("Custom tool input must be a string")
            item["arguments"] = json.dumps({"input": item.pop("input")}, ensure_ascii=False)
            item["type"] = "function_call"
        return item

    def output_item(self, item, partial=False):
        item = copy.deepcopy(item.model_dump() if hasattr(item, "model_dump") else item)
        if item.get("type") != "function_call":
            return item
        name, namespace, kind = self.output_identity(item["name"])
        item["name"] = name
        if namespace is not None:
            item["namespace"] = namespace
        if kind == "custom":
            arguments = item.pop("arguments")
            item["input"] = "" if partial else custom_input(arguments)
            item["type"] = "custom_tool_call"
        return item

    def response(self, response):
        response = copy.deepcopy(response.model_dump() if hasattr(response, "model_dump") else response)
        if self.reasoning is not None:
            response["reasoning"] = copy.deepcopy(self.reasoning)
        if self.tool_choice is not None:
            response["tool_choice"] = copy.deepcopy(self.tool_choice)
        response["tools"] = copy.deepcopy(self.original)
        response["output"] = [self.output_item(item) for item in response.get("output", [])]
        return response

    def response_model(self, response):
        wire = self.response(response)
        return response.model_copy(update={
            "output": _OUTPUT_ADAPTER.validate_python(wire["output"]),
            "tools": [ResponseTool.model_validate(tool) for tool in self.original],
            "reasoning": wire.get("reasoning"),
            "tool_choice": wire.get("tool_choice"),
        })

    async def stream(self, source):
        calls = {}
        sequence = 0
        try:
            async for frame in source:
                data = next((line[6:] for line in frame.splitlines() if line.startswith("data: ")), None)
                if data is None or data == "[DONE]":
                    yield frame
                    continue
                event = json.loads(data)
                kind = event.get("type", "")
                emitted = [event]
                if "response" in event:
                    event["response"] = self.response(event["response"])
                if kind in ("response.output_item.added", "response.output_item.done"):
                    item = event["item"]
                    if item.get("type") == "function_call":
                        calls[item["id"]] = self.identity(item["name"])
                        event["item"] = self.output_item(item, partial=kind.endswith("added"))
                if kind.startswith("response.function_call_arguments."):
                    identity = calls.get(event["item_id"])
                    if identity is None:
                        raise ValueError("Tool argument event has no declared output item")
                    name, namespace, tool_type = identity
                    event["name"] = name
                    if namespace is not None:
                        event["namespace"] = namespace
                    if tool_type == "custom":
                        if kind.endswith("delta"):
                            continue
                        payload = custom_input(event.pop("arguments"))
                        event["type"] = "response.custom_tool_call_input.done"
                        event["input"] = payload
                        delta = {key: value for key, value in event.items() if key != "input"}
                        delta.update(type="response.custom_tool_call_input.delta", delta=payload)
                        emitted = [delta, event]
                for wire_event in emitted:
                    if "sequence_number" in wire_event:
                        wire_event["sequence_number"] = sequence
                        sequence += 1
                    yield f"event: {wire_event.get('type', 'error')}\ndata: {json.dumps(wire_event, ensure_ascii=False)}\n\n"
        except ValueError as error:
            yield "event: error\ndata: " + json.dumps({"type": "error", "error": {"message": str(error), "type": "BadRequestError"}}) + "\n\n"
        finally:
            await source.aclose()
