from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CONFIG_PATH = Path(__file__).parent.parent / "settings.yaml"


def _parse_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value[0] in {'"', "'"} and value[-1] == value[0]:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _load_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, result)]

    with path.open() as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue
            indent = len(line) - len(line.lstrip(" "))
            line = line.lstrip(" ")
            if ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            while stack and indent < stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value == "":
                new_node: dict[str, Any] = {}
                parent[key] = new_node
                stack.append((indent + 2, new_node))
            else:
                parent[key] = _parse_value(value)
    return result


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config() -> SimpleNamespace:
    if yaml is not None:
        with CONFIG_PATH.open() as f:
            data = yaml.safe_load(f)
    else:
        data = _load_yaml(CONFIG_PATH)
    return _to_namespace(data)


cfg = load_config()


def reload() -> None:
    global cfg
    cfg = load_config()
