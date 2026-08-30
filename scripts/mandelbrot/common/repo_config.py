from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from common.project_root import expand_project_vars, find_code_root, find_project_root

DEFAULT_CONFIG_NAME = "mandelbrot.json"
DATA_ROOT_ENV = "MANDELBROT_DATA_ROOT"


@dataclass(frozen=True)
class RepositoryPaths:
    code_root: Path
    project_root: Path
    config_path: Path
    data_root: Path


def resolve_data_root(
    explicit: str | Path | None,
    *,
    project_root: Path,
) -> Path:
    """Resolve private Mandelbrot state without depending on the cwd.

    An explicit caller-supplied root wins over ``MANDELBROT_DATA_ROOT``.  A
    relative explicit/environment path is anchored at the repository root.
    """
    selected = explicit
    if selected is None:
        selected = os.environ.get(DATA_ROOT_ENV) or None
    if selected is None:
        return (project_root / "work" / "mandelbrot").resolve()

    text = str(selected)
    text = text.replace("${project_root}", str(project_root))
    text = text.replace("$project_root", str(project_root))
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


class RepoConfig:
    """Read the single repository-wide JSON configuration.

    Numeric/search settings live here.  Demo-only presentation settings remain
    under ``demo`` in the same file.  Callers may request nested values with a
    dotted path, for example ``runtime.threads``.
    """

    def __init__(self, data: dict[str, Any], paths: RepositoryPaths):
        self.data = data
        self.paths = paths

    @classmethod
    def load(
        cls,
        config: str | Path | None = None,
        *,
        start: str | Path,
        data_root: str | Path | None = None,
    ) -> "RepoConfig":
        code_root = find_code_root(start)
        project_root = find_project_root(code_root)
        config_path = Path(config) if config is not None else code_root / DEFAULT_CONFIG_NAME
        if not config_path.is_absolute():
            config_path = (code_root / config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Repository config does not exist: {config_path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Top-level config must be a JSON object: {config_path}")
        resolved_data_root = resolve_data_root(data_root, project_root=project_root)
        return cls(
            data,
            RepositoryPaths(code_root, project_root, config_path, resolved_data_root),
        )

    def get(self, dotted: str, default: Any = ...):
        current: Any = self.data
        for part in dotted.split(".") if dotted else ():
            if not isinstance(current, dict) or part not in current:
                if default is ...:
                    raise KeyError(f"Missing config value: {dotted}")
                return default
            current = current[part]
        return current

    def section(self, dotted: str) -> dict[str, Any]:
        value = self.get(dotted)
        if not isinstance(value, dict):
            raise TypeError(f"Config section {dotted!r} must be an object")
        return value

    def string(self, dotted: str, fallback: str = "") -> str:
        """Return a string value, matching the C++ RepoConfig API."""
        value = self.get(dotted, fallback)
        if not isinstance(value, str):
            raise TypeError(f"Config value {dotted!r} must be a string")
        return value

    def number(self, dotted: str, fallback: float = 0.0) -> float:
        """Return a numeric value, rejecting booleans as numbers."""
        value = self.get(dotted, fallback)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Config value {dotted!r} must be numeric")
        return float(value)

    def integer(self, dotted: str, fallback: int = 0) -> int:
        """Return an integer using the same numeric conversion as C++."""
        return int(self.number(dotted, fallback))

    def u64(self, dotted: str, fallback: int = 0) -> int:
        """Return a non-negative integer suitable for uint64 settings."""
        value = self.integer(dotted, fallback)
        if value < 0:
            raise ValueError(f"Config value {dotted!r} must be non-negative")
        return value

    def boolean(self, dotted: str, fallback: bool = False) -> bool:
        """Return a Boolean value without coercing strings or numbers."""
        value = self.get(dotted, fallback)
        if not isinstance(value, bool):
            raise TypeError(f"Config value {dotted!r} must be boolean")
        return value

    def number_array(self, dotted: str) -> list[float]:
        """Return a numeric array, matching the C++ RepoConfig API."""
        value = self.get(dotted)
        if not isinstance(value, list):
            raise TypeError(f"Config value {dotted!r} must be an array")
        result: list[float] = []
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(
                    f"Config value {dotted!r}[{index}] must be numeric"
                )
            result.append(float(item))
        return result

    def string_array(self, dotted: str) -> list[str]:
        """Return a string array, matching the C++ RepoConfig API."""
        value = self.get(dotted)
        if not isinstance(value, list):
            raise TypeError(f"Config value {dotted!r} must be an array")
        result: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise TypeError(
                    f"Config value {dotted!r}[{index}] must be a string"
                )
            result.append(item)
        return result

    def path(self, dotted: str, default: str | Path | None = None) -> Path:
        raw = self.get(dotted, default)
        if raw is None:
            raise KeyError(f"Missing config path: {dotted}")
        value = str(raw)
        value = value.replace("${data_root}", str(self.paths.data_root))
        value = value.replace("$data_root", str(self.paths.data_root))
        return expand_project_vars(
            value,
            code_root=self.paths.code_root,
            project_root=self.paths.project_root,
        )

    @property
    def data_root(self) -> Path:
        return self.paths.data_root

    @property
    def catalogue_root(self) -> Path:
        return self.data_root / "component_catalogue"

    @property
    def contours_root(self) -> Path:
        return self.data_root / "G_contours"

    @property
    def promotion_root(self) -> Path:
        return self.paths.project_root / "work" / "promote" / "mandelbrot"

    @property
    def threads(self) -> int:
        value = int(self.get("runtime.threads", 0))
        if value > 0:
            return value
        import os
        return max(1, os.cpu_count() or 1)


def add_config_argument(parser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Repository JSON config (default: <code-root>/{DEFAULT_CONFIG_NAME})",
    )
