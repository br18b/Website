from __future__ import annotations

from pathlib import Path


def find_code_root(start: str | Path) -> Path:
    """Return the scripts/mandelbrot repository root.

    The repository has a fixed layout.  We deliberately fail when it cannot be
    found instead of silently inventing a standalone fallback mode.
    """
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "build.sh").is_file() and (candidate / "components").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate Mandelbrot code root from {start}")


def find_project_root(start: str | Path) -> Path:
    code_root = find_code_root(start)
    for candidate in (code_root, *code_root.parents):
        if (candidate / ".git").exists() or (candidate / ".root").exists():
            return candidate
    # The website checkout commonly contains scripts/mandelbrot without a .git
    # directory in exported bundles.  In that fixed layout, the project root is
    # two levels above scripts/mandelbrot.
    if code_root.parent.name == "scripts":
        return code_root.parent.parent
    raise RuntimeError(f"Could not locate project root from {start}")


def expand_project_vars(value: str | Path, *, code_root: Path, project_root: Path) -> Path:
    text = str(value)
    text = text.replace("${code_root}", str(code_root)).replace("$code_root", str(code_root))
    text = text.replace("${project_root}", str(project_root)).replace("$project_root", str(project_root))
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = code_root / path
    return path.resolve()
