from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the nearest parent containing .git or .root.

    If no marker is found, return the resolved starting directory. This keeps
    scripts portable across machines and avoids hard-coded repository paths.
    """
    if start is None:
        cur = Path.cwd().resolve()
    else:
        cur = Path(start).expanduser().resolve()

    if cur.is_file():
        cur = cur.parent

    original = cur
    while True:
        if (cur / ".git").exists() or (cur / ".root").exists():
            return cur
        if cur.parent == cur:
            return original
        cur = cur.parent


def expand_project_vars(value: str | Path, *, code_root: str | Path, project_root: str | Path) -> Path:
    """Expand $code_root and $project_root in a path-like config value."""
    text = str(value).strip()
    code_root = Path(code_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    text = text.replace("${code_root}", str(code_root)).replace("$code_root", str(code_root))
    text = text.replace("${project_root}", str(project_root)).replace("$project_root", str(project_root))
    return Path(text).expanduser()
