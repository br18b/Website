#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

COMMON_DIR = Path(__file__).resolve().parent
CODE_ROOT = COMMON_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.project_root import find_code_root, find_project_root  # noqa: E402
from common.repo_config import resolve_data_root  # noqa: E402


class RepositoryResolutionTests(unittest.TestCase):
    def make_layout(self, base: Path, marker: str) -> tuple[Path, Path]:
        repository = base / "movable-repository"
        code_root = repository / "scripts" / "mandelbrot"
        (code_root / "components").mkdir(parents=True)
        (code_root / "build.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        marker_path = repository / ".git"
        if marker == "directory":
            marker_path.mkdir()
        else:
            marker_path.write_text("gitdir: elsewhere\n", encoding="utf-8")
        return repository, code_root

    def test_git_directory_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, code_root = self.make_layout(Path(temporary), "directory")
            self.assertEqual(find_code_root(code_root / "components"), code_root)
            self.assertEqual(find_project_root(code_root), repository)

    def test_git_file_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, code_root = self.make_layout(Path(temporary), "file")
            self.assertEqual(find_project_root(code_root / "build.sh"), repository)

    def test_missing_repository_marker_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            code_root = Path(temporary) / "scripts" / "mandelbrot"
            (code_root / "components").mkdir(parents=True)
            (code_root / "build.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Could not locate project root"):
                find_project_root(code_root)

    def test_default_data_root_moves_with_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "relocated"
            old = os.environ.pop("MANDELBROT_DATA_ROOT", None)
            try:
                self.assertEqual(
                    resolve_data_root(None, project_root=repository),
                    (repository / "work" / "mandelbrot").resolve(),
                )
            finally:
                if old is not None:
                    os.environ["MANDELBROT_DATA_ROOT"] = old


if __name__ == "__main__":
    unittest.main()
