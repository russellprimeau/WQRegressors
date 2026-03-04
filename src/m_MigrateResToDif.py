#!/usr/bin/env python3
"""Repository-wide rename tool for migrating feature suffixes (e.g., _res -> _dif).

Exact behavior:
- Performs a literal text replacement of "_res" with "_dif" in file contents.
- This is not semantic and not regex-suffix-aware, so it will also change tokens
  like "ss_res" -> "ss_dif" and "add_res" -> "add_dif".
- Default target file extensions are: .py, .csv, .json, .yml, .yaml.
- Default excluded directories are: .git, .venv, .idea, .vscode, __pycache__.
- With --rename-paths, it also renames filenames and directory names containing
  "_res" (deepest paths first).
- Rename collisions are detected; if a destination already exists, the script
  reports collisions and exits without applying those path renames.
- Default mode is preview (dry-run). Only --apply writes changes.

Approval and grouping model:
- The script does not ask per-file approval.
- Changes are grouped per execution:
  1) content edits (file list + replacement counts),
  2) path renames (old -> new),
  3) collisions (if any).
- If run through Codex tools, approval is command-level, not per file.
- If run directly in your terminal, there is no approval prompt; with --apply,
  all changes from that run are applied.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXTENSIONS = [".py", ".csv", ".json", ".yml", ".yaml"]
DEFAULT_EXCLUDED_DIRS = [".git", ".venv", ".idea", ".vscode", "__pycache__"]


@dataclass
class ContentChange:
    path: Path
    replacements: int
    encoding: str


@dataclass
class PathChange:
    old_path: Path
    new_path: Path


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_extensions(items: Iterable[str]) -> set[str]:
    normalized = set()
    for item in items:
        ext = item.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def is_excluded(path: Path, root: Path, excluded_dirs: set[str]) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in excluded_dirs for part in rel_parts)


def is_hidden_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part.startswith(".") for part in parts)


def read_text_with_fallback(path: Path) -> tuple[str | None, str | None]:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None


def collect_file_content_changes(
    root: Path,
    from_token: str,
    to_token: str,
    extensions: set[str],
    excluded_dirs: set[str],
    include_hidden: bool,
    all_files: bool,
) -> list[ContentChange]:
    changes: list[ContentChange] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_excluded(path, root, excluded_dirs):
            continue
        if not include_hidden and is_hidden_path(path, root):
            continue
        if not all_files and path.suffix.lower() not in extensions:
            continue

        text, encoding = read_text_with_fallback(path)
        if text is None or encoding is None:
            continue

        replacements = text.count(from_token)
        if replacements:
            changes.append(ContentChange(path=path, replacements=replacements, encoding=encoding))

    return changes


def apply_file_content_changes(changes: list[ContentChange], from_token: str, to_token: str) -> None:
    for change in changes:
        text = change.path.read_text(encoding=change.encoding)
        updated = text.replace(from_token, to_token)
        if updated != text:
            change.path.write_text(updated, encoding=change.encoding)


def collect_path_changes(
    root: Path,
    from_token: str,
    to_token: str,
    excluded_dirs: set[str],
    include_hidden: bool,
) -> tuple[list[PathChange], list[PathChange]]:
    changes: list[PathChange] = []
    collisions: list[PathChange] = []

    all_paths = sorted(
        (p for p in root.rglob("*") if not is_excluded(p, root, excluded_dirs)),
        key=lambda p: len(p.relative_to(root).parts),
        reverse=True,
    )

    for old_path in all_paths:
        if not include_hidden and is_hidden_path(old_path, root):
            continue

        if from_token not in old_path.name:
            continue

        new_name = old_path.name.replace(from_token, to_token)
        new_path = old_path.with_name(new_name)

        if new_path.exists() and new_path != old_path:
            collisions.append(PathChange(old_path=old_path, new_path=new_path))
        else:
            changes.append(PathChange(old_path=old_path, new_path=new_path))

    return changes, collisions


def apply_path_changes(changes: list[PathChange]) -> None:
    for change in changes:
        if change.old_path.exists():
            change.old_path.rename(change.new_path)


def verify_remaining_tokens(root: Path, token: str, excluded_dirs: set[str], include_hidden: bool) -> tuple[int, int]:
    files_with_token = 0
    token_count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_excluded(path, root, excluded_dirs):
            continue
        if not include_hidden and is_hidden_path(path, root):
            continue

        text, _ = read_text_with_fallback(path)
        if text is None:
            continue

        count = text.count(token)
        if count:
            files_with_token += 1
            token_count += count

    return files_with_token, token_count


def print_preview(title: str, rows: list[str], limit: int = 30) -> None:
    print(title)
    if not rows:
        print("  (none)")
        return

    for row in rows[:limit]:
        print(f"  {row}")
    if len(rows) > limit:
        print(f"  ... and {len(rows) - limit} more")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename token occurrences in file contents and optionally in file/directory names."
    )
    parser.add_argument("--root", default=".", help="Repository root to process (default: current directory)")
    parser.add_argument("--from-token", default="_res", help="Token to replace (default: _res)")
    parser.add_argument("--to-token", default="_dif", help="Replacement token (default: _dif)")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated file extensions to edit (default: .py,.csv,.json,.yml,.yaml)",
    )
    parser.add_argument(
        "--exclude-dirs",
        default=",".join(DEFAULT_EXCLUDED_DIRS),
        help="Comma-separated directory names to skip everywhere",
    )
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden paths")
    parser.add_argument("--all-files", action="store_true", help="Edit all decodable text files, ignoring --extensions")
    parser.add_argument("--rename-paths", action="store_true", help="Also rename files/directories containing from-token")
    parser.add_argument("--verify", action="store_true", help="After processing, scan for remaining from-token in text files")
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: root path is not a directory: {root}", file=sys.stderr)
        return 2

    if not args.from_token:
        print("ERROR: --from-token cannot be empty", file=sys.stderr)
        return 2

    if args.from_token == args.to_token:
        print("ERROR: --from-token and --to-token are the same", file=sys.stderr)
        return 2

    excluded_dirs = set(parse_csv_arg(args.exclude_dirs))
    extensions = normalize_extensions(parse_csv_arg(args.extensions))

    content_changes = collect_file_content_changes(
        root=root,
        from_token=args.from_token,
        to_token=args.to_token,
        extensions=extensions,
        excluded_dirs=excluded_dirs,
        include_hidden=args.include_hidden,
        all_files=args.all_files,
    )

    path_changes: list[PathChange] = []
    collisions: list[PathChange] = []
    if args.rename_paths:
        path_changes, collisions = collect_path_changes(
            root=root,
            from_token=args.from_token,
            to_token=args.to_token,
            excluded_dirs=excluded_dirs,
            include_hidden=args.include_hidden,
        )

    total_replacements = sum(change.replacements for change in content_changes)

    print(f"Root: {root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Content files with replacements: {len(content_changes)}")
    print(f"Total token replacements in content: {total_replacements}")
    print(f"Path rename operations: {len(path_changes)}")

    print_preview(
        "Content edits preview:",
        [f"{c.path.relative_to(root)}  (x{c.replacements})" for c in sorted(content_changes, key=lambda x: str(x.path))],
    )

    if args.rename_paths:
        print_preview(
            "Path rename preview:",
            [f"{c.old_path.relative_to(root)} -> {c.new_path.relative_to(root)}" for c in path_changes],
        )

    if collisions:
        print_preview(
            "Name collisions detected (manual resolution needed):",
            [f"{c.old_path.relative_to(root)} -> {c.new_path.relative_to(root)}" for c in collisions],
        )
        return 1

    if args.apply:
        apply_file_content_changes(content_changes, args.from_token, args.to_token)
        if args.rename_paths:
            apply_path_changes(path_changes)
        print("Changes applied.")
    else:
        print("Dry-run only. Re-run with --apply to write changes.")

    if args.verify:
        files_with_token, token_count = verify_remaining_tokens(
            root=root,
            token=args.from_token,
            excluded_dirs=excluded_dirs,
            include_hidden=args.include_hidden,
        )
        print(f"Verification: {files_with_token} files still contain '{args.from_token}' ({token_count} total matches)")
        if files_with_token > 0:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
