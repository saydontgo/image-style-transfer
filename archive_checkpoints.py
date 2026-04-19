from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


ARCHIVE_SUFFIXES = (
    "_final.pth",
    "_config.json",
    ".ckpt",
    ".pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive flat checkpoint files under checkpoints/<style>/."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained model files to archive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying files.",
    )
    return parser.parse_args()


def is_archive_candidate(file_path: Path) -> bool:
    if not file_path.is_file():
        return False
    return any(file_path.name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def infer_style_name(filename: str) -> str | None:
    for marker in ("_custom_", "_custom."):
        if marker in filename:
            style_name = filename.split(marker, maxsplit=1)[0].strip("._-")
            return style_name or None
    return None


def infer_run_name(filename: str) -> str | None:
    for suffix in ("_final.pth", "_config.json"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    for suffix in (".ckpt", ".pth"):
        if filename.endswith(suffix) and "_step_" in filename:
            return filename.rsplit("_step_", maxsplit=1)[0]
    return None


def next_archive_index(style_dir: Path) -> int:
    if not style_dir.exists():
        return 1

    max_index = 0
    for child in style_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^(\d{3})_", child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def build_archive_plan(checkpoints_dir: Path) -> list[tuple[Path, Path]]:
    files_by_run: dict[str, list[Path]] = defaultdict(list)
    run_styles: dict[str, str] = {}

    for file_path in sorted(checkpoints_dir.iterdir()):
        if not is_archive_candidate(file_path):
            continue

        run_name = infer_run_name(file_path.name)
        if run_name is None:
            continue

        style_name = infer_style_name(f"{run_name}_final.pth")
        if style_name is None:
            continue

        files_by_run[run_name].append(file_path)
        run_styles[run_name] = style_name

    planned_moves: list[tuple[Path, Path]] = []
    next_indices: dict[str, int] = {}

    for run_name in sorted(files_by_run):
        style_name = run_styles[run_name]
        style_dir = checkpoints_dir / style_name
        if style_name not in next_indices:
            next_indices[style_name] = next_archive_index(style_dir)

        archive_index = next_indices[style_name]
        archive_dir = style_dir / f"{archive_index:03d}_{run_name}"

        for file_path in files_by_run[run_name]:
            destination = archive_dir / file_path.name
            planned_moves.append((file_path, destination))

        next_indices[style_name] += 1

    return planned_moves


def execute_archive(plan: list[tuple[Path, Path]], dry_run: bool) -> None:
    if not plan:
        print("No checkpoint files need archiving.")
        return

    for source, destination in plan:
        print(f"{source} -> {destination}")
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    if dry_run:
        print(f"Dry run finished. {len(plan)} file(s) would be moved.")
    else:
        print(f"Archiving finished. Moved {len(plan)} file(s).")


def main() -> None:
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir)

    if not checkpoints_dir.exists():
        print(f"Checkpoint directory does not exist: {checkpoints_dir}")
        return

    if not checkpoints_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {checkpoints_dir}")

    plan = build_archive_plan(checkpoints_dir)
    execute_archive(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
