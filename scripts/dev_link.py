#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Link this working tree into a Hermes profile for development.

For normal use, install from GitHub instead::

    hermes plugins install hermes-pro/mem0 --no-enable

That clones a snapshot, so it can't see uncommitted edits. This script points
``$HERMES_HOME/plugins/mem0_hermes`` at *this checkout* — by symlink, or a
Windows junction when symlinks need privileges, falling back to a copy — so
edits take effect in the next session with no reinstall.

    python scripts/dev_link.py                 # link into $HERMES_HOME/plugins
    python scripts/dev_link.py --activate      # also set memory.provider
    python scripts/dev_link.py --copy          # copy instead of linking
    python scripts/dev_link.py --uninstall     # remove the link/copy again

Nothing is deleted without either ``--force`` or the target being a link this
script created; run with ``--dry-run`` to see the plan first.

The destination directory name is fixed at ``mem0_hermes`` (the ``name`` in
plugin.yaml) regardless of what this checkout is called — Hermes resolves
``memory.provider`` by directory name, and bundled providers win collisions, so
a directory named ``mem0`` would be shadowed by the bundled Mem0 plugin.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_NAME = "mem0_hermes"
# The plugin lives at the repo root (so `hermes plugins install owner/repo`
# finds plugin.yaml there); this script sits one level down in scripts/.
SOURCE_DIR = Path(__file__).resolve().parent.parent

# A --copy install must not drag in the gitignored hermes-agent checkout (a
# whole second repo) or this repo's git history.
_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.py[cod]", ".git", "hermes-agent", ".venv", "venv",
)


def resolve_hermes_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def _is_link(path: Path) -> bool:
    """True for symlinks and for Windows junctions (which aren't symlinks)."""
    if path.is_symlink():
        return True
    try:  # junctions report ST_REPARSE_POINT
        return bool(os.stat(path, follow_symlinks=False).st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _norm(path) -> str:
    """Comparable path string.

    Windows junctions read back with the ``\\\\?\\`` extended-length prefix, so
    a plain equality check against the source directory would never match and
    an installed link would look like a stranger's.
    """
    text = os.path.realpath(str(path))
    if text.startswith("\\\\?\\"):
        text = text[4:]
    return os.path.normcase(text)


def _link_target(path: Path) -> str:
    try:
        target = os.readlink(path)
    except OSError:
        target = str(path)
    text = os.path.realpath(target)
    return text[4:] if text.startswith("\\\\?\\") else text


def _looks_like_this_plugin(path: Path) -> bool:
    manifest = path / "plugin.yaml"
    if not manifest.is_file():
        return False
    try:
        return f"name: {PLUGIN_NAME}" in manifest.read_text(encoding="utf-8")
    except OSError:
        return False


def _make_link(source: Path, dest: Path) -> str:
    """Create a directory link at ``dest``. Returns the method used."""
    try:
        os.symlink(source, dest, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError) as exc:
        if os.name != "nt":
            raise RuntimeError(f"could not create symlink: {exc}") from exc
    # Windows without Developer Mode / admin: a junction needs no privilege.
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dest), str(source)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(
            "could not create a symlink or junction "
            f"({result.stderr.strip() or 'unknown error'}); re-run with --copy"
        )
    return "junction"


def _remove(path: Path) -> None:
    if _is_link(path):
        try:
            path.unlink()
            return
        except OSError:
            os.rmdir(path)  # junctions on some Windows builds
            return
    shutil.rmtree(path)


def install(dest_parent: Path, *, copy: bool, force: bool, dry_run: bool) -> Path:
    dest = dest_parent / PLUGIN_NAME
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"error: plugin source not found at {SOURCE_DIR}")

    if dest.exists() or _is_link(dest):
        if _is_link(dest):
            target = _link_target(dest)
            if _norm(target) == _norm(SOURCE_DIR):
                print(f"already installed: {dest} -> {target}")
                return dest
            what = f"link to {target}"
        elif _looks_like_this_plugin(dest):
            what = "an existing copy of this plugin"
        else:
            what = "an unrelated directory"
        if not force:
            raise SystemExit(
                f"error: {dest} exists ({what}).\n"
                "       Re-run with --force to replace it, or --uninstall first."
            )
        print(f"replacing {dest} ({what})")
        if not dry_run:
            _remove(dest)

    print(f"{'would install' if dry_run else 'installing'}: {dest} <- {SOURCE_DIR}")
    if dry_run:
        return dest

    dest_parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(SOURCE_DIR, dest, ignore=_COPY_IGNORE)
        method = "copy"
    else:
        try:
            method = _make_link(SOURCE_DIR, dest)
        except RuntimeError as exc:
            print(f"note: {exc}\n      falling back to a copy")
            shutil.copytree(SOURCE_DIR, dest, ignore=_COPY_IGNORE)
            method = "copy"
    print(f"installed via {method}")
    return dest


def uninstall(dest_parent: Path, *, force: bool, dry_run: bool) -> None:
    dest = dest_parent / PLUGIN_NAME
    if not (dest.exists() or _is_link(dest)):
        print(f"nothing to remove at {dest}")
        return
    if not (_is_link(dest) or _looks_like_this_plugin(dest) or force):
        raise SystemExit(
            f"error: {dest} does not look like this plugin; refusing to delete it "
            "(use --force to override)"
        )
    print(f"{'would remove' if dry_run else 'removing'}: {dest}")
    if not dry_run:
        _remove(dest)


def activate(provider: str, dry_run: bool) -> None:
    """Set ``memory.provider`` in config.yaml using Hermes's own config writer."""
    try:
        from hermes_cli.config import load_config, save_config
    except ImportError:
        print(
            "note: --activate needs hermes-agent importable (run this with the "
            "Hermes venv interpreter).\n"
            f"      Set `memory: {{provider: {provider}}}` in config.yaml, or run "
            "`hermes memory setup`."
        )
        return
    config = load_config()
    memory = config.get("memory")
    if not isinstance(memory, dict):
        memory = {}
        config["memory"] = memory
    previous = memory.get("provider")
    if previous == provider:
        print(f"memory.provider already set to {provider}")
        return
    print(
        f"{'would set' if dry_run else 'setting'} memory.provider: "
        f"{previous or '<unset>'} -> {provider}"
    )
    if dry_run:
        return
    memory["provider"] = provider
    save_config(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hermes-home", help="override HERMES_HOME")
    parser.add_argument("--copy", action="store_true", help="copy instead of linking")
    parser.add_argument(
        "--activate", action="store_true", help="set memory.provider in config.yaml"
    )
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="replace/remove an existing destination"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    home = resolve_hermes_home(args.hermes_home)
    if not home.exists():
        print(f"warning: {home} does not exist yet (run `hermes setup` first?)")
    plugins_dir = home / "plugins"
    print(f"HERMES_HOME: {home}")

    if args.uninstall:
        uninstall(plugins_dir, force=args.force, dry_run=args.dry_run)
        print("\nRemember to point memory.provider elsewhere in config.yaml.")
        return 0

    install(plugins_dir, copy=args.copy, force=args.force, dry_run=args.dry_run)
    if args.activate:
        activate(PLUGIN_NAME, args.dry_run)

    print(
        "\nNext steps:\n"
        "  1. hermes memory setup        # pick mem0_hermes, choose an embedder\n"
        f"  2. confirm `memory.provider: {PLUGIN_NAME}` in config.yaml\n"
        "  3. start a new session; memory extraction now runs on your Hermes model"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
