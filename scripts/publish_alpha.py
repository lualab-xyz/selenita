#!/usr/bin/env python
"""Squash-publish a Selenita tree snapshot to the public alpha repository.

Policy (see .context/decisions.md, D017):
- The public repo ``lualab-xyz/selenita`` (``alpha`` git remote) carries the
  distribution branch ``alpha-1``.
- Every publish creates ONE squashed commit (full tree sync of the tracked
  content of the source ref) on top of the previous ``alpha-1`` tip, so the
  public history reads one commit per release.
- The publish also pushes a release tag ``v<version>``; the version is the
  pyproject version suffixed with an alpha counter that auto-increments
  against the tags already present on the public repo (``0.1.0a1``,
  ``0.1.0a2``, ...).
- Internal-only content never ships: ``.context/``, ``docs/api/``,
  user scratch files, and local sandbox URLs/keys in the READMEs are
  replaced with placeholders in the public copy.

The private repo keeps a ``mirror/alpha-1`` branch that mirrors the last
published public state.

Usage:
    python scripts/publish_alpha.py                    # publish HEAD's branch
    python scripts/publish_alpha.py --source <branch>  # publish another ref
    python scripts/publish_alpha.py --dry-run          # plan only, no git writes
    python scripts/publish_alpha.py --skip-tag         # publish without a release tag
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

#: The public distribution repository (used directly as a git URL: this dev
#: environment blocks `git remote add` — CodePods proxy policy).
DEFAULT_REMOTE = "https://github.com/lualab-xyz/selenita.git"

#: Paths removed from the public copy (internal docs, research notes, scratch).
EXCLUSIONS: tuple[str, ...] = (
    ".context",
    "docs/api",
    "opencode-research",
    "Ejemplo.cs",
    "segundo_ejemplo.cs",
    "ascii_clock.py",
    "haikus.txt",
    "two_lines.txt",
)

#: (path relative to root, list of (pattern, replacement)) content transforms.
#: Local sandbox endpoints/keys in the dev READMEs become placeholders.
TRANSFORMS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "README.md",
        (
            (
                r"SELENITA_OPENAI_BASE_URL=http://host\.docker\.internal:3000/api/ai-proxy/\S+",
                "SELENITA_OPENAI_BASE_URL=https://your-llm-endpoint.example/v1",
            ),
            (r"SELENITA_OPENAI_API_KEY=123456", "SELENITA_OPENAI_API_KEY=sk-your-key"),
        ),
    ),
    (
        "README.es.md",
        (
            (
                r"SELENITA_OPENAI_BASE_URL=http://host\.docker\.internal:3000/api/ai-proxy/\S+",
                "SELENITA_OPENAI_BASE_URL=https://your-llm-endpoint.example/v1",
            ),
            (r"SELENITA_OPENAI_API_KEY=123456", "SELENITA_OPENAI_API_KEY=sk-your-key"),
        ),
    ),
)


def run(*args: str, check: bool = True, cwd: Path | None = None) -> str:
    """Run a git/shell command, return stdout (utf-8, stripped)."""
    proc = subprocess.run(args, check=False, capture_output=True, text=True, cwd=cwd)
    if check and proc.returncode != 0:
        raise SystemExit(
            f"error: {' '.join(args)} failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def run_bytes(*args: str) -> bytes:
    """Run a git command, return raw stdout bytes (for binary payloads)."""
    proc = subprocess.run(args, check=False, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"error: {' '.join(args)} failed (exit {proc.returncode}): {proc.stderr.decode()}"
        )
    return proc.stdout


def next_alpha_version(source_root: Path, tags: list[str]) -> str:
    """Compute the next ``<base>a<n>`` version for the publish."""
    pyproject = tomllib.loads((source_root / "pyproject.toml").read_text(encoding="utf-8"))
    base = re.fullmatch(r"(\d+\.\d+\.\d+)", str(pyproject["project"]["version"]))
    if not base:
        raise SystemExit(
            f"error: pyproject version is not a plain X.Y.Z: {pyproject['project']['version']}"
        )
    base_version = base[0]
    counter = 0
    for tag in tags:
        match = re.fullmatch(rf"v{re.escape(base_version)}a(\d+)", tag)
        if match:
            counter = max(counter, int(match[1]))
    return f"{base_version}a{counter + 1}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", default="HEAD", metavar="REF", help="private ref to publish (default HEAD)"
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help="public repo, URL or named remote (default: the public distribution URL)",
    )
    parser.add_argument(
        "--skip-tag", action="store_true", help="push the squash without creating a release tag"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force-replace the alpha-1 history (first publish or fix-ups; reuses are non-forced)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="plan only: no git writes, no pushes"
    )
    args = parser.parse_args(argv)

    repo_root = Path(run("git", "rev-parse", "--show-toplevel"))
    # Every git command below needs repository context — run from the root.
    os.chdir(repo_root)
    source = run("git", "rev-parse", "--abbrev-ref", args.source) or "HEAD"
    # Accept either a named remote or a full URL.
    remote_url = run("git", "remote", "get-url", args.remote, check=False)
    if not remote_url:
        remote_url = args.remote

    # Public channel state.
    branch_listing = run("git", "ls-remote", args.remote, "refs/heads/alpha-1", check=False)
    public_tip = branch_listing.split("\t")[0] or None
    tag_out = run("git", "ls-remote", "--tags", args.remote, "v*", check=False)
    public_tags = [
        line.split("refs/tags/")[1].split("^{}")[0]
        for line in tag_out.splitlines()
        if "refs/tags/" in line
    ]
    next_version = next_alpha_version(repo_root, public_tags)
    head_sha = run("git", "rev-parse", args.source)

    stage_dir = Path(tempfile.mkdtemp(prefix="selenita-publish-"))
    try:
        # 1. Extract the tracked tree of the source ref (untracked scratch never ships).
        #    git writes with a different uid in this dev environment, so the archive
        #    is streamed through stdout instead of a `-o` file.
        tree = stage_dir / "tree"
        tree.mkdir()
        archive_bytes = run_bytes("git", "archive", "--format=tar", args.source)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive_handle:
            archive_handle.extractall(tree, filter="data")

        # 2. Prune internal-only paths.
        excluded: list[str] = []
        for rel in EXCLUSIONS:
            victim = tree / rel
            if victim.is_dir():
                shutil.rmtree(victim)
            elif victim.exists():
                victim.unlink()
            else:
                continue
            excluded.append(rel)

        # 3. Apply content transforms to the public copy.
        transformed: list[str] = []
        for rel, rules in TRANSFORMS:
            path = tree / rel
            if path.exists():
                content = path.read_text(encoding="utf-8")
                for pattern, replacement in rules:
                    content = re.sub(pattern, replacement, content)
                path.write_text(content, encoding="utf-8")
                transformed.append(rel)

        # 4. Summarize what ships (squash message lists recent upstream commits).
        recent = run("git", "log", "--oneline", "-8", args.source).splitlines()
        print("== publish plan ==")
        print(f"source:    {source} @ {head_sha[:12]}")
        print(f"remote:    {args.remote} ({remote_url})")
        print(f"version:   {next_version} (public tags: {', '.join(public_tags) or 'none'})")
        print(f"public tip: {public_tip[:12] if public_tip else 'none (branch will be created)'}")
        print(f"excluded in the public copy: {', '.join(excluded) or 'none present'}")
        print(f"transformed: {', '.join(transformed) or 'none'}")
        print("squash commit will list upstream commits:")
        for line in recent:
            print(f"  {line}")

        if args.dry_run:
            print("dry-run: nothing published")
            return 0

        # 5. Materialize the squashed commit in a throwaway worktree inside the
        #    workspace, pre-created with wide permissions: in this dev environment the
        #    sandboxed git writes to the worktree with a different uid, and the
        #    worktree path must already exist for it to be accepted. The worktree's
        #    git operations also need `safe.directory` on the command line.

        worktree = repo_root / ".publish-wt"
        # Relative -C path only: in this dev environment the sandboxed git cannot
        # chdir into agent-side bind mounts (/workspace) — a relative worktree path
        # under the repo root resolves on both sides.
        GIT_WT = ("git", "-c", "safe.directory=*", "-C", ".publish-wt")  # noqa: N806
        # Start clean: stale registrations, stale dir, stale local mirror branch.
        run("git", "worktree", "remove", "--force", ".publish-wt", check=False)
        run("git", "worktree", "prune")
        shutil.rmtree(worktree, ignore_errors=True)
        run("git", "branch", "-D", "mirror/alpha-1", check=False)
        worktree.mkdir()
        os.chmod(worktree, 0o777)  # unmasked: the sandboxed git must write into it
        if public_tip:
            run("git", "fetch", "-q", args.remote, "+alpha-1:mirror/alpha-1")
            run(
                "git", "worktree", "add", "--force", "-B", "mirror/alpha-1",
                ".publish-wt", public_tip,
            )
        else:
            run(
                "git", "worktree", "add", "--force", "--orphan",
                "-b", "mirror/alpha-1", ".publish-wt",
            )

        try:
            # Replace the worktree content wholesale with the staged tree.
            for child in worktree.iterdir():
                if child.name != ".git":
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
            shutil.copytree(tree, worktree, dirs_exist_ok=True)

            (worktree / "install.sh").chmod(0o755)

            message_lines = [
                f"alpha-1: v{next_version} — squash publish of {source}",
                "",
                "Full tree sync from the private dev repo; recent upstream commits:",
                *(f"- {line}" for line in recent),
                "",
                "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
            ]
            run(*GIT_WT, "add", "-A")
            run(*GIT_WT, "commit", "-q", "-m", "\n".join(message_lines))
            published_sha = run(*GIT_WT, "rev-parse", "HEAD")

            # 6. Push branch first (the tag needs its parent commit), then the tag.
            push_args = ["git", "push"]
            if args.force:
                push_args.append("--force")
            run(*push_args, args.remote, "mirror/alpha-1:alpha-1")
            print(f"pushed alpha-1 → {published_sha[:12]}{' (forced)' if args.force else ''}")
            if not args.skip_tag:
                run("git", "tag", "-f", f"v{next_version}", published_sha)
                run("git", "push", args.remote, f"v{next_version}")
                print(f"pushed tag v{next_version} — release pipeline will build the assets")
        finally:
            run("git", "worktree", "remove", "--force", ".publish-wt", check=False)
            run("git", "worktree", "prune")
            run("chmod", "-R", "a+rwX", str(worktree), check=False)
            shutil.rmtree(worktree, ignore_errors=True)

        print("done")
        return 0
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
