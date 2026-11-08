"""Read an immutable, bounded pull request snapshot without executing its contents."""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass

from .github import GitHubClient, GitHubError
from .models import Issue, PRContext
from .policy import (
    FORBIDDEN_DIRECTORY_NAMES,
    FORBIDDEN_SUFFIXES,
    MAX_CHANGED_FILES,
    MAX_FILE_BYTES,
    MAX_SUBMISSION_BYTES,
    MAX_SUBMISSION_FILES,
    login_is_valid,
)


class StaleSnapshotError(GitHubError):
    """The pull request changed while it was being read."""


@dataclass
class Snapshot:
    context: PRContext
    submission_path: str | None
    lab: int | None
    files: dict[str, bytes]
    repository_files: set[str]
    issues: list[Issue]
    infrastructure: bool = False


# GitHub can return very large trees; even an untruncated response must be bounded.
_MAX_TREE_ENTRIES = 100_000
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_LAB_PATH = re.compile(r"^works/([^/]+)/lab([1-5])(?:/|$)")
_LAB_LIKE_PATH = re.compile(r"^works/[^/]+/lab[^/]*(?:/|$)")


def _path(value: object) -> str:
    if (not isinstance(value, str) or not value or value.startswith("/") or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise GitHubError("Invalid repository path in GitHub response")
    return value


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GitHubError(f"Missing or invalid {label}")
    return value


def _changes(client: GitHubClient, context: PRContext) -> list[str]:
    prefix = f"/repos/{context.repository}"
    pull = client.get(f"{prefix}/pulls/{context.number}")
    if not isinstance(pull, dict):
        raise GitHubError("Invalid pull request response")
    user, head, base = pull.get("user"), pull.get("head"), pull.get("base")
    if not all(isinstance(value, dict) for value in (user, head, base)):
        raise GitHubError("Invalid pull request identity or revisions")
    body = pull.get("body")
    if body is not None and not isinstance(body, str):
        raise GitHubError("Invalid pull request body")
    if (user.get("login") != context.author or head.get("sha") != context.head_sha
            or base.get("sha") != context.base_sha or base.get("ref") != context.base_ref
            or pull.get("title") != context.title or (body if body is not None else "") != context.body
            or pull.get("updated_at") != context.updated_at):
        raise StaleSnapshotError("Pull request changed before reading its file list")
    expected = _count(pull.get("changed_files"), "changed_files")
    if expected > MAX_CHANGED_FILES:
        raise GitHubError(f"Pull request exceeds {MAX_CHANGED_FILES} changed files")
    paths: list[str] = []
    names: set[str] = set()
    page = 1
    while True:
        batch = client.get(f"{prefix}/pulls/{context.number}/files?per_page=100&page={page}")
        if not isinstance(batch, list) or len(batch) > 100:
            raise GitHubError("Invalid pull request file page")
        for item in batch:
            if not isinstance(item, dict):
                raise GitHubError("Invalid changed file entry")
            name = _path(item.get("filename"))
            if name in names:
                raise GitHubError("Duplicate changed file entry")
            names.add(name)
            status = item.get("status")
            if status not in {"added", "removed", "modified", "renamed", "copied", "changed", "unchanged"}:
                raise GitHubError("Invalid changed file status")
            paths.append(name)
            if status in {"renamed", "copied"}:
                paths.append(_path(item.get("previous_filename")))
        if len(names) > expected or len(names) > MAX_CHANGED_FILES:
            raise GitHubError("Pull request file listing exceeds its declared count")
        if len(batch) < 100:
            break
        page += 1
    if len(names) != expected:
        raise GitHubError("Incomplete pull request file listing")
    return paths


def _tree(client: GitHubClient, context: PRContext) -> list[dict]:
    prefix = f"/repos/{context.repository}"
    commit = client.get(f"{prefix}/git/commits/{context.head_sha}")
    if not isinstance(commit, dict) or commit.get("sha") != context.head_sha:
        raise GitHubError("Invalid Git commit response")
    tree = commit.get("tree")
    tree_sha = tree.get("sha") if isinstance(tree, dict) else None
    if not isinstance(tree_sha, str) or not _SHA.fullmatch(tree_sha):
        raise GitHubError("Invalid Git commit tree SHA")
    result = client.get(f"{prefix}/git/trees/{tree_sha}?recursive=1")
    if isinstance(result, dict) and result.get("sha") != tree_sha:
        raise GitHubError("Git tree SHA does not match the commit")
    if not isinstance(result, dict) or result.get("truncated") is not False:
        raise GitHubError("Missing or truncated Git tree")
    entries = result.get("tree")
    if not isinstance(entries, list) or len(entries) > _MAX_TREE_ENTRIES:
        raise GitHubError("Invalid Git tree or tree entry budget exceeded")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise GitHubError("Invalid Git tree entry")
        path = _path(entry.get("path"))
        if path in seen:
            raise GitHubError("Duplicate Git tree path")
        seen.add(path)
        mode, kind = entry.get("mode"), entry.get("type")
        if (mode, kind) not in {("100644", "blob"), ("100755", "blob"), ("120000", "blob"),
                               ("040000", "tree"), ("160000", "commit")}:
            raise GitHubError(f"Invalid Git tree mode or type for {path}")
        sha = entry.get("sha")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise GitHubError(f"Invalid Git object SHA for {path}")
        if mode in {"100644", "100755", "120000"}:
            _count(entry.get("size"), f"size of {path}")
    return entries


def _blob(client: GitHubClient, context: PRContext, entry: dict) -> bytes:
    path, sha, size = entry["path"], entry["sha"], entry["size"]
    blob = client.get(f"/repos/{context.repository}/git/blobs/{sha}")
    if (not isinstance(blob, dict) or blob.get("encoding") != "base64" or blob.get("sha") != sha
            or _count(blob.get("size"), f"blob size of {path}") != size
            or not isinstance(blob.get("content"), str)):
        raise GitHubError(f"Invalid Git blob for {path}")
    encoded = blob["content"].replace("\n", "").replace("\r", "")
    if len(encoded) > ((size + 2) // 3) * 4:
        raise GitHubError(f"Invalid Git blob length for {path}")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GitHubError(f"Invalid Git blob encoding for {path}") from exc
    if len(data) != size:
        raise GitHubError(f"Incomplete Git blob for {path}")
    return data


def _compiled_binary(data: bytes) -> bool:
    """Recognize native build outputs even when their filename has no extension."""
    if data.startswith((b"\x7fELF", b"!<arch>\n")):
        return True
    if data[:4] in {b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}:
        return True
    if len(data) >= 64 and data.startswith(b"MZ"):
        offset = int.from_bytes(data[60:64], "little")
        return offset >= 64 and data[offset:offset + 4] == b"PE\x00\x00"
    return False


def load_snapshot(client: GitHubClient, number: int) -> Snapshot:
    context = client.get_pr_context(number)
    issues: list[Issue] = []
    files: dict[str, bytes] = {}
    repository_files: set[str] = set()
    submission_path: str | None = None
    lab: int | None = None
    infrastructure = False

    def issue(message: str, path: str = "") -> None:
        issues.append(Issue("structure", message, path))

    if context.base_ref != context.default_branch:
        issue("Pull request must target the repository default branch")
    valid_login = login_is_valid(context.author)
    if not valid_login:
        issue("Pull request author has an invalid GitHub login")
    paths = _changes(client, context)
    labs = [_LAB_PATH.match(path) for path in paths]
    matches = [match for match in labs if match]
    if not paths:
        issue("Pull request has no changed files")
    elif not matches:
        invalid_labs = [path for path in paths if _LAB_LIKE_PATH.match(path)]
        for path in invalid_labs:
            issue("Invalid laboratory directory; expected lab1 through lab5", path)
        if valid_login and not invalid_labs:
            permission = client.get(f"/repos/{context.repository}/collaborators/{context.author}/permission")
            if not isinstance(permission, dict) or not isinstance(permission.get("permission"), str):
                raise GitHubError("Invalid collaborator permission response")
            if permission["permission"] in {"write", "maintain", "admin"}:
                infrastructure = not issues
            else:
                issue("Infrastructure changes require write, maintain, or admin permission")
    else:
        first = matches[0]
        submission_path = f"works/{first.group(1)}/lab{first.group(2)}"
        lab = int(first.group(2))
        if first.group(1).casefold() != context.author.casefold():
            issue("Submission directory must match the pull request author", submission_path)
        for path in paths:
            if not path.startswith(submission_path + "/"):
                issue("Every changed path must belong to the same submission", path)

        entries = _tree(client, context)
        repository_files = {entry["path"] for entry in entries if entry["mode"] in {"100644", "100755"}}
        selected = [entry for entry in entries if entry["path"].startswith(submission_path + "/")]
        ordinary: list[dict] = []
        total = 0
        for entry in selected:
            path = entry["path"]
            relative = path[len(submission_path) + 1:]
            parts = relative.split("/")
            mode = entry["mode"]
            if any(part.casefold() in {name.casefold() for name in FORBIDDEN_DIRECTORY_NAMES}
                   for part in (parts if mode == "040000" else parts[:-1])):
                issue("Forbidden directory in submission", path)
            if mode in {"120000", "160000"}:
                issue("Symlinks and gitlinks are forbidden in a submission", path)
            elif mode in {"100644", "100755"}:
                ordinary.append(entry)
                size = entry["size"]
                total += size
                if size > MAX_FILE_BYTES:
                    issue(f"File exceeds {MAX_FILE_BYTES} bytes", path)
                if any(parts[-1].casefold().endswith(suffix.casefold()) for suffix in FORBIDDEN_SUFFIXES):
                    issue("Forbidden file type in submission", path)
        if len(ordinary) > MAX_SUBMISSION_FILES:
            issue(f"Submission exceeds {MAX_SUBMISSION_FILES} files", submission_path)
        if total > MAX_SUBMISSION_BYTES:
            issue(f"Submission exceeds {MAX_SUBMISSION_BYTES} bytes", submission_path)
        ordinary_paths = {entry["path"] for entry in ordinary}
        for required in ("README.md", "CMakeLists.txt"):
            if f"{submission_path}/{required}" not in ordinary_paths:
                issue(f"Missing required {required}", submission_path)
        if not any(path.startswith(f"{submission_path}/src/") for path in ordinary_paths):
            issue("Missing required src/ directory", submission_path)
        if not any(path.startswith(f"{submission_path}/src/") and path.endswith(".cu") for path in ordinary_paths):
            issue("Missing required .cu source file in src/", submission_path)
        if len(ordinary) <= MAX_SUBMISSION_FILES and total <= MAX_SUBMISSION_BYTES:
            for entry in ordinary:
                if entry["size"] <= MAX_FILE_BYTES:
                    data = _blob(client, context, entry)
                    if _compiled_binary(data):
                        issue("Compiled binary artifact is forbidden in a submission", entry["path"])
                    files[entry["path"]] = data

    after = client.get_pr_context(number)
    if after != context:
        raise StaleSnapshotError("Pull request changed while the snapshot was read")
    return Snapshot(context, submission_path, lab, files, repository_files, issues, infrastructure)
