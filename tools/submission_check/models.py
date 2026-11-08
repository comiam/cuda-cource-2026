"""Small shared value objects for independent checks."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Issue:
    """A finding about the submission, not an execution verdict."""

    category: str
    message: str
    path: str = ""
    line: int | None = None


@dataclass(frozen=True, slots=True)
class PRContext:
    """An immutable snapshot of PR identity and metadata."""

    repository: str
    number: int
    author: str
    head_sha: str
    base_sha: str
    title: str
    body: str
    default_branch: str
    base_ref: str
    updated_at: str = ""


def context_fingerprint(context: PRContext) -> str:
    """Bind a result to the head, base, author, title and body, not just a commit."""
    canonical = json.dumps(asdict(context), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
