"""Constrained GitHub REST client for a single repository."""

from __future__ import annotations

import json
import re
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .models import PRContext


class GitHubError(Exception):
    """GitHub returned an error, malformed data, or an incomplete response."""


Transport = Callable[[str, str, dict[str, str], bytes | None], object]
_MAX_RESPONSE_BYTES = 40 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> object:
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with build_opener(_NoRedirect()).open(request, timeout=30) as response:
            data = response.read(_MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise GitHubError(f"GitHub request failed: {exc}") from exc
    if len(data) > _MAX_RESPONSE_BYTES:
        raise GitHubError("GitHub response exceeded the size limit")
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise GitHubError("GitHub response is not valid JSON") from exc


def _string(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise GitHubError(f"Missing or invalid {name}")
    return value


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise GitHubError(f"Missing or invalid {name}")
    return value


class GitHubClient:
    """Only sends authenticated requests to this repository on api.github.com.

    A supplied transport receives (method, absolute_url, headers, body) and
    returns a decoded JSON value. Redirects must not be followed by transports.
    """

    def __init__(self, repository: str, token: str, *, transport: Transport | None = None) -> None:
        if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise GitHubError("Invalid repository name")
        if any(part in {".", ".."} for part in repository.split("/")):
            raise GitHubError("Invalid repository name")
        if not isinstance(token, str) or not token or "\n" in token or "\r" in token:
            raise GitHubError("Invalid GitHub token")
        self.repository = repository
        self._token = token
        self._transport = transport or _http_transport
        self._prefix = f"/repos/{repository}"

    def _request(self, method: str, path: str, payload: dict | None = None) -> object:
        if not isinstance(path, str):
            raise GitHubError("Invalid GitHub API path")
        parsed = urlsplit(path)
        decoded = unquote(parsed.path)
        if (parsed.scheme or parsed.netloc or parsed.fragment or decoded != parsed.path
                or "\\" in decoded or any(part in {".", ".."} for part in decoded.split("/"))
                or not (decoded == self._prefix or decoded.startswith(self._prefix + "/"))
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in path)):
            raise GitHubError("GitHub API path must remain within the configured repository")
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {self._token}",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "submission-static-check"}
        body = None
        if payload is not None:
            try:
                body = json.dumps(payload, allow_nan=False).encode("utf-8")
            except (ValueError, TypeError) as exc:
                raise GitHubError("Invalid GitHub request payload") from exc
            headers["Content-Type"] = "application/json"
        try:
            return self._transport(method, "https://api.github.com" + path, headers, body)
        except GitHubError:
            raise
        except (OSError, TimeoutError, ValueError) as exc:
            raise GitHubError(f"GitHub request failed: {exc}") from exc

    def get(self, path: str) -> object:
        return self._request("GET", path)

    def post(self, path: str, payload: dict) -> object:
        if not isinstance(payload, dict):
            raise GitHubError("GitHub POST payload must be an object")
        return self._request("POST", path, payload)

    def get_pr_context(self, number: int) -> PRContext:
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise GitHubError("Invalid pull request number")
        pull = _mapping(self.get(f"{self._prefix}/pulls/{number}"), "pull request")
        repo = _mapping(self.get(self._prefix), "repository")
        user = _mapping(pull.get("user"), "pull request author")
        head = _mapping(pull.get("head"), "pull request head")
        base = _mapping(pull.get("base"), "pull request base")
        head_sha = _string(head.get("sha"), "head SHA")
        base_sha = _string(base.get("sha"), "base SHA")
        if not re.fullmatch(r"[0-9a-f]{40}", head_sha) or not re.fullmatch(r"[0-9a-f]{40}", base_sha):
            raise GitHubError("Invalid pull request SHA")
        body = pull.get("body")
        if body is None:
            body = ""
        return PRContext(repository=self.repository, number=number,
                         author=_string(user.get("login"), "author login"), head_sha=head_sha,
                         base_sha=base_sha, title=_string(pull.get("title"), "title", empty=True),
                         body=_string(body, "body", empty=True),
                         default_branch=_string(repo.get("default_branch"), "default branch"),
                         base_ref=_string(base.get("ref"), "base branch"),
                         updated_at=_string(pull.get("updated_at"), "updated_at"))
