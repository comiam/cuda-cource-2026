"""Render actionable findings in logs and GitHub file annotations."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from typing import TextIO

from .models import Issue

MAX_MESSAGE_CHARS = 3000
MAX_DIAGNOSTICS = 100
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TITLES = {
    "structure": "Структура папки",
    "pr": "Оформление PR",
    "readme": "README",
    "reproducibility": "Воспроизводимость",
    "formatting": "Форматирование",
    "tool": "Ошибка проверки",
}


def redact_secrets(text: str) -> str:
    """Keep diagnostic text printable and remove known authentication tokens."""
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return _CONTROL.sub("", text)


def category_title(category: str) -> str:
    return _TITLES.get(category, category)


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def emit_issues(
    issues: Iterable[Issue], *, stage: str = "", stream: TextIO | None = None
) -> None:
    """Print each cause locally, or annotate it safely in GitHub Actions."""
    destination = stream if stream is not None else sys.stdout
    github = os.environ.get("GITHUB_ACTIONS") == "true"
    for index, issue in enumerate(issues):
        if index >= MAX_DIAGNOSTICS:
            notice = "Остальные замечания приведены в отчёте проверки."
            print(f"::warning::{notice}" if github else notice, file=destination)
            break
        title = redact_secrets(category_title(issue.category))
        if stage:
            title = f"{stage}: {title}"
        message = redact_secrets(issue.message)
        if len(message) > MAX_MESSAGE_CHARS:
            message = message[:MAX_MESSAGE_CHARS] + "\n…"
        path = redact_secrets(issue.path)
        line = issue.line if type(issue.line) is int and issue.line > 0 else None
        if github:
            properties = []
            if path:
                properties.append(f"file={_escape_property(path)}")
                if line is not None:
                    properties.append(f"line={line}")
            properties.append(f"title={_escape_property(title)}")
            print(f"::error {','.join(properties)}::{_escape_data(message)}", file=destination)
        else:
            location = f" {path}" if path else ""
            if path and line is not None:
                location += f":{line}"
            print(f"[{title}]{location}: {message.replace(chr(13), '<CR>')}", file=destination)
