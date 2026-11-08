"""Syntactic PR checks; no identity registry or checklist validation."""

from __future__ import annotations

import posixpath
import re

from .models import Issue
from .policy import LAB_TITLES
from .readme import _MARKDOWN, _has_content, _links, _resolved_link, _sections


_TITLE = re.compile(
    r"(?P<surname>[A-Za-zЀ-ԯ]+(?:[-'’][A-Za-zЀ-ԯ]+)*)"
    r"[ \t]+(?P<initials>(?:[A-ZЀ-ԯ]\.){1,2})"
    r"[ \t]+(?P<group>[A-ZЀ-ԯ])[0-9]{4,6}\."
    r"[ \t]+Lab[ \t]+(?P<lab>[0-9]+):[ \t]+(?P<title>.+)"
)


def validate_pr(title: str, body: str, lab: int, submission_path: str) -> list[Issue]:
    """Check title syntax, required Markdown sections, and the report link."""
    issues: list[Issue] = []
    match = _TITLE.fullmatch(title.strip())
    identity_valid = bool(
        match
        and all(part.isalpha() for part in re.split(r"[-'’]", match.group("surname")))
        and match.group("initials").replace(".", "").isupper()
        and match.group("group").isupper()
    )
    if not match or not identity_valid:
        issues.append(
            Issue(
                "pr",
                "Заголовок PR не соответствует формату «Фамилия И.О. M4101. Lab N: Название»",
            )
        )
    else:
        if match.group("lab") != str(lab):
            issues.append(Issue("pr", f"Номер лабораторной в заголовке должен быть {lab}"))
        expected = LAB_TITLES.get(lab)
        if expected is None:
            issues.append(Issue("pr", f"Неизвестный номер лабораторной: {lab}"))
        elif match.group("title") != expected:
            issues.append(Issue("pr", f"Название лабораторной должно быть «{expected}»"))

    tokens = _MARKDOWN.parse(body)
    sections = {}
    for level in range(1, 7):
        for name, groups in _sections(tokens, level).items():
            sections.setdefault(name, []).extend(groups)
    for name in ("Что сделано", "Проверка", "Отчёт"):
        groups = sections.get(name, [])
        if not groups:
            issues.append(Issue("pr", f"Отсутствует раздел «{name}»"))
        elif not any(_has_content(group) for group in groups):
            issues.append(Issue("pr", f"Раздел «{name}» пуст или содержит заглушку"))

    expected_path = posixpath.join(submission_path, "README.md")
    if not any(
        _resolved_link(destination, "") == ("local", expected_path)
        for group in sections.get("Отчёт", [])
        for destination, _ in _links(group, include_images=False)
    ):
        issues.append(Issue("pr", f"Раздел «Отчёт» должен содержать относительную ссылку на {expected_path}"))
    return issues
