"""Static checks for the main submission README; never execute its contents."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterator
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token

from .models import Issue
from .policy import (
    PROFILE_SECTION,
    REPRODUCTION_SUBSECTIONS,
    REQUIRED_README_SECTIONS,
    TRAINING_REPRODUCTION_SUBSECTIONS,
)


_MARKDOWN = MarkdownIt("commonmark")
_SEED = re.compile(r"\b(?:seed|сид|зерно)\b", re.IGNORECASE)
_PLACEHOLDER = re.compile(
    r"^(?:todo|tbd|fixme|заполнить|указать|описать|добавить|заглушка|"
    r"нет|n/?a|\.\.\.|…|<[^>]+>)(?:[\s.!?:…-]*)$",
    re.IGNORECASE,
)


def _inline_text(token: Token) -> str:
    """Return rendered text, excluding inline HTML comments and markup."""
    return " ".join(
        child.content
        for child in token.children or []
        if child.type in {"text", "code_inline", "image"}
    )


def _sections(tokens: list[Token], level: int) -> dict[str, list[list[Token]]]:
    """Collect top-level sections through the next heading of equal/higher rank."""
    result: dict[str, list[list[Token]]] = {}
    starts = [
        i
        for i, token in enumerate(tokens)
        if token.type == "heading_open" and token.level == 0
    ]
    for position, start in enumerate(starts):
        token = tokens[start]
        if token.tag != f"h{level}":
            continue
        end = len(tokens)
        for next_start in starts[position + 1 :]:
            if int(tokens[next_start].tag[1:]) <= level:
                end = next_start
                break
        name = _inline_text(tokens[start + 1]).strip()
        result.setdefault(name, []).append(tokens[start + 3 : end])
    return result


def _content(tokens: list[Token]) -> str:
    parts: list[str] = []
    for index, token in enumerate(tokens):
        if token.type == "inline":
            # A heading is not evidence that its parent contains a description.
            if index and tokens[index - 1].type == "heading_open":
                continue
            parts.append(re.sub(r"^\s*\[[ xX]\]\s*", "", _inline_text(token)))
        elif token.type in {"fence", "code_block"}:
            parts.append(token.content)
    return " ".join(parts).strip()


def _has_content(tokens: list[Token]) -> bool:
    content = _content(tokens).strip()
    return bool(content and not _PLACEHOLDER.fullmatch(content))


def _has_command(tokens: list[Token]) -> bool:
    return any(
        token.type in {"fence", "code_block"} and _has_content([token])
        for token in tokens
    )


def _has_seed_description(tokens: list[Token]) -> bool:
    content = _content(tokens)
    if not _SEED.search(content):
        return False
    rest = _SEED.sub("", content).strip(" \t\n\r:;,.!?=-–—`'")
    return bool(rest and not _PLACEHOLDER.fullmatch(rest))


def _resolved_link(destination: str, base_dir: str) -> tuple[str, str]:
    """Return (kind, path); reject paths that escape the repository root."""
    try:
        parsed = urlsplit(destination)
    except ValueError:
        return "invalid", ""
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return "external", ""
    if parsed.scheme or parsed.netloc:
        return "invalid", ""
    path = unquote(parsed.path)
    if not path:
        return ("anchor", "") if parsed.fragment else ("invalid", "")
    if path.startswith("/") or "\\" in path or "\x00" in path:
        return "invalid", ""
    parts = base_dir.split("/") if base_dir else []
    for part in path.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return "invalid", ""
            parts.pop()
        else:
            parts.append(part)
    return "local", "/".join(parts)


class _HtmlLinks(HTMLParser):
    def __init__(self, include_images: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.include_images = include_images
        self.destinations: list[tuple[str, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        key = "href" if tag == "a" else "src" if tag == "img" and self.include_images else None
        if key is not None:
            for name, value in attrs:
                if name == key and value is not None:
                    self.destinations.append((value, self.getpos()[0]))


def _html_links(
    text: str, line: int | None, include_images: bool
) -> Iterator[tuple[str, int | None]]:
    parser = _HtmlLinks(include_images)
    parser.feed(text)
    for destination, offset in parser.destinations:
        yield destination, line + offset - 1 if line is not None else None


def _links(
    tokens: list[Token], *, include_images: bool = True
) -> Iterator[tuple[str, int | None]]:
    """Yield destinations and one-based source lines for Markdown links/images."""
    for token in tokens:
        line = token.map[0] + 1 if token.map else None
        if token.type == "html_block":
            yield from _html_links(token.content, line, include_images)
        elif token.type == "inline":
            for child in token.children or []:
                if child.type == "link_open":
                    yield child.attrGet("href") or "", line
                elif include_images and child.type == "image":
                    yield child.attrGet("src") or "", line
                elif child.type == "html_inline":
                    yield from _html_links(child.content, line, include_images)


def validate_readme(
    text: str, lab: int, submission_path: str, repository_files: set[str]
) -> list[Issue]:
    """Validate Markdown structure and repository-relative links, without I/O."""
    path = posixpath.join(submission_path, "README.md")
    tokens = _MARKDOWN.parse(text)
    sections = _sections(tokens, 2)
    issues: list[Issue] = []

    required = list(REQUIRED_README_SECTIONS)
    if 3 <= lab <= 5:
        required.append(PROFILE_SECTION)
    for name in required:
        groups = sections.get(name, [])
        category = "reproducibility" if name in {"Окружение", "Воспроизводимость"} else "readme"
        if not groups:
            issues.append(Issue(category, f"Отсутствует раздел H2 «{name}»", path))
        elif not any(_has_content(group) for group in groups):
            issues.append(Issue(category, f"Раздел «{name}» пуст или содержит заглушку", path))
        if name in {"Компиляция", "Запуск"} and groups and not any(
            _has_command(group) for group in groups
        ):
            issues.append(Issue("readme", f"В разделе «{name}» нужен блок команд", path))

    reproduction = sections.get("Воспроизводимость", [])
    subsections: dict[str, list[list[Token]]] = {}
    for group in reproduction:
        for name, entries in _sections(group, 3).items():
            subsections.setdefault(name, []).extend(entries)
    required_subsections = list(REPRODUCTION_SUBSECTIONS)
    if lab in {4, 5}:
        required_subsections.extend(TRAINING_REPRODUCTION_SUBSECTIONS)
    for name in required_subsections:
        groups = subsections.get(name, [])
        if not groups:
            issues.append(Issue("reproducibility", f"Отсутствует подраздел H3 «{name}»", path))
        elif not any(_has_content(group) for group in groups):
            issues.append(Issue("reproducibility", f"Подраздел «{name}» пуст или содержит заглушку", path))
        if name == "Полный запуск" and groups and not any(_has_command(group) for group in groups):
            issues.append(Issue("reproducibility", "В подразделе «Полный запуск» нужен блок команд", path))
        if name == "Параметры обучения" and groups and not any(
            _has_seed_description(group) for group in groups
        ):
            issues.append(Issue("reproducibility", "В подразделе «Параметры обучения» нужно описать seed", path))

    for destination, line in _links(tokens):
        kind, resolved = _resolved_link(destination, submission_path)
        if kind == "invalid" or (kind == "local" and resolved not in repository_files):
            issues.append(Issue("readme", f"Недопустимая или отсутствующая ссылка: {destination}", path, line))
    return issues
