"""Publish a head commit status without importing the validation runtime."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import re
import sys

from .diagnostics import category_title, emit_issues, redact_secrets
from .github import GitHubClient, GitHubError
from .models import Issue, PRContext, context_fingerprint

STATUS_CONTEXT = "submission/static"
MAX_FILE = 2 * 1024 * 1024
MAX_SUMMARY = 60000
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class ReportError(Exception):
    """Cannot safely publish a status."""


def _read_json(path: str) -> object:
    try:
        with open(path, "rb") as source:
            data = source.read(MAX_FILE + 1)
    except OSError as exc:
        raise ReportError(f"Не удалось прочитать {path}: {exc.strerror or exc}") from exc
    if len(data) > MAX_FILE:
        raise ReportError(f"Файл {path} превышает лимит {MAX_FILE} байт")
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise ReportError(f"JSON {path}: строка {exc.lineno}, столбец {exc.colno}: {exc.msg}") from exc
    except (UnicodeError, ValueError) as exc:
        raise ReportError(f"Некорректный JSON в {path}: {exc}") from exc


def _run_identity(repository: str) -> tuple[str, tuple[int, int]]:
    run = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not re.fullmatch(r"[1-9][0-9]*", run) or not re.fullmatch(r"[1-9][0-9]*", attempt):
        raise ReportError("Invalid workflow run identity")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    if server != "https://github.com":
        raise ReportError("Unsupported GitHub server")
    return f"{server}/{repository}/actions/runs/{run}/attempts/{attempt}", (int(run), int(attempt))


def _owner(url: object, repository: str) -> tuple[int, int] | None:
    if not isinstance(url, str):
        return None
    pattern = rf"https://github\.com/{re.escape(repository)}/actions/runs/([1-9][0-9]*)(?:/attempts/([1-9][0-9]*))?"
    match = re.fullmatch(pattern, url)
    return (int(match[1]), int(match[2] or "1")) if match else None


def _latest(client: GitHubClient, sha: str) -> dict | None:
    # The commit statuses endpoint is newest-first. Search all pages, not just 100 statuses.
    for page in range(1, 101):
        data = client.get(f"/repos/{client.repository}/commits/{sha}/statuses?per_page=100&page={page}")
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ReportError("Invalid status response")
        for item in data:
            if item.get("context") == STATUS_CONTEXT:
                return item
        if len(data) < 100:
            return None
    raise ReportError("Status history exceeds safety limit")


def _description(state: str, issues: list[dict]) -> str:
    defaults = {"pending": "Проверка сдачи выполняется", "success": "Структура, PR, README и форматирование проверены",
                "failure": "Есть замечания к сдаче", "error": "Ошибка CI: подробности в отчёте запуска"}
    if not issues:
        return defaults[state]
    first = issues[0]
    location = first['path']
    if location and first['line'] is not None:
        location += f":{first['line']}"
    title = category_title(first['category'])
    if location:
        title += f" {location}"
    text = f"{title}: {first['message']}"
    return " ".join(redact_secrets(text).split())[:140]


def _post(client: GitHubClient, sha: str, state: str, url: str, description: str | None = None) -> None:
    client.post(f"/repos/{client.repository}/statuses/{sha}",
                {"state": state, "context": STATUS_CONTEXT, "target_url": url,
                 "description": description if description is not None else _description(state, [])})


def _state(path: str, client: GitHubClient, number: int) -> tuple[PRContext, str, str]:
    obj = _read_json(path)
    if not isinstance(obj, dict) or type(obj.get("schema_version")) is not int or obj["schema_version"] != 1:
        raise ReportError(f"Некорректный файл состояния {path}")
    try:
        context = PRContext(**obj["context"])
        fingerprint, url = obj["fingerprint"], obj["run_url"]
        if (context.repository != client.repository or context.number != number
                or not isinstance(context.head_sha, str) or not HEX40.fullmatch(context.head_sha)
                or not isinstance(fingerprint, str) or not HEX64.fullmatch(fingerprint)
                or context_fingerprint(context) != fingerprint
                or url != _run_identity(client.repository)[0]):
            raise ReportError(f"Файл состояния {path} относится к другому PR или запуску")
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportError(f"Некорректный файл состояния {path}") from exc
    return context, fingerprint, url


def _stale(message: str) -> int:
    issue = Issue("pr", message)
    _summary("stale", [asdict(issue)])
    if os.environ.get("GITHUB_ACTIONS") == "true":
        emit_issues([issue], stage="Устаревшая проверка")
    return 3


def pending(client: GitHubClient, number: int, state_file: str) -> int:
    context = client.get_pr_context(number)
    if not HEX40.fullmatch(context.head_sha):
        raise ReportError("Invalid head SHA")
    fingerprint = context_fingerprint(context)
    url, identity = _run_identity(client.repository)
    latest = _latest(client, context.head_sha)
    if latest is not None:
        owner = _owner(latest.get("target_url"), client.repository)
        if owner is None or owner >= identity:
            return _stale("Статус этого коммита уже принадлежит другому или более новому запуску.")
    if context_fingerprint(client.get_pr_context(number)) != fingerprint:
        return _stale("Коммит или описание PR изменились во время подготовки проверки.")
    latest = _latest(client, context.head_sha)
    if latest is not None:
        owner = _owner(latest.get("target_url"), client.repository)
        if owner is None or owner >= identity:
            return _stale("Статус этого коммита уже принадлежит другому или более новому запуску.")
    _post(client, context.head_sha, "pending", url)
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    Path(state_file).write_text(json.dumps({"schema_version": 1, "context": asdict(context),
                                            "fingerprint": fingerprint, "run_url": url}), encoding="utf-8")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as out:
            out.write(f"head_sha={context.head_sha}\nfingerprint={fingerprint}\n")
    return 0


def _result(path: str, context: PRContext, fingerprint: str, outcome: str) -> tuple[str, list[dict]]:
    try:
        if outcome not in {"success", "failure", "cancelled", "skipped"}:
            raise ReportError(f"Неизвестный результат job: {outcome}")
        data = _read_json(path)
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ReportError("В файле результата ожидается schema_version=1")
        state = data.get("state")
        if state == "stale":
            return "stale", []
        issues = data.get("issues")
        if (state not in {"success", "failure", "error"}
                or not isinstance(issues, list)
                or any(not isinstance(i, dict) or not isinstance(i.get("category"), str)
                       or not isinstance(i.get("message"), str) or not isinstance(i.get("path"), str)
                       or "line" not in i
                       or (i.get("line") is not None and (type(i.get("line")) is not int or i["line"] <= 0))
                       for i in issues)
                or "submission_path" not in data or not isinstance(data["submission_path"], (str, type(None)))
                or "lab" not in data or (data["lab"] is not None and type(data["lab"]) is not int)
                or type(data.get("infrastructure")) is not bool):
            raise ReportError("Некорректные поля или список замечаний в результате проверки")
        early_error = state == "error" and data.get("context") is None and data.get("fingerprint") == ""
        if not early_error and (data.get("context") != asdict(context) or data.get("fingerprint") != fingerprint):
            raise ReportError("Результат относится к другому коммиту или описанию PR")
        if state == "success" and outcome == "success" and not issues:
            if data["infrastructure"] or (data["submission_path"] and data["lab"] in range(1, 6)):
                return "success", issues
            raise ReportError("В успешном результате отсутствуют путь или номер лабораторной")
        if state == "failure" and outcome == "failure" and issues and all(i["category"] != "tool" for i in issues):
            return "failure", issues
        if state == "error" and issues:
            return "error", issues
        reason = f"Результат {state!r} не согласован с job {outcome!r} или не содержит причины ошибки"
        return "error", issues + [asdict(Issue("tool", reason, path))]
    except ReportError as exc:
        message = f"Этап «Проверка сдачи» завершился со статусом {outcome}. {exc}"
        return "error", [asdict(Issue("tool", redact_secrets(message), path))]


def _summary(state: str, issues: list[dict]) -> None:
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if not destination:
        return
    state_titles = {"success": "Проверка пройдена", "failure": "Есть замечания",
                    "error": "Ошибка CI", "stale": "Результат устарел"}
    lines = ["## Результат проверки", "", f"**{state_titles.get(state, state)}** ({state})", "",
             "Структура, оформление PR, README, воспроизводимость и форматирование.", ""]
    remaining = MAX_SUMMARY - 1000
    for issue in issues:
        location = issue['path']
        if location and issue['line'] is not None:
            location += f":{issue['line']}"
        title = category_title(issue['category'])
        if location:
            title += f" — {location}"
        title = html.escape(redact_secrets(title), quote=True)
        message = html.escape(redact_secrets(issue['message']), quote=True)
        block = f"<p><strong>{title}</strong></p><pre>{message}</pre>"
        if len(block) > remaining:
            lines.append("<p>Остальные замечания приведены в файле результата проверки.</p>")
            break
        lines.append(block)
        remaining -= len(block) + 1
    try:
        with open(destination, "a", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")
    except OSError as exc:
        raise ReportError(f"Не удалось записать отчёт {destination}: {exc}") from exc


def finish(client: GitHubClient, number: int, state_file: str, result_file: str, outcome: str) -> int:
    context, fingerprint, url = _state(state_file, client, number)
    state, issues = _result(result_file, context, fingerprint, outcome)
    if context_fingerprint(client.get_pr_context(number)) != fingerprint or state == "stale":
        return _stale("Коммит или описание PR изменились после начала проверки.")
    latest = _latest(client, context.head_sha)
    if not latest or latest.get("target_url") != url or latest.get("state") != "pending":
        return _stale("Текущий статус PR уже обновлён другим запуском.")
    _post(client, context.head_sha, state, url, _description(state, issues))
    _summary(state, issues)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        emit_issues([Issue(i['category'], i['message'], i['path'], i['line']) for i in issues],
                    stage="Результат проверки")
    return {"success": 0, "failure": 1, "error": 2}[state]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("pending", "finish"):
        child = sub.add_parser(command)
        child.add_argument("--pr-number", type=int, required=True)
        child.add_argument("--state-file", required=True)
        if command == "finish":
            child.add_argument("--result-file", required=True)
            child.add_argument("--validation-outcome", required=True, choices=("success", "failure", "cancelled", "skipped"))
    args = parser.parse_args(argv)
    try:
        if args.pr_number <= 0:
            raise ReportError("Invalid pull request number")
        client = GitHubClient(os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_TOKEN", ""))
        if args.command == "pending":
            return pending(client, args.pr_number, args.state_file)
        return finish(client, args.pr_number, args.state_file, args.result_file, args.validation_outcome)
    except (GitHubError, ReportError, OSError) as exc:
        stage = "Подготовка PR" if args.command == "pending" else "Публикация результата"
        issue = Issue("tool", redact_secrets(f"{stage}: {exc}"))
        emit_issues([issue], stream=sys.stderr)
        try:
            _summary("error", [asdict(issue)])
        except ReportError as summary_error:
            emit_issues([Issue("tool", str(summary_error))], stage="Запись отчёта", stream=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
