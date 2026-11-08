"""Orchestrate immutable CI snapshots and bounded local submissions."""
from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .diagnostics import redact_secrets
from .formatting import format_submission
from .github import GitHubClient, GitHubError
from .models import Issue, PRContext, context_fingerprint
from .policy import (FORBIDDEN_DIRECTORY_NAMES, FORBIDDEN_SUFFIXES, MAX_FILE_BYTES,
                     MAX_SUBMISSION_BYTES, MAX_SUBMISSION_FILES, login_is_valid)
from .pr import validate_pr
from .readme import validate_readme
from .snapshot import StaleSnapshotError, _compiled_binary, load_snapshot

TRUSTED_ROOT = Path(__file__).resolve().parents[2]


def _result(
    context: PRContext | None = None, submission_path: str | None = None,
    lab: int | None = None, infrastructure: bool = False,
) -> dict[str, Any]:
    return {'schema_version': 1, 'state': 'error', 'context': asdict(context) if context else None,
            'fingerprint': context_fingerprint(context) if context else '', 'issues': [],
            'submission_path': submission_path, 'lab': lab, 'infrastructure': infrastructure}


def _finish(result: dict[str, Any], issues: list[Issue]) -> dict[str, Any]:
    result['issues'] = [asdict(issue) for issue in issues]
    result['state'] = 'error' if any(i.category == 'tool' for i in issues) else 'failure' if issues else 'success'
    return result


def _student_issues(files, repo_files, path, lab, formatter, root):
    issues = []
    readme = files.get(path + '/README.md')
    if readme is None:
        issues.append(Issue('structure', 'Missing required README.md', path))
    else:
        try:
            text = readme.decode('utf-8')
        except UnicodeDecodeError:
            issues.append(Issue('readme', 'README.md is not valid UTF-8', path + '/README.md'))
        else:
            issues.extend(validate_readme(text, lab, path, repo_files))
    format_files = files
    if readme is not None:
        try:
            readme.decode('utf-8')
        except UnicodeDecodeError:
            format_files = {name: data for name, data in files.items() if name != path + '/README.md'}
    try:
        issues.extend(formatter(format_files, path, root))
    except Exception as exc:
        message = redact_secrets(f'Форматирование ({type(exc).__name__}): {exc}')
        issues.append(Issue('tool', message[:3000], path))
    return issues


def run_ci(
    number: int, expected_head_sha: str, expected_fingerprint: str, *,
    client: GitHubClient | None = None,
    format_function: Callable[[dict[str, bytes], str, Path], list[Issue]] | None = None,
    trusted_root: Path = TRUSTED_ROOT,
) -> dict[str, Any]:
    result = _result()
    try:
        if client is None:
            client = GitHubClient(os.environ.get('GITHUB_REPOSITORY', ''), os.environ.get('GITHUB_TOKEN', ''))
        snapshot = load_snapshot(client, number)
        result = _result(snapshot.context, snapshot.submission_path, snapshot.lab, snapshot.infrastructure)
        if snapshot.context.head_sha != expected_head_sha or result['fingerprint'] != expected_fingerprint:
            result['state'] = 'stale'
            result['issues'] = [asdict(Issue('tool', 'Snapshot does not match expected PR context'))]
            return result
        issues = list(snapshot.issues)
        if not snapshot.infrastructure and snapshot.submission_path is not None and snapshot.lab is not None:
            issues.extend(validate_pr(snapshot.context.title, snapshot.context.body, snapshot.lab, snapshot.submission_path))
            issues.extend(_student_issues(snapshot.files, snapshot.repository_files, snapshot.submission_path,
                                          snapshot.lab, format_function or format_submission, Path(trusted_root)))
        return _finish(result, issues)
    except StaleSnapshotError:
        result['state'] = 'stale'
        result['issues'] = [asdict(Issue('tool', 'Pull request changed during snapshot capture'))]
    except GitHubError as exc:
        result['state'] = 'error'
        result['issues'] = [asdict(Issue('tool', str(exc)[:512]))]
    except Exception as exc:
        result['state'] = 'error'
        message = redact_secrets(f'Проверка PR #{number} ({type(exc).__name__}): {exc}')
        result['issues'] = [asdict(Issue('tool', message[:3000]))]
    return result


def _local_files(root: Path, path: str):
    """Open directories and files without following symlinks, applying snapshot budgets."""
    issues, files, repo_files = [], {}, set()
    forbidden = {name.casefold() for name in FORBIDDEN_DIRECTORY_NAMES} | {'examples'}
    for current, dirs, names, dirfd in os.fwalk(root, topdown=True, follow_symlinks=False):
        dirs[:] = [d for d in dirs if d.casefold() not in forbidden]
        for name in names:
            try:
                if stat.S_ISREG(os.stat(name, dir_fd=dirfd, follow_symlinks=False).st_mode):
                    repo_files.add((Path(current) / name).relative_to(root).as_posix())
            except OSError:
                pass
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors = []
    try:
        fd = os.open(root, flags)
        descriptors.append(fd)
        for part in path.split('/'):
            fd = os.open(part, flags, dir_fd=fd)
            descriptors.append(fd)
        total = count = 0
        def visit(dirfd, prefix):
            nonlocal total, count
            with os.scandir(dirfd) as iterator:
                entries = sorted(iterator, key=lambda e: e.name)
            for entry in entries:
                name = entry.name
                full = prefix + '/' + name
                try:
                    info = entry.stat(follow_symlinks=False)
                    mode = info.st_mode
                    if stat.S_ISLNK(mode):
                        issues.append(Issue('structure', 'Symlinks are forbidden in a submission', full))
                    elif stat.S_ISDIR(mode):
                        if name.casefold() in forbidden:
                            issues.append(Issue('structure', 'Forbidden directory in submission', full))
                            continue
                        child = os.open(name, flags, dir_fd=dirfd)
                        try:
                            visit(child, full)
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(mode):
                        count += 1
                        total += info.st_size
                        if any(name.casefold().endswith(s.casefold()) for s in FORBIDDEN_SUFFIXES):
                            issues.append(Issue('structure', 'Forbidden file type in submission', full))
                        if info.st_size > MAX_FILE_BYTES:
                            issues.append(Issue('structure', f'File exceeds {MAX_FILE_BYTES} bytes', full))
                            continue
                        if count > MAX_SUBMISSION_FILES or total > MAX_SUBMISSION_BYTES:
                            continue
                        filefd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dirfd)
                        try:
                            actual = os.fstat(filefd)
                            if not stat.S_ISREG(actual.st_mode) or actual.st_size != info.st_size or (actual.st_dev, actual.st_ino) != (info.st_dev, info.st_ino):
                                raise OSError('file changed')
                            with os.fdopen(filefd, 'rb', closefd=False) as f:
                                data = f.read(MAX_FILE_BYTES + 1)
                            if len(data) != info.st_size:
                                raise OSError('file changed')
                            files[full] = data
                            if _compiled_binary(data):
                                issues.append(Issue('structure', 'Compiled binary artifact is forbidden in a submission', full))
                        finally:
                            os.close(filefd)
                    else:
                        issues.append(Issue('structure', 'Special files are forbidden in a submission', full))
                except OSError as exc:
                    issues.append(Issue('tool', f'Не удалось прочитать файл: {exc}', full))
        visit(fd, path)
        if count > MAX_SUBMISSION_FILES:
            issues.append(Issue('structure', f'Submission exceeds {MAX_SUBMISSION_FILES} files', path))
        if total > MAX_SUBMISSION_BYTES:
            issues.append(Issue('structure', f'Submission exceeds {MAX_SUBMISSION_BYTES} bytes', path))
    except OSError as exc:
        issues.append(Issue('structure', f'Не удалось открыть каталог: {exc}', path))
    finally:
        for fd in reversed(descriptors):
            os.close(fd)
    for required in ('README.md', 'CMakeLists.txt'):
        if path + '/' + required not in files:
            issues.append(Issue('structure', f'Missing required {required}', path))
    if not any(p.startswith(path + '/src/') for p in files):
        issues.append(Issue('structure', 'Missing required src/ directory', path))
    if not any(p.startswith(path + '/src/') and p.endswith('.cu') for p in files):
        issues.append(Issue('structure', 'Missing required .cu source file in src/', path))
    return files, repo_files, issues


def run_local(
    submission_path: str, *, trusted_root: Path = TRUSTED_ROOT,
    format_function: Callable[[dict[str, bytes], str, Path], list[Issue]] | None = None,
) -> dict[str, Any]:
    result = _result()
    result['skipped'] = ['PR metadata', 'PR diff']
    match = re.fullmatch(r'works/([^/]+)/lab([1-5])', submission_path)
    if not match or not login_is_valid(match.group(1)):
        return _finish(result, [Issue('structure', 'Expected works/<login>/lab[1-5]')])
    lab = int(match.group(2))
    result['submission_path'], result['lab'] = submission_path, lab
    try:
        files, repo_files, issues = _local_files(Path(trusted_root).resolve(), submission_path)
        issues.extend(_student_issues(files, repo_files, submission_path, lab,
                                      format_function or format_submission, Path(trusted_root)))
        return _finish(result, issues)
    except Exception as exc:
        message = redact_secrets(f'Локальная проверка ({type(exc).__name__}): {exc}')
        return _finish(result, [Issue('tool', message[:3000], submission_path)])
