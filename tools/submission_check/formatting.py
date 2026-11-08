"""Run trusted formatter configuration against isolated, non-executable copies."""
from __future__ import annotations

import os
import re
import resource
import subprocess
import tempfile
from itertools import zip_longest
from pathlib import Path, PurePosixPath

from .models import Issue
from .policy import MAX_FILE_BYTES, lint_group

_TIMEOUT = 30
_OUTPUT_LIMIT = MAX_FILE_BYTES + 65536


def _limit_output() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (_OUTPUT_LIMIT, _OUTPUT_LIMIT))


def _bounded(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    marker = '... [truncated]'
    return text[:limit - len(marker)] + marker


def _clean(text: str, target: Path, name: str, base: Path, root: Path) -> str:
    return (text.replace(str(target), name)
            .replace(str(base / 'inputs') + '/', '')
            .replace(str(base), '<temporary>')
            .replace(str(root) + '/', ''))


def _decoded(data: bytes) -> str:
    return data.decode('utf-8', errors='replace')


def _diagnostics(output: bytes, errors: bytes, target: Path, name: str,
                 base: Path, root: Path) -> tuple[str, int | None]:
    stdout = _clean(_decoded(output), target, name, base, root)
    stderr = _clean(_decoded(errors), target, name, base, root)
    match = re.search(re.escape(name) + r':(\d+)(?::\d+)?(?=[:\s]|$)', stdout + '\n' + stderr)
    line = int(match.group(1)) if match and int(match.group(1)) > 0 else None
    parts = []
    for label, text in (('stderr', stderr), ('stdout', stdout)):
        if text.strip():
            parts.append(f'{label}: {_bounded(text.strip())}')
    return ('\n' + '\n'.join(parts) if parts else ''), line


def _first_difference(content: bytes, output: bytes) -> tuple[int, str]:
    for number, (actual, expected) in enumerate(
            zip_longest(content.splitlines(keepends=True), output.splitlines(keepends=True)), 1):
        if actual != expected:
            before = '<end of file>' if actual is None else repr(_decoded(actual))
            after = '<end of file>' if expected is None else repr(_decoded(expected))
            return number, f'- {_bounded(before)}\n+ {_bounded(after)}'
    # splitlines() handles all byte differences, but retain a safe fallback.
    return 1, '(byte contents differ)'


def format_submission(files: dict[str, bytes], submission_path: str, trusted_root: Path) -> list[Issue]:
    """Format-check only policy-selected files, never the submission checkout."""
    issues: list[Issue] = []
    root = Path(trusted_root).resolve()
    configs = {'cpp': '.clang-format', 'python': '.ruff.toml',
               'cmake': '.cmake-format.yaml', 'markdown': '.pymarkdown.json'}
    with tempfile.TemporaryDirectory(prefix='submission-format-') as tmp:
        base = Path(tmp)
        inputs, working, home = base / 'inputs', base / 'work', base / 'home'
        for directory in (inputs, working, home):
            directory.mkdir(mode=0o700)
        env = {'PATH': os.defpath if not os.environ.get('PATH') else os.environ['PATH'],
               'HOME': str(home), 'XDG_CONFIG_HOME': str(home / 'config'),
               'TMPDIR': str(base), 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
               'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        for name, content in sorted(files.items()):
            group = lint_group(name, submission_path)
            if group is None:
                continue
            parts = name.split('/')
            if any(p in ('', '.', '..') for p in parts) or any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
                issues.append(Issue('tool', 'Invalid formatter input path', name))
                continue
            target = inputs.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(content)
            target.chmod(0o400)
            config = root / configs[group]
            tool = {'cpp': 'clang-format', 'python': 'ruff',
                    'cmake': 'cmake-format', 'markdown': 'pymarkdown'}[group]
            if not config.is_file():
                issues.append(Issue('tool', f'{tool}: trusted configuration {configs[group]} is missing for {name}', name))
                continue
            args = {
                'cpp': ['clang-format', '--style=file:' + str(config), str(target)],
                'python': ['ruff', 'format', '--check', '--config', str(config), str(target)],
                'cmake': ['cmake-format', '-c', str(config), '--', str(target)],
                'markdown': ['pymarkdown', '--strict-config', '--return-code-scheme', 'explicit', '--config', str(config), 'scan', str(target)],
            }[group]
            output = errors = b''
            try:
                with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                    try:
                        result = subprocess.run(args, cwd=working, env=env, stdin=subprocess.DEVNULL,
                                                stdout=stdout, stderr=stderr, timeout=_TIMEOUT,
                                                check=False, preexec_fn=_limit_output)
                    finally:
                        stdout.seek(0)
                        output = stdout.read(_OUTPUT_LIMIT + 1)
                        stderr.seek(0)
                        errors = stderr.read(_OUTPUT_LIMIT + 1)
                if len(output) >= _OUTPUT_LIMIT or len(errors) >= _OUTPUT_LIMIT:
                    raise OSError('formatter output exceeded limit')
                if result.returncode == 0 and group in {'cpp', 'cmake'}:
                    if output != content:
                        line, diff = _first_difference(content, output)
                        message = f'{tool}: formatting differs in {name} at line {line}\n{diff}'
                        issues.append(Issue('formatting', _bounded(message, 3500), name, line))
                elif result.returncode != 0:
                    diagnostics, line = _diagnostics(output, errors, target, name, base, root)
                    style = ((group == 'python' and result.returncode == 1)
                             or (group == 'markdown' and result.returncode == 4))
                    if style:
                        message = f'{tool}: formatting differs in {name}{diagnostics}'
                        issues.append(Issue('formatting', _bounded(message, 3500), name, line))
                    else:
                        message = f'{tool}: failed for {name} (exit code {result.returncode}){diagnostics}'
                        issues.append(Issue('tool', _bounded(message, 3500), name, line))
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                if isinstance(exc, subprocess.TimeoutExpired):
                    if not output and isinstance(exc.output, bytes):
                        output = exc.output
                    if not errors and isinstance(exc.stderr, bytes):
                        errors = exc.stderr
                    reason = f'timeout after {exc.timeout} seconds'
                else:
                    reason = f'{type(exc).__name__}: {exc}'
                reason = _clean(reason, target, name, base, root)
                diagnostics, line = _diagnostics(output, errors, target, name, base, root)
                message = f'{tool}: failed for {name} ({reason}){diagnostics}'
                issues.append(Issue('tool', _bounded(message, 3500), name, line))
    return issues
