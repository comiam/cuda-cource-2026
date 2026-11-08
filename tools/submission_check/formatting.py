"""Run trusted formatter configuration against isolated, non-executable copies."""
from __future__ import annotations

import os
import resource
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from .models import Issue
from .policy import MAX_FILE_BYTES, lint_group

_TIMEOUT = 30
_OUTPUT_LIMIT = MAX_FILE_BYTES + 65536


def _limit_output() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (_OUTPUT_LIMIT, _OUTPUT_LIMIT))


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
            if not config.is_file():
                issues.append(Issue('tool', 'Trusted formatter configuration is missing', name))
                continue
            args = {
                'cpp': ['clang-format', '--style=file:' + str(config), str(target)],
                'python': ['ruff', 'format', '--check', '--config', str(config), str(target)],
                'cmake': ['cmake-format', '-c', str(config), '--', str(target)],
                'markdown': ['pymarkdown', '--strict-config', '--return-code-scheme', 'explicit', '--config', str(config), 'scan', str(target)],
            }[group]
            try:
                with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                    result = subprocess.run(args, cwd=working, env=env, stdin=subprocess.DEVNULL,
                                            stdout=stdout, stderr=stderr, timeout=_TIMEOUT,
                                            check=False, preexec_fn=_limit_output)
                    stdout.seek(0)
                    output = stdout.read(_OUTPUT_LIMIT + 1)
                    stderr.seek(0)
                    errors = stderr.read(_OUTPUT_LIMIT + 1)
                if len(output) >= _OUTPUT_LIMIT or len(errors) >= _OUTPUT_LIMIT:
                    raise OSError('formatter output exceeded limit')
                if group in {'cpp', 'cmake'}:
                    if result.returncode != 0:
                        issues.append(Issue('tool', f'{group} formatter failed', name))
                    elif output != content:
                        issues.append(Issue('formatting', f'{group} formatting differs', name))
                elif (group == 'python' and result.returncode == 1) or (group == 'markdown' and result.returncode == 4):
                    issues.append(Issue('formatting', f'{group} formatting differs', name))
                elif result.returncode != 0:
                    issues.append(Issue('tool', f'{group} formatter failed', name))
            except (OSError, subprocess.SubprocessError, ValueError):
                issues.append(Issue('tool', f'{group} formatter unavailable or failed', name))
    return issues
