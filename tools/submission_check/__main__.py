"""Command line entry point for trusted CI and optional local checking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .diagnostics import emit_issues, redact_secrets
from .models import Issue
from .runner import _finish, _result, run_ci, run_local


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = _Parser(prog='python -m tools.submission_check')
    commands = parser.add_subparsers(dest='command', required=True, parser_class=_Parser)
    ci = commands.add_parser('ci')
    ci.add_argument('--pr-number', type=int, required=True)
    ci.add_argument('--expected-head-sha', required=True)
    ci.add_argument('--expected-fingerprint', required=True)
    ci.add_argument('--output', required=True)
    local = commands.add_parser('local')
    local.add_argument('submission_path')
    local.add_argument('--output')
    output = None
    try:
        args = parser.parse_args(argv)
        output = args.output
        if args.command == 'ci':
            result = run_ci(args.pr_number, args.expected_head_sha, args.expected_fingerprint)
        else:
            result = run_local(args.submission_path, trusted_root=Path.cwd())
    except Exception as exc:
        # Preserve the JSON contract for malformed invocations when --output is available.
        for index, value in enumerate(argv):
            if value == '--output' and index + 1 < len(argv):
                output = argv[index + 1]
            elif value.startswith('--output='):
                output = value.partition('=')[2]
        result = _finish(_result(), [Issue('tool', redact_secrets(f'Запуск проверки ({type(exc).__name__}): {exc}'))])
    code = {'success': 0, 'failure': 1, 'error': 2, 'stale': 3}[result['state']]
    if output:
        try:
            with open(output, 'w', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=True)
                stream.write('\n')
        except OSError as exc:
            emit_issues([Issue('tool', f'Не удалось записать результат: {exc}', output)], stage='Сохранение результата', stream=sys.stderr)
            return 2
    print('Submission check:', result['state'])
    emit_issues(
        [Issue(item['category'], item['message'], item.get('path', ''), item.get('line'))
         for item in result.get('issues', [])],
        stage='Проверка сдачи',
    )
    if result.get('infrastructure'):
        print('Infrastructure policy: PR scope and author permissions checked')
    if result.get('skipped'):
        print('SKIP: PR metadata and PR diff (local mode)')
    return code


if __name__ == '__main__':
    sys.exit(main())
