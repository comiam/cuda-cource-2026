#!/usr/bin/env python3
"""Сверка результата Lab 1 с эталоном NumPy.

Читает вход (a, b) и выход программы (секции add, saxpy, relu),
сравнивает с a+b, alpha*a+b, maximum(a, 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SECTIONS = ("add", "saxpy", "relu")


def read_output(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and line.split("=")[0] in ("mode", "n", "alpha"):
            continue
        key = line
        if key in SECTIONS:
            current = key
            result[current] = np.empty(0, dtype=np.float64)
            continue
        if current is None:
            raise ValueError(f"Неизвестная секция перед данными: {key!r}")
        result[current] = np.fromstring(line, dtype=np.float64, sep=" ")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Сверка с эталоном NumPy")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    values = args.input.read_text(encoding="utf-8").split()
    n = int(float(values[0]))
    rest = np.asarray([float(v) for v in values[1 : 1 + 2 * n]], dtype=np.float64)
    # Эталон считает по тем же float32-значениям, которые получает GPU:
    # исходные десятичные числа округляются до float32 при чтении программой.
    a = rest[0::2].astype(np.float32).astype(np.float64)
    b = rest[1::2].astype(np.float32).astype(np.float64)

    expected = {
        "add": a + b,
        "saxpy": args.alpha * a + b,
        "relu": np.maximum(a, 0.0),
    }
    # Допуск по масштабу слагаемых (а не по |результата|): при вычитании
    # близких величин округление float32 даёт погрешность порядка ulp слагаемых.
    scales = {
        "add": np.abs(a) + np.abs(b),
        "saxpy": np.abs(args.alpha * a) + np.abs(b),
        "relu": np.abs(a),
    }

    got = read_output(args.output)
    if set(got) != set(SECTIONS):
        print(f"Выход содержит секции {sorted(got)}, ожидалось {list(SECTIONS)}")
        return 1

    worst = 0.0
    failed = False
    for name in SECTIONS:
        actual = got[name]
        ref = expected[name]
        if actual.size != ref.size:
            print(f"{name}: длина {actual.size}, ожидалось {ref.size}")
            failed = True
            continue
        diff = np.abs(actual - ref)
        limit = args.atol + args.rtol * scales[name]
        bad = int(np.count_nonzero(diff > limit))
        worst = max(worst, float(diff.max(initial=0.0)))
        status = "OK" if bad == 0 else f"РАСХОЖДЕНИЕ ({bad} элементов)"
        print(f"{name}: max|diff|={float(diff.max(initial=0.0)):.3e} — {status}")
        if bad:
            failed = True

    print(f"Итог: max|diff|={worst:.3e}, допуск atol={args.atol} rtol={args.rtol}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
