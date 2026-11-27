#!/usr/bin/env python3
"""Сверка результата Lab 1 с эталоном NumPy.

Читает вход («N alpha», пары x y) и выход программы (N строк
«x+y  alpha*x+y  max(0,x)»), сравнивает с a+b, alpha*a+b, maximum(a, 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SECTIONS = ("add", "saxpy", "relu")


def read_output(path: Path, n: int) -> dict[str, np.ndarray]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != n:
        raise ValueError(f"Выход: {len(lines)} строк, ожидалось {n}")
    rows = np.asarray([[float(v) for v in ln.split()] for ln in lines], dtype=np.float64)
    if rows.shape != (n, 3):
        raise ValueError(f"Выход имеет форму {rows.shape}, ожидалось ({n}, 3)")
    return dict(zip(SECTIONS, (rows[:, 0], rows[:, 1], rows[:, 2])))


def main() -> int:
    parser = argparse.ArgumentParser(description="Сверка с эталоном NumPy")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alpha", type=float, default=None, help="по умолчанию — alpha из первой строки входа"
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    values = args.input.read_text(encoding="utf-8").split()
    n = int(float(values[0]))
    alpha = args.alpha if args.alpha is not None else float(values[1])
    rest = np.asarray([float(v) for v in values[2 : 2 + 2 * n]], dtype=np.float64)
    # Эталон считает по тем же float32-значениям, которые получает GPU:
    # исходные десятичные числа округляются до float32 при чтении программой.
    a = rest[0::2].astype(np.float32).astype(np.float64)
    b = rest[1::2].astype(np.float32).astype(np.float64)

    expected = {
        "add": a + b,
        "saxpy": alpha * a + b,
        "relu": np.maximum(a, 0.0),
    }
    # Допуск по масштабу слагаемых (а не по |результата|): при вычитании
    # близких величин округление float32 даёт погрешность порядка ulp слагаемых.
    scales = {
        "add": np.abs(a) + np.abs(b),
        "saxpy": np.abs(alpha * a) + np.abs(b),
        "relu": np.abs(a),
    }

    got = read_output(args.output, n)

    worst = 0.0
    failed = False
    for name in SECTIONS:
        actual = got[name]
        ref = expected[name]
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
