#!/usr/bin/env python3
"""Генерация входов для Lab 1.

Формат: первая строка — n, далее n строк «a b» (float).
Seed фиксируется для воспроизводимости.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация векторов a, b")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=["random", "zeros", "mixed-sign", "tiny"],
        default="random",
        help="random: знаковые величины; zeros: нули; "
        "mixed-sign: смена знака; tiny: малые значения около границы блока",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    a: list[float] = []
    b: list[float] = []
    for _ in range(args.n):
        if args.kind == "zeros":
            a.append(0.0)
            b.append(0.0)
        elif args.kind == "mixed-sign":
            a.append(rng.uniform(-1000, 1000))
            b.append(-rng.uniform(-1000, 1000) if rng.random() < 0.5 else rng.uniform(-1000, 1000))
        else:
            a.append(rng.uniform(-1000, 1000))
            b.append(rng.uniform(-1000, 1000))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(f"{args.n}\n")
        for x, y in zip(a, b):
            f.write(f"{x:.9g} {y:.9g}\n")
    print(f"n={args.n} seed={args.seed} kind={args.kind} -> {args.out}")


if __name__ == "__main__":
    main()
