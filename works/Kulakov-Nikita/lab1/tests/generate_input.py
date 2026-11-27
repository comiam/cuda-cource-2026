import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.n < 0:
        parser.error("--n должен быть неотрицательным")

    random.seed(args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(f"{args.n} {args.alpha:.9g}\n")
        for i in range(args.n):
            x = random.uniform(-10.0, 10.0)
            y = random.uniform(-10.0, 10.0)

            # В начале файла будут нули и значения разных знаков.
            if i == 0:
                x, y = 0.0, 0.0
            elif i == 1:
                x, y = -1.0, 2.0
            elif i == 2:
                x, y = 3.0, -4.0

            file.write(f"{x:.9g} {y:.9g}\n")

    print(f"Создан файл {path}, N={args.n}, alpha={args.alpha}, seed={args.seed}")


if __name__ == "__main__":
    main()
