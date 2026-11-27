import argparse
import struct
import subprocess
import tempfile
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def make_input(path, n):
    alpha = f32(0.5)
    x_values = []
    y_values = []

    with path.open("w", encoding="utf-8") as file:
        file.write(f"{n} {alpha:.9g}\n")

        for i in range(n):
            x = f32(((i % 17) - 8) * 0.25)
            y = f32(((i % 11) - 5) * 0.5)

            if i % 17 == 0:
                x, y = 0.0, 0.0
            elif i % 17 == 1:
                x, y = -1.0, 2.0

            x_values.append(x)
            y_values.append(y)
            file.write(f"{x:.9g} {y:.9g}\n")

    return alpha, x_values, y_values


def check_result(path, alpha, x_values, y_values):
    n = len(x_values)
    lines = path.read_text(encoding="utf-8").splitlines()

    if len(lines) != n:
        raise RuntimeError(f"Ожидалось строк: {n}, получено: {len(lines)}")

    for i, line in enumerate(lines):
        values = line.split()
        if len(values) != 3:
            raise RuntimeError(f"Строка {i + 1}: ожидалось три числа")

        got_add, got_saxpy, got_relu = map(float, values)

        expected_add = f32(x_values[i] + y_values[i])
        expected_saxpy = f32(f32(alpha * x_values[i]) + y_values[i])
        expected_relu = x_values[i] if x_values[i] > 0.0 else 0.0

        if got_add != expected_add:
            raise RuntimeError(f"Add, элемент {i}: ожидалось {expected_add}, получено {got_add}")

        if got_relu != expected_relu:
            raise RuntimeError(f"ReLU, элемент {i}: ожидалось {expected_relu}, получено {got_relu}")

        tolerance = 1e-5 + 1e-5 * abs(expected_saxpy)
        if abs(got_saxpy - expected_saxpy) > tolerance:
            message = f"SAXPY, элемент {i}: ожидалось {expected_saxpy}, получено {got_saxpy}"
            raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("program", nargs="?", default="build/submission")
    args = parser.parse_args()

    program = Path(args.program).resolve()
    if not program.exists():
        raise SystemExit(f"Не найдена программа: {program}")

    sizes = [0, 1, 255, 256, 257, 513]
    modes = ["coalesced", "noncoalesced"]
    blocks = [128, 256, 512]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)

        for n in sizes:
            input_path = temp / f"input_{n}.txt"
            alpha, x_values, y_values = make_input(input_path, n)

            for mode in modes:
                for block in blocks:
                    output_path = temp / f"output_{n}_{mode}_{block}.txt"

                    subprocess.run(
                        [
                            str(program),
                            "--input",
                            str(input_path),
                            "--output",
                            str(output_path),
                            "--mode",
                            mode,
                            "--block",
                            str(block),
                            "--repeat",
                            "1",
                            "--warmup",
                            "0",
                        ],
                        check=True,
                    )

                    check_result(output_path, alpha, x_values, y_values)
                    print(f"OK: N={n}, mode={mode}, block={block}")

    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
