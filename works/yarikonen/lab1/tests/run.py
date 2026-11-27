import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

MODES = ("coalesced", "noncoalesced")
DATA = Path(__file__).resolve().parent.parent / "data"


def run(exe, source, output, *options):
    return subprocess.run(
        [str(exe), "--input", str(source), "--output", str(output), *options],
        capture_output=True,
        text=True,
    )


def check(exe, work):
    output = work / "output.txt"
    source = work / "input.txt"
    subprocess.run([str(exe), "--help"], check=True, capture_output=True)

    expected = (DATA / "sample.expected.txt").read_text()
    for args in ((), ("--mode", "coalesced"), ("--mode", "noncoalesced")):
        run(exe, DATA / "sample.txt", output, *args).check_returncode()
        assert output.read_text() == expected, args
    print("Course example: default and both explicit modes match exactly")

    rng = np.random.default_rng(42)
    alpha = np.float32(-0.731)
    max_error = np.zeros(3)
    special = np.array([(0.0, -0.0), (-0.0, 1.0), (-1.0, 0.0), (1.0, -1.0)])
    for n in (0, 1, 255, 256, 257, 1023, 1024, 1025, 2049):
        xy = rng.uniform(-100, 100, size=(n, 2)).astype(np.float32)
        xy[: min(n, len(special))] = special[:n]
        x, y = xy.T
        reference = np.column_stack((x + y, np.multiply(alpha, x) + y, np.maximum(0, x)))
        with source.open("w", encoding="ascii") as file:
            file.write(f"{n} {alpha:.9g}\n")
            np.savetxt(file, xy, fmt="%.9g")
        for mode in MODES:
            block = 255 if n == 2049 else 256
            options = ["--mode", mode, "--block-size", str(block)]
            if n == 1025:
                options.extend(("--repeat", "3"))
            result = run(exe, source, output, *options)
            if result.returncode:
                raise RuntimeError(result.stderr)
            if n == 0:
                assert output.stat().st_size == 0, mode
            else:
                actual = np.loadtxt(output, dtype=np.float32, ndmin=2)
                np.testing.assert_array_equal(actual, reference, err_msg=f"{n=}, {mode=}")
                max_error = np.maximum(max_error, np.max(np.abs(actual - reference), axis=0))
            print(f"N={n:4d} block={block} {mode}: OK")

    for bad_input in ("-1 1\n", "1 1\n2\n", "0 1\n42\n", "1 abc\n0 0\n", "1 2-3 4\n"):
        source.write_text(bad_input)
        assert run(exe, source, output).returncode == 2, bad_input
    assert run(exe, DATA / "sample.txt", output, "--block-size", "0").returncode == 2
    assert run(exe, DATA / "sample.txt", output, "--mode", "unknown").returncode == 2
    assert run(exe, DATA / "sample.txt", output / "missing.txt").returncode == 1
    print("Maximum absolute errors (add, SAXPY, ReLU):", *max_error)


def make_input(path, n):
    rows = [f"{(i % 511 - 255) / 8:g} {(i % 257 - 128) / 4:g}\n" for i in range(4096)]
    chunk = "".join(rows).encode("ascii")
    count, tail = divmod(n, len(rows))
    with path.open("wb") as file:
        file.write(f"{n} 0.731\n".encode("ascii"))
        for _ in range(count):
            file.write(chunk)
        file.write("".join(rows[:tail]).encode("ascii"))


def benchmark(exe, work):
    source = work / "benchmark.txt"
    output = work / "output.txt"
    for n, blocks in ((257, (256,)), (8_388_609, (128, 256, 512))):
        make_input(source, n)
        for block in blocks:
            for mode in MODES:
                result = run(
                    exe,
                    source,
                    output,
                    "--mode",
                    mode,
                    "--block-size",
                    str(block),
                    "--repeat",
                    "21",
                )
                if result.returncode:
                    raise RuntimeError(result.stderr)
                print(f"N={n} block={block} {mode} {result.stdout.strip()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "benchmark"))
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        action = check if args.action == "check" else benchmark
        action(args.executable.resolve(), Path(tmp))


if __name__ == "__main__":
    main()
