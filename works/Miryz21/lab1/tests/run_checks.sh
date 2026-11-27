#!/usr/bin/env bash
# Проверки и измерения Lab 1: размеры вокруг границы блока, оба режима.
# Запуск из каталога лабораторной: bash tests/run_checks.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data results

SIZES=(1 2 15 255 256 257 1000 1048576 10485760)
for n in "${SIZES[@]}"; do
    python3 tests/gen_data.py --n "$n" --seed 42 --kind mixed-sign \
        --out "data/rand_${n}.txt"
done
python3 tests/gen_data.py --n 256 --seed 7 --kind zeros --out data/zeros_256.txt

for f in data/rand_255.txt data/rand_256.txt data/rand_257.txt data/zeros_256.txt; do
    base="$(basename "$f" .txt)"
    for mode in coalesced noncoalesced; do
        ./build/submission --input "$f" --output "results/${base}.${mode}.out" \
            --mode "$mode"
        python3 tests/verify.py --input "$f" \
            --output "results/${base}.${mode}.out" --alpha 2.0
    done
done

# Измерения на больших массивах: оба режима, несколько размеров блока.
for n in 1048576 10485760; do
    for block in 128 256 512; do
        for mode in coalesced noncoalesced; do
            ./build/submission --input "data/rand_${n}.txt" \
                --output "results/perf_${n}.${block}.${mode}.out" \
                --mode "$mode" --block "$block" --repeat 20
        done
    done
done
