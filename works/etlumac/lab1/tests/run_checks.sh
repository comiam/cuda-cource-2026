#!/usr/bin/env bash
# Проверки программы через командную строку. Запуск из папки lab1:
#   bash tests/run_checks.sh
set -u
cd "$(dirname "$0")/.."
BIN=./build/submission
fail=0
tmp=$(mktemp -d)

ok() { echo "OK   $1"; }
bad() { echo "FAIL $1"; fail=1; }

# expect_code <ожидаемый код> <описание> <команда...>
expect_code() {
    local want=$1 name=$2
    shift 2
    "$@" > /dev/null 2>&1
    local got=$?
    if [ "$got" -eq "$want" ]; then ok "$name (код $got)"; else bad "$name (ожидали $want, получили $got)"; fi
}

# 1. Пример из задания, оба режима
for mode in coalesced noncoalesced; do
    $BIN --input data/example.txt --output "$tmp/ex_$mode.txt" --mode "$mode" > /dev/null
    if diff -q data/example_expected.txt "$tmp/ex_$mode.txt" > /dev/null; then
        ok "пример из задания, режим $mode"
    else
        bad "пример из задания, режим $mode"
    fi
done

# 2. Случайные файлы разного размера: оба режима должны дать одинаковый файл
for n in 1 255 256 257 1024 1025 100000; do
    python3 -c "
import random
random.seed($n)
print($n, 0.75)
for _ in range($n):
    print(round(random.uniform(-10, 10), 4), round(random.uniform(-10, 10), 4))
" > "$tmp/in_$n.txt"
    $BIN --input "$tmp/in_$n.txt" --output "$tmp/c_$n.txt" --mode coalesced > /dev/null
    $BIN --input "$tmp/in_$n.txt" --output "$tmp/n_$n.txt" --mode noncoalesced > /dev/null
    lines=$(wc -l < "$tmp/c_$n.txt")
    if cmp -s "$tmp/c_$n.txt" "$tmp/n_$n.txt" && [ "$lines" -eq "$n" ]; then
        ok "N=$n: режимы совпадают, строк $lines"
    else
        bad "N=$n: режимы различаются или неверное число строк"
    fi
done

# 3. N = 0: пустой выходной файл
printf '0 1.0\n' > "$tmp/zero.txt"
expect_code 0 "N=0" $BIN --input "$tmp/zero.txt" --output "$tmp/zero_out.txt"
if [ ! -s "$tmp/zero_out.txt" ]; then ok "N=0: выходной файл пустой"; else bad "N=0: файл не пустой"; fi

# 4. Коды возврата
printf '5 1.0\n1 2\n' > "$tmp/short.txt"
expect_code 0 "--help" $BIN --help
expect_code 2 "без аргументов" $BIN
expect_code 2 "нет файла входа" $BIN --input "$tmp/nofile.txt" --output "$tmp/o.txt"
expect_code 2 "неверный режим" $BIN --input data/example.txt --output "$tmp/o.txt" --mode wrong
expect_code 2 "неизвестный аргумент" $BIN --foo
expect_code 2 "N больше числа пар" $BIN --input "$tmp/short.txt" --output "$tmp/o.txt"
expect_code 1 "нельзя записать результат" $BIN --input data/example.txt --output /no_such_dir/o.txt

rm -rf "$tmp"
if [ "$fail" -eq 0 ]; then echo "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ"; else echo "ЕСТЬ ОШИБКИ"; fi
exit $fail
