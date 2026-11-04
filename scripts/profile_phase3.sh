#!/usr/bin/env bash
# Phase 3: build the profiling targets, run each under Spike with --log-commits, disassemble
# each ELF, and classify the dynamic instruction trace into an instruction mix (load/store/alu/
# multiply/branch). Writes one CSV row per target to results/phase3_instruction_mix.csv.
set -euo pipefail

TOOLCHAIN_BIN="$HOME/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin"
SPIKE_BIN="$HOME/riscv-tools/spike-install/bin"
GCC="$TOOLCHAIN_BIN/riscv-none-elf-gcc"
OBJDUMP="$TOOLCHAIN_BIN/riscv-none-elf-objdump"
SPIKE="$SPIKE_BIN/spike"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
# Raw commit logs can be 100+ MB of text; writing them through the /mnt/* Windows-drive bridge
# (9p/DrvFS) is dramatically slower than native WSL ext4, so build/run there instead and only
# copy the small parsed CSV back into the repo.
LOG_DIR="$HOME/riscv-tools/artifacts/phase3_logs"
RESULTS_DIR="$ROOT_DIR/results"
CSV="$RESULTS_DIR/phase3_instruction_mix.csv"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
rm -f "$CSV"

CFLAGS=(-march=rv64imc -mabi=lp64 -mcmodel=medany -Wall -O2 -fno-section-anchors -nostdlib -nostartfiles
        -fno-builtin -fno-tree-loop-distribute-patterns -I "$SRC_DIR")

build_and_run() {
    local name="$1"; shift
    local defines="$1"; shift
    local sources=("$@")

    local elf="$LOG_DIR/$name.elf"
    local log="$LOG_DIR/$name.commit.log"
    local dis="$LOG_DIR/$name.dis"

    echo "== $name =="
    "$GCC" "${CFLAGS[@]}" $defines "$SRC_DIR/common/crt0.S" "${sources[@]}" -T "$SRC_DIR/common/link.ld" -o "$elf"
    "$OBJDUMP" -d "$elf" > "$dis"
    "$SPIKE" -l --log-commits --isa=rv64imc --log="$log" "$elf"
    python3 "$ROOT_DIR/scripts/parse_spike_log.py" --dis "$dis" --log "$log" --name "$name" --csv "$CSV"
}

build_and_run "gemm32"       ""                "$SRC_DIR/microbench/gemm_bench.c"
build_and_run "conv1x1"      "-DBENCH_CONV1X1"  "$SRC_DIR/kernels/kernels.c" "$SRC_DIR/profile_test/main.c"
build_and_run "depthwise"    "-DBENCH_DEPTHWISE" "$SRC_DIR/kernels/kernels.c" "$SRC_DIR/profile_test/main.c"
build_and_run "requantize"   "-DBENCH_REQUANT"  "$SRC_DIR/kernels/kernels.c" "$SRC_DIR/profile_test/main.c"
build_and_run "block3_full"  "-DBENCH_FULL"     "$SRC_DIR/kernels/kernels.c" "$SRC_DIR/profile_test/main.c"

echo
echo "Results written to $CSV"
