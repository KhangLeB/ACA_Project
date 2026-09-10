#!/usr/bin/env bash
# Build the bare-metal smoke test and run it under Spike; used to verify the toolchain end-to-end.
set -euo pipefail

TOOLCHAIN_BIN="$HOME/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin"
SPIKE_BIN="$HOME/riscv-tools/spike-install/bin"
GCC="$TOOLCHAIN_BIN/riscv-none-elf-gcc"
SPIKE="$SPIKE_BIN/spike"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/smoke_test" && pwd)"
OUT="$SRC_DIR/smoke.elf"

"$GCC" -march=rv64imc -mabi=lp64 -Wall -O2 -nostdlib -nostartfiles \
    -T "$SRC_DIR/link.ld" \
    "$SRC_DIR/crt0.S" "$SRC_DIR/main.c" \
    -o "$OUT"

echo "Built $OUT"
"$SPIKE" --isa=rv64imc "$OUT"
echo "Spike exit code: $?"
