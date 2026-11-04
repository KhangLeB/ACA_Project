#!/usr/bin/env bash
# Build the block3 bare-metal kernel test and run it under Spike; verifies the C integer
# kernels reproduce scripts/model_prep/export_block3.py's golden output bit-for-bit.
set -euo pipefail

TOOLCHAIN_BIN="$HOME/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin"
SPIKE_BIN="$HOME/riscv-tools/spike-install/bin"
GCC="$TOOLCHAIN_BIN/riscv-none-elf-gcc"
SPIKE="$SPIKE_BIN/spike"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
OUT="$SRC_DIR/block3_test/block3.elf"

"$GCC" -march=rv64imc -mabi=lp64 -mcmodel=medany -Wall -O2 -fno-section-anchors -nostdlib -nostartfiles \
    -I "$SRC_DIR" \
    -T "$SRC_DIR/common/link.ld" \
    "$SRC_DIR/common/crt0.S" \
    "$SRC_DIR/kernels/kernels.c" \
    "$SRC_DIR/block3_test/main.c" \
    -o "$OUT"

echo "Built $OUT"
"$SPIKE" --isa=rv64imc "$OUT"
echo "Spike exit code: $?"
