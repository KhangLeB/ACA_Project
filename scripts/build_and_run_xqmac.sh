#!/usr/bin/env bash
# Phase 4 step 1: verify the Xqmac8/Xqrequant software instruction models.
# Directed edge-case tests + randomized bit-exact check of conv1x1_xqmac vs the scalar
# conv1x1_int reference, all under Spike. Spike exit code 0 == every check passed.
set -euo pipefail

TOOLCHAIN_BIN="$HOME/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin"
SPIKE_BIN="$HOME/riscv-tools/spike-install/bin"
GCC="$TOOLCHAIN_BIN/riscv-none-elf-gcc"
SPIKE="$SPIKE_BIN/spike"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
OUT="$SRC_DIR/xqmac_test/xqmac.elf"

"$GCC" -march=rv64imc -mabi=lp64 -mcmodel=medany -Wall -O2 \
    -fno-section-anchors -fno-builtin -fno-tree-loop-distribute-patterns \
    -nostdlib -nostartfiles \
    -I "$SRC_DIR" \
    -T "$SRC_DIR/common/link.ld" \
    "$SRC_DIR/common/crt0.S" \
    "$SRC_DIR/kernels/kernels.c" \
    "$SRC_DIR/xqmac_test/main.c" \
    -o "$OUT"

echo "Built $OUT"
"$SPIKE" --isa=rv64imc "$OUT"
echo "Spike exit code: $?"
