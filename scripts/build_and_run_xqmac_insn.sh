#!/usr/bin/env bash
# Phase 4 step 2 verification: build the SAME xqmac test but with -DXQMAC_USE_INSN so conv1x1_xqmac
# and the directed tests emit the REAL custom instructions (.insn r 0x0b,...), then run under Spike
# with the xqnn extension loaded. Spike exit 0 == the hardware instruction semantics match the
# scalar reference bit-for-bit. (Requires scripts/build_spike_ext.sh to have been run first.)
set -euo pipefail

TOOLCHAIN_BIN="$HOME/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin"
SPIKE_BIN="$HOME/riscv-tools/spike-install/bin"
GCC="$TOOLCHAIN_BIN/riscv-none-elf-gcc"
SPIKE="$SPIKE_BIN/spike"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
OUT="$SRC_DIR/xqmac_test/xqmac_insn.elf"

"$GCC" -march=rv64imc -mabi=lp64 -mcmodel=medany -Wall -O2 \
    -DXQMAC_USE_INSN \
    -fno-section-anchors -fno-builtin -fno-tree-loop-distribute-patterns \
    -nostdlib -nostartfiles \
    -I "$SRC_DIR" \
    -T "$SRC_DIR/common/link.ld" \
    "$SRC_DIR/common/crt0.S" \
    "$SRC_DIR/kernels/kernels.c" \
    "$SRC_DIR/xqmac_test/main.c" \
    -o "$OUT"

echo "Built $OUT"
"$SPIKE" --isa=rv64imc --extension=xqnn "$OUT"
echo "Spike exit code: $?"
