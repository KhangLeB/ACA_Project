#!/usr/bin/env bash
# Phase 4 step 2: measure dynamic instruction-count reduction of the Xqmac8 conv1x1 (X1) vs the
# scalar conv1x1 (B0), the spec's ">=2x reduction for a major GEMM/FC kernel" target. Both run the
# REAL kernel at 8x8 spatial / real channel counts (24->72); each retired instruction is one commit
# line, so total line count is the dynamic instruction count. Logs go to a native WSL path (writing
# 100k+ line logs to /mnt/d is far slower).
set -euo pipefail

TOOLCHAIN_BIN="$HOME/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin"
SPIKE="$HOME/riscv-tools/spike-install/bin/spike"
GCC="$TOOLCHAIN_BIN/riscv-none-elf-gcc"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
LOGDIR="$HOME/riscv-tools/artifacts/x1_logs"
mkdir -p "$LOGDIR"

CFLAGS=(-march=rv64imc -mabi=lp64 -mcmodel=medany -Wall -O2
        -fno-section-anchors -fno-builtin -fno-tree-loop-distribute-patterns
        -nostdlib -nostartfiles -I "$SRC_DIR" -T "$SRC_DIR/common/link.ld")
SRCS=("$SRC_DIR/common/crt0.S" "$SRC_DIR/kernels/kernels.c" "$SRC_DIR/profile_test/main.c")

build_run() { # name, extra-define, use-insn(0/1), spike-ext-args
  local name="$1" def="$2" useinsn="$3"; shift 3
  local out="$SRC_DIR/profile_test/${name}.elf"
  local flags=("${CFLAGS[@]}" "-D${def}")
  [ "$useinsn" = "1" ] && flags+=(-DXQMAC_USE_INSN)
  "$GCC" "${flags[@]}" "${SRCS[@]}" -o "$out"
  "$SPIKE" "$@" -l --log-commits --log="$LOGDIR/${name}.log" --isa=rv64imc "$out" >/dev/null 2>&1 || true
  # One retire line per instruction carries the privilege digit after the colon; the bare
  # disassembly echo line does not, so this regex counts each retired instruction exactly once.
  grep -cE '^core +[0-9]+: +[0-9]+ +0x' "$LOGDIR/${name}.log"
}

b0=$(build_run conv1x1_b0 BENCH_CONV1X1     0)
x1=$(build_run conv1x1_x1 BENCH_CONV1X1_XQ  1 --extension=xqnn)

echo "conv1x1 (24->72, 8x8) dynamic instruction count"
echo "  B0 scalar : $b0"
echo "  X1 Xqmac8 : $x1"
awk -v b="$b0" -v x="$x1" 'BEGIN{ if (x>0) printf "  reduction : %.2fx\n", b/x }'
