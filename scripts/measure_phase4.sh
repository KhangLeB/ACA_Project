#!/usr/bin/env bash
# Phase 4 evaluation: full instruction-mix (load/store/alu/multiply/branch/custom) and total
# dynamic instruction count for every B0/B1/X1 kernel variant, via Spike --log-commits joined
# against objdump disassembly (scripts/parse_spike_log.py; custom opcode 0x0b -> "custom").
# Writes results/phase4_instruction_mix.csv. 8x8 spatial, real channel counts; logs on native WSL.
set -euo pipefail

TOOLCHAIN_BIN="$HOME/riscv-tools/xpack-riscv-none-elf-gcc-14.2.0-3/bin"
SPIKE="$HOME/riscv-tools/spike-install/bin/spike"
GCC="$TOOLCHAIN_BIN/riscv-none-elf-gcc"
OBJDUMP="$TOOLCHAIN_BIN/riscv-none-elf-objdump"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
LOGDIR="$HOME/riscv-tools/artifacts/phase4_logs"
CSV="$ROOT_DIR/results/phase4_instruction_mix.csv"
mkdir -p "$LOGDIR"
rm -f "$CSV"

CFLAGS=(-march=rv64imc -mabi=lp64 -mcmodel=medany -Wall -O2
        -fno-section-anchors -fno-builtin -fno-tree-loop-distribute-patterns
        -nostdlib -nostartfiles -I "$SRC_DIR" -T "$SRC_DIR/common/link.ld")
SRCS=("$SRC_DIR/common/crt0.S" "$SRC_DIR/kernels/kernels.c" "$SRC_DIR/profile_test/main.c")

run() { # name, define, use-insn(0/1), [spike extra args...]
  local name="$1" def="$2" useinsn="$3"; shift 3
  local out="$SRC_DIR/profile_test/${name}.elf"
  local flags=("${CFLAGS[@]}" "-D${def}")
  [ "$useinsn" = "1" ] && flags+=(-DXQMAC_USE_INSN)
  "$GCC" "${flags[@]}" "${SRCS[@]}" -o "$out" 2>/dev/null
  "$OBJDUMP" -d "$out" > "$LOGDIR/${name}.dis"
  "$SPIKE" "$@" -l --log-commits --log="$LOGDIR/${name}.log" --isa=rv64imc "$out" >/dev/null 2>&1 || true
  python3 "$ROOT_DIR/scripts/parse_spike_log.py" \
      --dis "$LOGDIR/${name}.dis" --log "$LOGDIR/${name}.log" --name "$name" --csv "$CSV"
}

echo "===== conv1x1 (GEMM/FC): B0 scalar vs X1 Xqmac8 ====="
run conv1x1_b0 BENCH_CONV1X1     0
run conv1x1_x1 BENCH_CONV1X1_XQ  1 --extension=xqnn
echo
echo "===== requantize: B0 scalar vs X1 Xqrequant ====="
run requant_b0 BENCH_REQUANT     0
run requant_x1 BENCH_REQUANT_XQ  1 --extension=xqnn
echo
echo "===== depthwise: B0 baseline vs B1 interior/border split ====="
run depthwise_b0 BENCH_DEPTHWISE    0
run depthwise_b1 BENCH_DEPTHWISE_B1 0

echo
echo "Wrote $CSV"
column -t -s, "$CSV"
