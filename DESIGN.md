# ISA Extension Design

## Model & Quantization Scheme (Phase 1, decided 2026-09-10)

- **Model**: MobileNetV3-Large (torchvision pretrained, ImageNet1K_V2 weights), 5.48M params —
  matches spec's recommended candidate table. Real pretrained weights (not synthetic).
- **Representative unit chosen for first bare-metal port**: `features[3]`, an `InvertedResidual`
  block with no squeeze-excite (simplest complete block covering 2 major operator types):
  expand pointwise conv (24->72, 1x1) -> depthwise conv (72ch, 3x3, pad1) -> project pointwise
  conv (72->24, 1x1) -> residual add. Operates on a 56x56 spatial feature map.
  Satisfies the spec's minimum "Evaluation Coverage": 2 major operators (pointwise conv ~=
  GEMM, depthwise conv) + 1 complete CNN block.
- **BatchNorm folding**: each Conv2dNormActivation's BN is folded into the preceding conv's
  weight/bias analytically (`W' = W * gamma/sqrt(var+eps)`, `b' = beta - gamma*mean/sqrt(var+eps)`)
  so inference is pure conv+bias, no separate BN op needed at runtime.
- **Weight quantization**: INT8, per-output-channel symmetric (zero_point = 0). scale[oc] =
  max(abs(W[oc])) / 127.
- **Activation quantization**: INT8, per-tensor asymmetric (nonzero zero-point), calibrated from
  observed min/max over a synthetic calibration batch (16 random ImageNet-shaped Gaussian images
  run through the real pretrained float model). Using synthetic (non-photographic) calibration
  data is a known limitation — it widens activation ranges vs real images and increases
  quantization error (~4.9% max relative error vs the float model for this block) — but this is
  irrelevant to the project's grading, which verifies bit-exact kernel correctness against our
  own quantized reference, not ImageNet classification accuracy.
- **Requantization** (TFLite/gemmlowp-style, matches the project's required pipeline exactly):
  `accumulator (int32) -> multiply by fixed-point multiplier (int64) -> round -> right-shift ->
  add zero_point -> saturate to [-128,127]` (or `[zero_point,127]` when fused with ReLU).
  Multiplier decomposition: `real_multiplier = q_fixed * 2^(shift-31)`, `q_fixed` in `[2^30,2^31)`,
  computed via `math.frexp` (see `multiply_by_quantized_multiplier` in both the Python reference
  and `src/kernels/kernels.c` — implementations are numerically identical, verified bit-exact
  under Spike).
- **Residual add**: both branches (project-conv output and original block input) are rescaled to
  a common domain via their own multiplier/shift, added as int64, then requantized once to the
  output scale/zero-point.
- **Reproducibility**: `scripts/model_prep/export_block3.py` (extraction + quantization + Python
  integer golden reference) -> `scripts/model_prep/export_to_c.py` (writes
  `src/model_data/block3_params.h` and `block3_golden.h`) -> `src/kernels/kernels.c` +
  `src/block3_test/main.c` (bare-metal C reproduction) -> `scripts/build_and_run_block3.sh`
  (build + run under Spike, exit code 0 = bit-exact match). Python env: WSL venv at
  `~/riscv-tools/pyenv` (torch/torchvision CPU-only, see docs/toolchain_setup.md).
- **Known toolchain requirement**: bare-metal code with large static arrays must be compiled with
  `-mcmodel=medany` (RAM base 0x80000000 is right at the edge of the default `medlow` model's
  absolute-addressing range, causing `R_RISCV_HI20 relocation truncated to fit` otherwise).

## Status
ISA extension design itself is still draft — to be filled in during Phase 4, after Spike
bottleneck profiling (Phase 3) is complete. Instruction semantics must be justified by measured
bottlenecks, not chosen in advance.

## `Xqmac8` — packed INT8 multiply-accumulate

- **Purpose**: TBD (fill in after profiling — expected to reduce scalar multiply/loop overhead).
- **Semantics**: `acc += A0*W0 + A1*W1 + ... + A7*W7` over 8 packed INT8 lanes.
- **Encoding**: TBD (custom-0/custom-1 opcode space, operand registers, funct fields).
- **Operand packing**: TBD (two 64-bit regs holding 8x INT8 each, or memory-packed).
- **Accumulator width**: TBD (e.g. single 32-bit acc vs 8x lane-wise accumulators) — document tradeoff.
- **Exceptions/edge cases**: overflow/saturation behavior, alignment requirements.

## Requantization instruction

- **Purpose**: fuse `accumulator -> round -> shift -> zero-point add -> saturate` into one op.
- **Semantics**: TBD.
- **Encoding**: TBD.

## Design alternatives considered

TBD — record rejected alternatives and why, once design work starts.

## Verification plan

- Directed tests: hand-computed input/output pairs covering edge cases (saturation, zero, max/min INT8).
- Randomized tests: random operand vectors compared against a software reference model.
- Kernel-level verification: kernel output using the new instruction matches the golden reference
  produced in Phase 1/2.
