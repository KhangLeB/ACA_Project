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

**Phase 4 step 1 done (2026-10-07):** instruction semantics locked and validated as C software
models (`src/kernels/xqmac.h`); `conv1x1_xqmac` rewritten to use them and proven bit-exact
against the scalar `conv1x1_int` reference plus directed edge-case tests
(`scripts/build_and_run_xqmac.sh`, Spike exit 0).

**Phase 4 step 2 done (2026-10-07):** both instructions implemented as a real Spike extension
(`src/spike_ext/xqnn.cc`, custom-0 opcode 0x0b) and emitted from the kernel via `.insn` inline asm
(`-DXQMAC_USE_INSN`). Functionally verified bit-exact (`scripts/build_and_run_xqmac_insn.sh`) and
measured **4.13× dynamic-instruction reduction** for conv1x1 (B0 970k → X1 235k; 13,824 retired
xqmac8 = 8 MACs each), **1.55×** for requantize (Xqrequant), and **1.21×** for depthwise from the
B1 interior/border software split (branches 28%→21%) — meets the required ≥2× GEMM/FC target
(`scripts/measure_phase4.sh`, `results/phase4_instruction_mix.csv`, EXPERIMENTS.md 2026-10-07).
X1's new dominant category is ALU (51%: address generation, loop control, `zp·colsum` correction),
motivating hardware-loop / post-increment / fused extensions next. Remaining before midterm: push
Phase 3+4 to GitHub and write the report; then CV-Wally RTL.

## `Xqmac8` — packed INT8 multiply-accumulate

- **Purpose**: confirmed by Phase 3 Spike profiling (EXPERIMENTS.md, 2026-09-12): in the naive
  scalar kernels only ~11-13% of dynamic instructions are the actual `mul`, while ~7-9 total
  instructions retire per useful MAC (sign-extension, zero-point subtraction, address
  computation, load/store). Folding 8 loads + 8 sign-extends + 8 muls + 8 adds into one packed
  instruction directly targets this dominant overhead category.
- **Semantics**: `acc += A0*W0 + A1*W1 + ... + A7*W7` over 8 packed INT8 lanes. Reference
  implementation: `xqmac8()` in `src/kernels/xqmac.h` (validated bit-exact, 2026-10-07).
- **Operand/accumulator model** (locked 2026-10-07):
  - Two 64-bit GPR source operands `rs1`, `rs2`, each holding 8 **signed INT8** lanes
    (lane i = bits [8i+7:8i], little-endian).
  - Accumulator is a 32-bit GPR, read-modify-write: `rd = rd + dot8(rs1, rs2)`. 32 bits is
    sufficient — worst case is 8*128*128 = 131072 per instruction, and a full K=512 reduction
    stays below 2^24, far inside int32. This avoids a wider/second accumulator register.
  - **Zero-point handling (resolves the "packed INT8 is broken by (a-zp)" risk):** Xqmac8 does
    NOT subtract the activation zero-point. The kernel folds it out of the inner loop once per
    output channel using `sum((a-z)*w) = sum(a*w) - z*sum(w)` (weights are symmetric, zp_w=0,
    so only the activation term needs correction). The packed operands therefore stay true INT8
    and the 8-lane packing is never widened to 9 bits.
  - **Data layout (resolves the CHW packed-load risk):** `conv1x1_xqmac` gathers each output
    pixel's `c_in` activations into a contiguous scratch buffer once, then reuses it across all
    `c_out` channels. Weights `w[oc*c_in+ic]` are already contiguous in the reduction dimension,
    so after the gather BOTH operands feed packed 64-bit loads. The gather also directly attacks
    Phase-3 bottleneck #3 (no activation reuse across output channels).
  - Tail handling: `c_in` not a multiple of 8 falls back to scalar MACs for the remainder (all
    block3 configs have c_in in {24,72}, both divisible by 8, but the kernel stays general).
- **Encoding**: custom-0 opcode space, R-type (`rd`, `rs1`, `rs2`, funct3/funct7). Exact bit
  fields to be fixed when the Spike decoder stub is added (Phase 4 step 2).

## Requantization instruction

- **Purpose**: fuse `accumulator -> fixed-point multiply -> round -> shift -> zero-point add ->
  saturate` into one op (Phase-3 showed ~18 instr per output element for this "single" operation).
- **Semantics**: `xqrequant()` in `src/kernels/xqmac.h` — `real_multiplier = q_fixed * 2^(shift-31)`,
  round-half-up on the right shift, add zero-point, saturate to `[lo, hi]` (lo = zero_point when
  fused with ReLU, else -128; hi = 127). Numerically identical to
  `multiply_by_quantized_multiplier()` + clamp in `src/kernels/kernels.c` (so it stays bit-exact
  against the Phase-1 golden reference).
- **Encoding**: TBD (Phase 4 step 2). Likely a two-instruction pair or an immediate-fielded op
  because it needs q_fixed, shift, zp, and saturation bounds — operand count exceeds one R-type.

## Design alternatives considered

TBD — record rejected alternatives and why, once design work starts.

## Verification plan

- Directed tests: hand-computed input/output pairs covering edge cases (saturation, zero, max/min INT8).
- Randomized tests: random operand vectors compared against a software reference model.
- Kernel-level verification: kernel output using the new instruction matches the golden reference
  produced in Phase 1/2.
