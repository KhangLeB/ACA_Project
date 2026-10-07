# Experiment Log

Append-only running log. One entry per experiment/measurement run. This is the raw evidence
base for the final report's measurement -> architecture -> evaluation narrative.

## Template

```
### YYYY-MM-DD — <short title>
- Config: B0 / B1 / X1 / X2
- Command: <exact command used>
- Metrics: <dynamic instr count, instr mix, MAC/instr, cycles, etc.>
- Observation: <what this tells us>
- Next step: <what this motivates>
```

---

### 2026-09-10 — Toolchain setup + end-to-end smoke test
- Config: N/A (environment setup, not a kernel benchmark)
- Command: `./scripts/build_and_run_smoke.sh` (WSL2 Ubuntu; xPack riscv-none-elf-gcc 14.2.0 + Spike
  built from source, see docs/toolchain_setup.md)
- Metrics: Spike exit code 0 (bare-metal rv64imc ELF built, HTIF tohost exit worked, multiply
  (M-extension) sanity check 6*7==42 passed)
- Observation: Full pipeline (GCC cross-compile -> link -> Spike execution) is functional.
- Next step: Phase 1 — model selection + W8A8 quantization + static C export.

### 2026-09-10 — Model selection + block3 quantization + bare-metal kernel verification
- Config: N/A (Phase 1/2 functional verification, not yet a Spike instruction-count measurement)
- Command: `scripts/model_prep/export_block3.py` -> `export_to_c.py` -> `scripts/build_and_run_block3.sh`
- Model: MobileNetV3-Large (torchvision pretrained), block `features[3]` (InvertedResidual,
  no SE): expand 1x1 conv (24->72) -> depthwise 3x3 conv (72ch) -> project 1x1 conv (72->24)
  -> residual add, at 56x56 spatial resolution.
- Metrics: INT8 quantized (Python reference) vs float model: max_abs_err=4.93, rel_err=4.87%
  (expected — synthetic Gaussian calibration data, not representative images; irrelevant to
  kernel-correctness grading). Bare-metal C kernel output vs Python golden reference: **bit-exact
  match** (Spike exit code 0).
- Observation: Quantization scheme (per-channel symmetric weights, per-tensor asymmetric
  activations, TFLite-style multiplier+shift requantization) is implemented consistently in both
  Python and C. Required `-mcmodel=medany` for the C build (RAM base 0x80000000 overflows the
  default `medlow` model's absolute addressing for large static arrays).
- Next step: Phase 3 — profile this block (and a simple GEMM microbenchmark) under Spike for
  dynamic instruction count / mix; then Phase 4 — design Xqmac8 informed by that profiling.

### 2026-09-12 — Phase 3: Spike instruction-mix profiling (B0 baseline)
- Config: B0 (baseline RV64IMC, no Xqmac)
- Command: `scripts/profile_phase3.sh` (builds 5 targets, runs each under
  `spike -l --log-commits`, joins the PC trace against `objdump -d` static disassembly via
  `scripts/parse_spike_log.py`, writes `results/phase3_instruction_mix.csv`)
- Methodology notes (important, affects how these numbers should be read):
  - Spike's `--log-commits` on this build prints only `pc` + raw encoding, no mnemonic — the
    parser builds a `pc -> mnemonic` map from `objdump -d` of the same ELF and joins it against
    the dynamic PC trace to get the *dynamic* instruction mix.
  - A direct `--log-commits` run of the real block3 size (56x56, i.e. ~100M+ instructions) was
    attempted first and is impractically slow (many minutes, multi-GB log) — killed after no
    progress. Switched to reduced spatial size (8x8 = 64 "pixels") profiling harnesses
    (`src/profile_test/main.c`, `src/microbench/gemm_bench.c`) that call the *real* kernel
    functions from `kernels.c` with the real channel counts (24/72) but synthetic filler data.
    This is valid because instruction *mix* for these fixed-trip-count loops depends on the code
    path and loop structure, not on operand values (except the depthwise boundary-padding branch,
    which depends on position, not value, and is fully exercised at 8x8). Absolute counts scale
    to the real 56x56 size via the known iteration-count formulas below (loops have no
    data-dependent trip counts).
  - `/tmp` inside WSL is not reliable across separate `wsl.exe` invocations (can reset) — logs are
    written under `experiments/logs/` on the mounted drive instead.
  - Bash arrays (not string variables) must be used for compiler flags/sources because the repo
    path contains spaces (`Advanced Computer Architecture`) — a plain `"$CFLAGS"` string re-splits
    on those spaces.
  - `-fno-builtin -fno-tree-loop-distribute-patterns` needed: without them GCC turns some
    zero-init loops into `memset` calls, which don't exist in this `-nostdlib` build (link error).
  - The GEMM microbenchmark's output array must be `volatile`, otherwise `-O2` proves the result
    is unused (single translation unit) and deletes the entire triple-nested loop (measured only
    15,000 instructions instead of the expected ~250k before this fix).
- Targets profiled (8x8 spatial internally, real channel counts, real kernel code):
  | Target | Total instr | load | store | alu | multiply | branch | MACs (theory) | instr/MAC | (load+store)/MAC |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | gemm32 (32x32x32 microbenchmark) | 255,000 | 65,536 (25.7%) | 3,073 (1.2%) | 112,886 (44.3%) | 32,768 (12.9%) | 40,732 (16.0%) | 32,768 | 7.78 | 2.09 |
  | conv1x1 (expand, 24→72, 1x1) | 970,000 | 221,190 (22.8%) | 14,407 (1.5%) | 495,547 (51.1%) | 110,592 (11.4%) | 128,259 (13.2%) | 110,592 | 8.77 | 2.13 |
  | depthwise (72ch, 3x3) | 840,000 | 100,669 (12.0%) | 34,142 (4.1%) | 428,663 (51.0%) | 41,473 (4.9%) | 235,048 (28.0%) | 41,472 | 20.26 | 3.25 |
  | requantize (per-channel, 72ch) | 85,000 | 13,831 (16.3%) | 4,760 (5.6%) | 42,311 (49.8%) | 4,608 (5.4%) | 19,485 (22.9%) | n/a (0 MAC; 1 fixed-point mul/output elem) | n/a | n/a |
  | block3_full (expand→requant→depthwise→requant) | 1,935,000 | 349,518 (18.1%) | 48,127 (2.5%) | 988,177 (51.1%) | 161,281 (8.3%) | 387,892 (20.0%) | — | — | — |
  - Cross-check: measured `multiply` count equals the theoretical scalar-MAC count almost exactly
    in every kernel (conv1x1: 110,592 = 110,592; depthwise: 41,473 ≈ 41,472; requantize:
    4,608 = 72×64 fixed-point multiplies; gemm32: 32,768 = 32,768) — confirms the naive C kernels
    compile to exactly one `mul` instruction per MAC/multiply, with no instruction-level reuse.
  - Code size (`riscv-none-elf-size`, .text bytes): gemm32 198, requantize 924, depthwise 940,
    conv1x1 942, block3_full 1118.
- **Three largest bottlenecks identified (ranked by evidence):**
  1. **Per-MAC scalar overhead dominates over the multiply itself.** Only 11–13% of instructions
     in the GEMM/conv1x1 kernels are the actual `mul`; ALU ops (sign-extension of INT8 operands,
     zero-point subtraction, address computation) are 44–51% of all instructions, i.e. ~7–9 total
     instructions retired per useful MAC. This is exactly what a packed `Xqmac8` targets: folding
     8 loads + 8 sign-extends + 8 muls + 8 adds into one instruction removes the dominant category.
  2. **Branch/loop overhead is concentrated in depthwise conv (28.0% branch vs 13.2–16.0%
     elsewhere)**, caused by the per-tap zero-padding boundary check (`iy<0||iy>=h||ix<0||ix>=w`)
     evaluated at every one of the 9 taps regardless of whether the pixel is actually a border
     pixel. This motivates either a hardware-loop extension or restructuring the kernel to
     separate interior/border passes (software optimization for B1) before any hardware fix.
  3. **Load instructions are a large, unreduced fraction (12–26%) with no data reuse** — every
     operand is loaded individually as a single byte with no packing, register blocking, or
     reuse across output pixels (e.g. conv1x1's weight row is reloaded from memory for every
     spatial position `p`). This is a secondary target for either `Xqmac8`'s packed-load operand
     format or a small scratchpad/register-blocking software optimization (B1).
  - Requantization overhead (18.4 instr per output element for a conceptually "single" operation)
    is real but smaller in absolute weight than (1)-(3) at this point; still a strong case for the
    required requantization instruction, to be quantified further once Xqmac8 changes the mix.
- Next step: Phase 4 — design `Xqmac8` (packed 8×INT8 MAC) and the requantization instruction
  informed directly by bottleneck (1) above; implement B1 (loop restructuring for bottleneck 2)
  and X1 in Spike; re-measure with the same `profile_phase3.sh` methodology for a B0 vs B1 vs X1
  comparison.

### 2026-10-07 — Phase 4 step 1: Xqmac8/Xqrequant software models + bit-exact kernel rewrite
- Config: X1 (algorithm model only; not yet a real custom instruction in Spike — that is step 2)
- Command: `scripts/build_and_run_xqmac.sh`
- What was built: software functional models of both proposed instructions in
  `src/kernels/xqmac.h` (`xqmac8`: acc += dot product of 8 packed signed-INT8 lanes, 32-bit
  accumulator; `xqrequant`: fixed-point multiply→round→shift→+zero-point→saturate). Rewrote
  conv1x1 as `conv1x1_xqmac` in `src/kernels/kernels.c` using `xqmac8`, with two design
  decisions that resolve the two main Phase-4 risks:
  1. **Zero-point folding** — `sum((a-z)*w) = sum(a*w) - z*sum(w)`, applied once per output
     channel (weights symmetric, zp_w=0), so Xqmac8 operands stay true 8-bit (never widened to
     9-bit by subtracting zp inside the loop).
  2. **Gather-once-per-pixel** — each output pixel's c_in activations are gathered into a
     contiguous buffer once and reused across all c_out channels; weights are already contiguous
     in the reduction dim, so both operands feed packed 64-bit loads despite the CHW layout. This
     also attacks Phase-3 bottleneck #3 (no activation reuse).
- Metrics: Spike exit code 0. Verification coverage: 5 directed Xqmac8 edge cases (max positive
  product 8·127·127, double-negative 8·128·128, mixed sign, accumulate-onto-acc, zero operand),
  3 directed Xqrequant saturation cases, and a randomized bit-exact comparison of `conv1x1_xqmac`
  vs `conv1x1_int` over three block3-shaped configs (24→72, 72→72, 72→24) × five activation
  zero-points (−10…+10). All passed.
- Observation: the Xqmac8 algorithm (packing + zero-point correction + layout) is proven correct
  under the normal toolchain before touching Spike's C++ — de-risks the decoder work. Instruction
  *count* reduction is not measured yet because `xqmac8()` still expands to scalar ops here; that
  number comes in step 2 once it is a single retired instruction in Spike.
- Next step: Phase 4 step 2 — add Xqmac8 + Xqrequant as real custom instructions in the Spike C++
  source (decoder + execute), rebuild Spike, re-run `profile_phase3.sh` to get the X1 instruction
  mix, and produce the B0 vs B1 vs X1 comparison. Also still pending: implement B1 (depthwise
  interior/border loop split) and push Phase 3 + Phase 4 artifacts to GitHub.

### 2026-10-07 — Phase 4 step 2: Xqmac8/Xqrequant as real Spike instructions + X1 vs B0 count
- Config: X1 (RV64IMC + Xqmac8/Xqrequant) vs B0 (baseline RV64IMC)
- Commands: `scripts/build_spike_ext.sh` (install+build the extension into Spike),
  `scripts/build_and_run_xqmac_insn.sh` (functional verify), `scripts/measure_x1_vs_b0.sh` (counts)
- Implementation: `src/spike_ext/xqnn.cc` — a ROCC-style extension on custom-0 opcode 0x0b
  (funct3=0b111 so ROCC supplies rs1/rs2 and writes rd; funct7=0 → xqmac8, funct7=1 → xqrequant).
  xqmac8 reads the old rd via `state->XPR[insn.rd]` to implement a 32-bit read-modify-write
  accumulator entirely in a GPR (no internal accumulator file). The kernels emit the instructions
  via `.insn r` inline asm behind `-DXQMAC_USE_INSN` in `src/kernels/xqmac.h`, so the exact same
  `conv1x1_xqmac` source is bit-exact in both the pure-C and real-instruction builds.
- Build gotcha (recorded for reproducibility): this Spike's `--extension` path dlopens
  libcustomext.so, which statically bundles libriscv.a and re-runs the built-in MMIO device
  registrars → aborts with `Plugin "imsic_mmio" already registered` (happens even with the stock
  dummy_rocc). Fixed by linking `xqnn.o` straight into the spike binary via `spike_main_LDFLAGS`
  so the extension registers itself at startup and `find_extension()` never dlopens. The Makefile
  does not track LDFLAGS objects as prerequisites, so the build script force-removes the stale
  `spike` binary to trigger a relink.
- Functional: Spike exit 0 on the real-instruction build — 5 directed xqmac8 edge cases, 3
  xqrequant saturation cases, and randomized `conv1x1_xqmac` (now emitting real xqmac8) vs scalar
  `conv1x1_int` over {24→72,72→72,72→24} × zp{−10..+10} all pass. Satisfies the spec's "Xqmac
  passes directed and randomized verification tests" and "kernel output matches the reference".
- Performance (conv1x1 24→72, 8×8 spatial, real kernel, dynamic instruction count via
  `--log-commits`, one retire line per instruction):
  | Config | Dynamic instrs | instr/MAC | note |
  |---|---:|---:|---|
  | B0 scalar  | 970,000 | 8.77 | matches the Phase-3 B0 number exactly |
  | X1 Xqmac8  | 235,000 | 2.13 | **4.13× fewer instructions** |
  - MACs = 24×72×64 = 110,592. Retired xqmac8 instructions = **13,824** = 72×64×(24/8), i.e. each
    8-lane instruction does exactly 8 MACs (confirmed against the log), and custom-instruction
    frequency = 13,824 / 235,000 = **5.9%** of X1's dynamic instructions.
  - **Meets the required target** (≥2× dynamic-instruction reduction for a major GEMM/FC kernel),
    comfortably at 4.13×.
  - Codegen note (important for reproducibility): the packed operand load `xq_ld8()` MUST compile
    to a single `ld`. An initial `__builtin_memcpy` version made the xPack GCC emit conservative
    bytewise (misaligned-safe) loads, inflating X1 to 475k (only 2.04×); switching to a direct
    `*(const uint64_t*)` access (RV64 permits misaligned; Spike executes it) restored the single
    `ld` and the full 4.13×. Still bit-exact.
  - This 8×8 number is itself conservative: the per-output-channel column-sum and per-pixel gather
    are fixed/low-order costs that amortize far better at the real 56×56 size (hw=3136 vs 64).
- X1 instruction mix vs B0 (`scripts/measure_phase4.sh` → `results/phase4_instruction_mix.csv`):
  | kernel | total | load | store | alu | multiply | branch | custom |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | conv1x1 B0 | 970,000 | 22.8% | 1.5% | 51.1% | 11.4% | 13.2% | 0% |
  | conv1x1 X1 | 235,000 | 15.1% | 6.8% | 51.2% | 2.0% | 19.1% | 5.9% |
  - The scalar `mul` collapses from 11.4%→2.0% (110,592→4,608; the residual muls are address and
    the `zp·colsum` correction), replaced by 13,824 packed `xqmac8`. Loads drop 6.2× (221k→35.5k)
    from packed 64-bit operand loads. **New dominant category is ALU (51%)** — address generation,
    loop control, and the per-output `raw − zp·colsum` — which is what the next extensions
    (hardware loops, post-increment load/store, fused ops) should target.
- Requantize with Xqrequant (per-channel, 72ch, 8×8): **85,000 → 55,000 = 1.55×**. The fixed-point
  multiply→round→shift→+zp→saturate collapses into one `xqrequant`: `multiply` 4,608→0 and `branch`
  22.9%→14.4%; 4,608 custom instructions (8.4% of X1). Directed saturation + randomized per-channel
  tests pass (both ReLU and linear), bit-exact vs `requantize_per_channel`.
- B1 software optimization — depthwise interior/border split (no new instruction, `depthwise_
  conv3x3_int_b1`): **840,000 → 695,000 = 1.21×**, with `branch` 235,048 (28.0%) → 147,642 (21.2%),
  i.e. ~37% fewer branches, directly removing the per-tap boundary check on the 36/64 interior
  pixels (bottleneck #2). Bit-exact vs the baseline depthwise. This isolates a software-only gain,
  keeping the ISA (X1) and microarchitecture gains separable as the spec requires.
- Next step: implement B1 for the pointwise path too if useful, then push Phase 3 + Phase 4 to
  GitHub (remote only has up to Phase 1), and write the midterm report from these numbers.
