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
