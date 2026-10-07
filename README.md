# Quantized NN Acceleration on RV64IMC

Term project: design and evaluate architectural extensions (packed INT8 MAC `Xqmac8` +
requantization instruction, plus optional memory/pipeline extensions) for accelerating
quantized neural-network inference on a bare-metal RV64IMC RISC-V core (CV-Wally).

## Status

- [x] Toolchain setup — RV64IMC GCC (xPack prebuilt) + Spike (built from source) verified end-to-end
      via a bare-metal smoke test (see `scripts/build_and_run_smoke.sh`)
- [x] Phase 1 — Model selected (MobileNetV3-Large, real pretrained weights), one representative
      block (`features[3]`) quantized W8A8, exported to static C headers, golden reference
      generated (see `scripts/model_prep/`, `DESIGN.md`)
- [x] Phase 2 (partial) — Bare-metal INT8 kernels (pointwise conv, depthwise conv, requantization)
      implemented and verified bit-exact against the golden reference under Spike
      (`scripts/build_and_run_block3.sh`); more blocks/full model still TBD
- [x] Phase 3 — Spike workload profiling (`scripts/profile_phase3.sh`), top-3 bottlenecks
      identified: (1) scalar per-MAC overhead (~7-9 instr/MAC, only ~11-13% are the `mul` itself),
      (2) depthwise boundary-check branch overhead (28% branch vs 13-16% elsewhere), (3) unpacked
      byte-at-a-time loads with no reuse (12-26% of instructions) — see EXPERIMENTS.md 2026-09-12
- [x] Phase 4 — `Xqmac8` + `Xqrequant` designed, implemented as a real Spike extension
      (`src/spike_ext/xqnn.cc`, custom-0 opcode 0x0b), verified bit-exact (directed + randomized
      tests, `scripts/build_and_run_xqmac_insn.sh`). Measured dynamic-instruction reduction:
      conv1x1 (GEMM/FC) **4.13×** (970k→235k), requantize **1.55×**, depthwise B1 software split
      **1.21×** — meets the required ≥2× target. See `scripts/measure_phase4.sh`,
      `results/phase4_instruction_mix.csv`, EXPERIMENTS.md 2026-10-07.
- [ ] Phase 5 — Midterm report + presentation
- [ ] Phase 6 — CV-Wally RTL port, architecture extension (X2), synthesis, final report

## Model

MobileNetV3-Large (torchvision pretrained, real ImageNet weights, 5.48M params). First
bare-metal target: `features[3]` (InvertedResidual block, no SE). See [DESIGN.md](DESIGN.md).

## Repository structure

```
docs/         background notes and concept write-ups
experiments/  Spike run configs/scripts per configuration (B0/B1/X1/X2), raw logs
results/      processed metrics (CSV/tables) and plots
scripts/      quantization/export scripts, trace-parsing, Spike extension build, measurement
src/          bare-metal C kernels, Spike ISA extension (spike_ext/), verification tests
DESIGN.md     Xqmac/requantization instruction semantics, encoding, rationale
EXPERIMENTS.md running lab notebook: date, config, command, metrics
AI_USAGE.md   disclosure of AI-assisted work
STATUS.md     1-page current-progress summary (read this first)
```

## Build & run

Toolchain: RV64IMC bare-metal GCC (xPack prebuilt `riscv-none-elf-gcc`) + Spike (built from source),
set up under WSL2 Ubuntu. See [docs/toolchain_setup.md](docs/toolchain_setup.md) for install steps.

Smoke test (verifies the whole pipeline):

```bash
./scripts/build_and_run_smoke.sh
```

Block3 kernel correctness (bit-exact vs the golden reference):

```bash
./scripts/build_and_run_block3.sh
```

Phase 4 — build the Spike `xqnn` extension (Xqmac8/Xqrequant), then verify and measure:

```bash
./scripts/build_spike_ext.sh          # installs src/spike_ext/xqnn.cc into Spike and rebuilds it
./scripts/build_and_run_xqmac_insn.sh # functional verification (directed + randomized), exit 0
./scripts/measure_phase4.sh           # B0 vs B1 vs X1 instruction-mix -> results/phase4_instruction_mix.csv
```

## Reference

Project spec: [rv64imc_quantized_nn_term_project.md](rv64imc_quantized_nn_term_project.md)
