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
- [ ] Phase 3 — Spike workload profiling, bottleneck identification (top 3)
- [ ] Phase 4 — `Xqmac8` + requantization instruction design, Spike implementation, B1/X1
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
scripts/      quantization/export scripts, trace-parsing, plotting
src/          bare-metal C kernels, Spike ISA extension, CV-Wally RTL diffs
DESIGN.md     Xqmac/requantization instruction semantics, encoding, rationale
EXPERIMENTS.md running lab notebook: date, config, command, metrics
AI_USAGE.md   disclosure of AI-assisted work
```

## Build & run

Toolchain: RV64IMC bare-metal GCC (xPack prebuilt `riscv-none-elf-gcc`) + Spike (built from source),
set up under WSL2 Ubuntu. See [docs/toolchain_setup.md](docs/toolchain_setup.md) for install steps.

Smoke test (verifies the whole pipeline):

```bash
./scripts/build_and_run_smoke.sh
```

## Reference

Project spec: [rv64imc_quantized_nn_term_project.md](rv64imc_quantized_nn_term_project.md)
