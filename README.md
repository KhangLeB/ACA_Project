# Quantized NN Acceleration on RV64IMC

Term project: design and evaluate architectural extensions (packed INT8 MAC `Xqmac8` +
requantization instruction, plus optional memory/pipeline extensions) for accelerating
quantized neural-network inference on a bare-metal RV64IMC RISC-V core (CV-Wally).

## Status

- [ ] Phase 1 — Model selection, W8A8 quantization, static C data export
- [ ] Phase 2 — Bare-metal baseline kernels (B0) + golden-reference verification
- [ ] Phase 3 — Spike workload profiling, bottleneck identification (top 3)
- [ ] Phase 4 — `Xqmac8` + requantization instruction design, Spike implementation, B1/X1
- [ ] Phase 5 — Midterm report + presentation
- [ ] Phase 6 — CV-Wally RTL port, architecture extension (X2), synthesis, final report

## Model

TBD — candidate: MobileNetV3-Large (W8A8, ~0.22 GMAC). See [DESIGN.md](DESIGN.md) for rationale.

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

TBD once bare-metal toolchain (RV64IMC GCC + Spike) is set up. See [docs/](docs/).

## Reference

Project spec: [rv64imc_quantized_nn_term_project.md](rv64imc_quantized_nn_term_project.md)
