# Quantized NN Acceleration on RV64IMC

Term project: design and evaluate architectural extensions (packed INT8 MAC `Xqmac8` +
requantization instruction, plus optional memory/pipeline extensions) for accelerating
quantized neural-network inference on a bare-metal RV64IMC RISC-V core (CV-Wally).

**Midterm scope:** ISA-level evaluation only, via the Spike functional simulator (dynamic
instruction count / mix). No RTL, synthesis, cycle-accurate timing, or PPA (power/performance/
area) numbers are part of this submission — those are CV-Wally RTL work for the final submission.

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
slides/       presentation slides (5-6min midterm talk)
src/          bare-metal C kernels, Spike ISA extension (spike_ext/), verification tests
DESIGN.md     Xqmac/requantization instruction semantics, encoding, rationale
EXPERIMENTS.md running lab notebook: date, config, command, metrics
AI_USAGE.md   disclosure of AI-assisted work
```

## Build & run — script, purpose, expected result

Toolchain: RV64IMC bare-metal GCC (xPack prebuilt `riscv-none-elf-gcc`) + Spike (built from source),
set up under WSL2 Ubuntu. See [docs/toolchain_setup.md](docs/toolchain_setup.md) for install steps.

**Running from a Linux/WSL shell already at the repo root:**
```bash
./scripts/build_and_run_smoke.sh
```

**Running from Windows PowerShell** (WSL2 Ubuntu installed, PowerShell's current directory is
already this repo's `Project` folder) — every script must be invoked through `wsl.exe`; WSL
inherits PowerShell's current directory, so no `cd` is needed:
```powershell
wsl -d Ubuntu -- bash -lc "./scripts/build_and_run_smoke.sh"
```
Replace `build_and_run_smoke.sh` with any script name from the table below; the same
`wsl -d Ubuntu -- bash -lc "./scripts/<name>.sh"` pattern runs all of them. All commands below are
written in this exact form since that is how this project was actually built and verified.

| # | Script | Full command (from Windows, via WSL) | Purpose | Expected result |
|---|---|---|---|---|
| 1 | `build_and_run_smoke.sh` | `wsl -d Ubuntu -- bash -lc "./scripts/build_and_run_smoke.sh"` | Toolchain sanity check (minimal bare-metal ELF, M-extension multiply) | `Spike exit code: 0` |
| 2 | `build_and_run_block3.sh` | `wsl -d Ubuntu -- bash -lc "./scripts/build_and_run_block3.sh"` | Build the scalar (B0) kernels at the **real 56×56** block size and compare every output byte against the Python golden reference | `Spike exit code: 0` (bit-exact match) |
| 3 | `build_and_run_xqmac.sh` | `wsl -d Ubuntu -- bash -lc "./scripts/build_and_run_xqmac.sh"` | Verify the `Xqmac8`/`Xqrequant` **software model** (plain C, no custom opcode yet) against the scalar kernels: 5 directed + 3 saturation + randomized tests | `Spike exit code: 0` |
| 4 | `build_spike_ext.sh` | `wsl -d Ubuntu -- bash -lc "./scripts/build_spike_ext.sh"` | Install `src/spike_ext/xqnn.cc` into the Spike source tree and rebuild Spike with the real `Xqmac8`/`Xqrequant` custom-0 instructions | ends with `Spike rebuilt with xqnn extension -> .../spike-install/bin/spike` |
| 5 | `build_and_run_xqmac_insn.sh` | `wsl -d Ubuntu -- bash -lc "./scripts/build_and_run_xqmac_insn.sh"` | Same test suite as #3, but now emitting the **real** `.insn` custom instructions, executed by the Spike extension from #4 | `Spike exit code: 0` |
| 6 | `profile_phase3.sh` | `wsl -d Ubuntu -- bash -lc "./scripts/profile_phase3.sh"` | B0 baseline instruction-mix profiling (load/store/alu/mul/branch) for 5 kernels at 8×8 spatial size | writes `results/phase3_instruction_mix.csv` (5 rows) |
| 7 | `measure_phase4.sh` | `wsl -d Ubuntu -- bash -lc "./scripts/measure_phase4.sh"` | Full B0 vs B1 vs X1 instruction-mix comparison (adds the `custom` opcode category) for conv1x1, requantize, depthwise | writes `results/phase4_instruction_mix.csv` (6 rows, see table below) |

If PowerShell's current directory is NOT already the `Project` folder, `cd` into it first (or add
`cd '/mnt/<drive>/path/to/Project' &&` before `./scripts/...` inside the `bash -lc "..."` string).
Every script was re-run end-to-end on 2026-11-03 using exactly this invocation form; all 7
produced the expected result above with no deviation.

Note on scale: #2 runs the real 56×56 block size (correctness only). #6/#7 use a reduced 8×8
 spatial size with real channel counts, because logging every retired instruction (`--log-commits`)
 at the full 56×56 size is impractically slow; instruction *mix* (the % breakdown) is
 size-invariant for these fixed-trip-count loops, so the 8×8 numbers are a valid proxy — see
 EXPERIMENTS.md (2026-09-12 entry) for the full justification.

## Results summary (from `results/phase4_instruction_mix.csv`)

| Kernel | B0 (scalar) | X1/B1 (Xqmac8/Xqrequant/B1) | Reduction | Custom-instr retired |
|---|---:|---:|---:|---:|
| conv1x1 (GEMM/FC, 24→72ch) | 970,000 instr | 235,000 instr | **4.13×** | 13,824 `xqmac8` |
| requantize (72ch, per-channel) | 85,000 instr | 55,000 instr | **1.55×** | 4,608 `xqrequant` |
| depthwise (72ch, 3×3, B1 = software interior/border split, no new instruction) | 840,000 instr | 695,000 instr | **1.21×** | — |

conv1x1 meets the spec's required ≥×2 dynamic-instruction-reduction target for a major GEMM/FC
kernel. Full load/store/alu/multiply/branch/custom breakdown: `results/phase4_instruction_mix.csv`.

## Reference

Project spec: [rv64imc_quantized_nn_term_project.md](rv64imc_quantized_nn_term_project.md)
