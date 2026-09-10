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
