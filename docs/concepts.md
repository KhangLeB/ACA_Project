# Concept Notes (Phase 0)

Personal learning notes on fundamentals needed for this project. Fill in as you study each topic.

## 1. Quantized inference
- Affine quantization: `real_value = scale * (q - zero_point)`.
- W8A8: INT8 weights, INT8 activations. Accumulation typically in INT32 to avoid overflow.
- Requantization: convert an INT32 accumulator back to INT8 output —
  multiply by a fixed-point scale factor, round, shift, add output zero-point, saturate to INT8 range.

## 2. RISC-V ISA basics
- RV64IMC = 64-bit base integer (I) + multiply/divide (M) + compressed 16-bit instructions (C).
- No F/D (no floating point) and no V (no vector) in the baseline — all math is integer.
- Custom instructions use reserved opcode space (`custom-0`/`custom-1`/`custom-2`/`custom-3`
  per the RISC-V spec) so they don't collide with standard encodings.

## 3. Spike
- Functional/ISA-level simulator: executes instructions correctly but does not model pipeline
  timing, caches, or stalls accurately — used here for instruction-mix/count analysis, not cycles.
- Custom instructions are added via Spike's extension mechanism (register an opcode + semantics).

## 4. CV-Wally
- An open-source RV core RTL implementation; used in Phase 6 to get real cycle/CPI/area numbers
  for at least one representative kernel, and to host any pipeline/memory extension (X2).

## 5. Metrics glossary
- Dynamic instruction count: instructions actually executed at runtime (not static code size).
- CPI: cycles per instruction (needs RTL/cycle-accurate model, not just Spike).
- MAC/instruction: useful density metric — higher is better for compute-bound kernels.
- Speedup/area: normalizes performance gain by hardware cost, used as headline tradeoff metric.

## Links / references to fill in
- RISC-V ISA manual (custom opcode space section)
- Spike source / extension-writing guide
- CV-Wally repo
- litert-community model hub
