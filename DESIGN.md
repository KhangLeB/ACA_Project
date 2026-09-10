# ISA Extension Design

## Status
Draft — to be filled in during Phase 4, after Spike bottleneck profiling (Phase 3) is complete.
Instruction semantics must be justified by measured bottlenecks, not chosen in advance.

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
