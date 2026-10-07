/* Software functional models of the two proposed custom instructions (Phase 4).
 *
 * These C functions define the EXACT semantics that will later be implemented inside Spike
 * (and eventually CV-Wally RTL). Keeping them here first lets us (a) validate the algorithm
 * bit-exactly against the existing scalar kernels under the normal toolchain, and (b) hand the
 * same reference to the Spike extension and the RTL testbench so all three agree by construction.
 *
 * Design decisions locked here (see DESIGN.md "Xqmac8" / "Requantization instruction"):
 *  - Operands are two 64-bit GPRs, each holding 8 packed signed INT8 lanes (lane i = bits [8i+7:8i]).
 *  - Accumulator is a 32-bit GPR, read-modify-write (rd = rd + dot8(rs1, rs2)). 32 bits is ample:
 *    worst case 8*128*128 = 131072 per instruction, and a full K=512 reduction stays < 2^24.
 *  - Xqmac8 does NOT subtract the activation zero-point. The kernel removes it once per output
 *    channel via the identity  sum((a-z)*w) = sum(a*w) - z*sum(w)  (weights are symmetric, zp_w=0),
 *    so the packed operands stay true INT8 and the 8-lane packing is never broken.
 */
#ifndef XQMAC_H
#define XQMAC_H

#include <stdint.h>

/* Load 8 consecutive signed bytes as one little-endian 64-bit packed word (a packed load feeding
 * the instruction). The direct 64-bit access compiles to a single `ld`; RV64 permits the
 * misaligned weight-row case and Spike executes it. This is the operand-packing win Xqmac8 relies
 * on, so it must NOT be a bytewise loop. */
static inline uint64_t xq_ld8(const int8_t *p) {
    return *(const uint64_t *)(const void *)p;
}

#ifdef XQMAC_USE_INSN
/* Real custom instructions (Spike extension "xqnn", opcode 0x0b, funct3=0b111). The kernels stay
 * byte-for-byte identical to the pure-C build; only these two primitives change, so a bit-exact
 * pass here proves the hardware semantics match the software model. */

/* Xqmac8: rd = rd + dot8(rs1, rs2), funct7=0. rd is read-modify-write ("+r"). */
static inline int32_t xqmac8(int32_t acc, uint64_t rs1, uint64_t rs2) {
    int64_t a = acc;
    __asm__ volatile(".insn r 0x0b, 7, 0, %0, %1, %2"
                     : "+r"(a) : "r"(rs1), "r"(rs2));
    return (int32_t)a;
}

/* Xqrequant: rd = requantize(acc, packed-params), funct7=1. Packs the scalar params the same way
 * the extension unpacks them; lo==zp signals the ReLU saturation mode, hi is always 127. */
static inline int8_t xqrequant(int32_t acc, int32_t q_fixed, int32_t shift,
                               int32_t zp, int32_t lo, int32_t hi) {
    (void)hi;
    uint64_t params = (uint64_t)(uint32_t)q_fixed
                    | ((uint64_t)(uint8_t)(int8_t)shift << 32)
                    | ((uint64_t)(uint8_t)(int8_t)zp << 40)
                    | ((uint64_t)((lo == zp) ? 1u : 0u) << 48);
    uint64_t a = (uint64_t)(int64_t)acc;
    int64_t rd;
    __asm__ volatile(".insn r 0x0b, 7, 1, %0, %1, %2"
                     : "=r"(rd) : "r"(a), "r"(params));
    return (int8_t)rd;
}

#else
/* Xqmac8: acc += sum_{i=0..7} int8(rs1 lane i) * int8(rs2 lane i). */
static inline int32_t xqmac8(int32_t acc, uint64_t rs1, uint64_t rs2) {
    for (int i = 0; i < 8; i++) {
        int8_t a = (int8_t)(uint8_t)(rs1 >> (8 * i));
        int8_t b = (int8_t)(uint8_t)(rs2 >> (8 * i));
        acc += (int32_t)a * (int32_t)b;
    }
    return acc;
}

/* Xqrequant: fixed-point multiply -> round -> arithmetic shift -> add zero-point -> saturate.
 * Mirrors multiply_by_quantized_multiplier()+clamp in kernels.c exactly (real_multiplier =
 * q_fixed * 2^(shift-31)). lo/hi are the saturation bounds (e.g. [-128,127] or [zp,127] for ReLU). */
static inline int8_t xqrequant(int32_t acc, int32_t q_fixed, int32_t shift,
                               int32_t zp, int32_t lo, int32_t hi) {
    int total_right_shift = 31 - shift;
    int64_t scaled = (int64_t)acc * (int64_t)q_fixed;
    int64_t r;
    if (total_right_shift > 0) {
        int64_t rounding = (int64_t)1 << (total_right_shift - 1);
        r = (scaled + rounding) >> total_right_shift;
    } else if (total_right_shift < 0) {
        r = scaled << (-total_right_shift);
    } else {
        r = scaled;
    }
    int64_t v = r + zp;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return (int8_t)v;
}
#endif /* XQMAC_USE_INSN */

#endif /* XQMAC_H */
