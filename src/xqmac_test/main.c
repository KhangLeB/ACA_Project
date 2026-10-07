/* Phase 4 verification: proves the Xqmac8-based conv1x1 is bit-identical to the scalar reference
 * conv1x1_int, and that the Xqmac8 instruction model itself passes directed edge-case tests.
 * Returns 0 on full pass; a nonzero code identifies which check failed (see returns below). */
#include "kernels/kernels.h"
#include "kernels/xqmac.h"
#include <stdint.h>

#define CIN  72
#define COUT 72
#define HW   64

static int8_t  in_buf[CIN * HW];
static int8_t  w_buf[COUT * CIN];
static int32_t acc_ref[COUT * HW];
static int32_t acc_xq[COUT * HW];
static int8_t  out_ref[COUT * HW];
static int8_t  out_xq[COUT * HW];
static int32_t qf_buf[COUT];
static int32_t sh_buf[COUT];

/* xorshift32: deterministic pseudo-random INT8 stream for randomized testing. */
static uint32_t rng = 2463534242u;
static int8_t rnd8(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (int8_t)(uint8_t)(rng & 0xFF);
}

static uint64_t splat8(uint8_t b) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b << (8 * i);
    return v;
}

/* Directed edge cases for the Xqmac8 instruction model. */
static int directed_xqmac8(void) {
    uint64_t amax = splat8(127), amin = splat8(0x80); /* +127, -128 per lane */
    if (xqmac8(0, amax, amax) != 8 * 127 * 127)   return 1; /* max positive product */
    if (xqmac8(0, amin, amin) != 8 * 128 * 128)   return 2; /* two negatives -> positive */
    if (xqmac8(0, amin, amax) != 8 * (-128) * 127) return 3; /* mixed sign */
    if (xqmac8(100, amax, amax) != 100 + 8 * 127 * 127) return 4; /* accumulate onto acc */
    if (xqmac8(0, 0, amax) != 0)                   return 5; /* zero operand */
    return 0;
}

/* Directed edge cases for the Xqrequant instruction model vs the saturation contract. */
static int directed_xqrequant(void) {
    /* identity-ish multiplier 2^30 with shift 0 => multiply by ~0.5, large acc saturates high */
    if (xqrequant(2000000000, 1073741824, 0, 0, -128, 127) != 127) return 10;
    if (xqrequant(-2000000000, 1073741824, 0, 0, -128, 127) != -128) return 11;
    if (xqrequant(0, 1073741824, 0, 5, -128, 127) != 5) return 12; /* zero acc -> zero-point */
    return 0;
}

int main(void) {
    int d;
    if ((d = directed_xqmac8()))   return d;
    if ((d = directed_xqrequant())) return d;

    /* Randomized: three real block3-shaped configs across several activation zero-points. */
    const int configs[3][2] = { {24, 72}, {72, 72}, {72, 24} };
    for (int zp = -10; zp <= 10; zp += 5) {
        for (int i = 0; i < CIN * HW; i++)  in_buf[i] = rnd8();
        for (int i = 0; i < COUT * CIN; i++) w_buf[i] = rnd8();
        for (int c = 0; c < 3; c++) {
            int ci = configs[c][0], co = configs[c][1];
            conv1x1_int(in_buf, zp, w_buf, ci, co, HW, acc_ref);
            conv1x1_xqmac(in_buf, zp, w_buf, ci, co, HW, acc_xq);
            for (int k = 0; k < co * HW; k++)
                if (acc_ref[k] != acc_xq[k]) return 100 + c;
        }

        /* Xqrequant vs scalar requantize (per-channel multipliers, both ReLU and linear). */
        for (int oc = 0; oc < COUT; oc++) {
            qf_buf[oc] = 1000000000 + (int32_t)(uint8_t)rnd8() * 4000000;
            sh_buf[oc] = -8 + (oc % 5);
        }
        for (int i = 0; i < COUT * HW; i++) acc_ref[i] = (int32_t)(rnd8() * 100000 + i);
        for (int relu = 0; relu <= 1; relu++) {
            requantize_per_channel(acc_ref, COUT, HW, qf_buf, sh_buf, zp, relu, out_ref);
            requantize_per_channel_xq(acc_ref, COUT, HW, qf_buf, sh_buf, zp, relu, out_xq);
            for (int k = 0; k < COUT * HW; k++)
                if (out_ref[k] != out_xq[k]) return 200 + relu;
        }

        /* B1 depthwise (interior/border split) vs the baseline depthwise. HW laid as 8x8. */
        depthwise_conv3x3_int(in_buf, zp, w_buf, COUT, 8, 8, acc_ref);
        depthwise_conv3x3_int_b1(in_buf, zp, w_buf, COUT, 8, 8, acc_xq);
        for (int k = 0; k < COUT * 64; k++)
            if (acc_ref[k] != acc_xq[k]) return 210;
    }
    return 0;
}
