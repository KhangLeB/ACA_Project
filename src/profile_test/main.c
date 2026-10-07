/* Phase 3 profiling harness: exercises one real kernel at a reduced spatial size so Spike's
 * per-instruction commit log (slow) stays tractable. Instruction *mix* is size-invariant for
 * these fixed-trip-count loops, so results scale to the real 56x56 block3 dimensions via the
 * known iteration-count formulas documented in EXPERIMENTS.md. Data values are synthetic
 * (deterministic filler, not real weights/activations) because instruction mix depends only on
 * the code path and loop trip counts, not on operand values, for these kernels. Select one
 * kernel via -DBENCH_CONV1X1 / -DBENCH_DEPTHWISE / -DBENCH_REQUANT / -DBENCH_FULL /
 * -DBENCH_CONV1X1_XQ (the Xqmac8 version of conv1x1, for the X1 vs B0 instruction-count compare). */
#include "kernels/kernels.h"
#include <stdint.h>

#ifndef P_CIN
#define P_CIN 24
#endif
#ifndef P_MID
#define P_MID 72
#endif
#ifndef P_DIM
#define P_DIM 8
#endif
#define P_HW (P_DIM * P_DIM)

static int8_t in_buf[P_MID * P_HW];
static int8_t w_buf[P_MID * P_MID]; /* oversized: covers both cin*cout (1x1) and channels*9 (dw) */
static int32_t acc_buf[P_MID * P_HW];
static int8_t out_buf[P_MID * P_HW];
static int32_t qfixed_buf[P_MID];
static int32_t shift_buf[P_MID];

static void fill(void) {
    for (int i = 0; i < P_MID * P_HW; i++) in_buf[i] = (int8_t)(i * 3 + 7);
    for (int i = 0; i < P_MID * P_MID; i++) w_buf[i] = (int8_t)(i * 5 + 1);
    for (int i = 0; i < P_MID; i++) { qfixed_buf[i] = 1073741824; shift_buf[i] = 0; }
}

int main(void) {
    fill();
#if defined(BENCH_CONV1X1)
    conv1x1_int(in_buf, 0, w_buf, P_CIN, P_MID, P_HW, acc_buf);
#elif defined(BENCH_CONV1X1_XQ)
    conv1x1_xqmac(in_buf, 0, w_buf, P_CIN, P_MID, P_HW, acc_buf);
#elif defined(BENCH_DEPTHWISE)
    depthwise_conv3x3_int(in_buf, 0, w_buf, P_MID, P_DIM, P_DIM, acc_buf);
#elif defined(BENCH_DEPTHWISE_B1)
    depthwise_conv3x3_int_b1(in_buf, 0, w_buf, P_MID, P_DIM, P_DIM, acc_buf);
#elif defined(BENCH_REQUANT)
    requantize_per_channel(acc_buf, P_MID, P_HW, qfixed_buf, shift_buf, 0, 1, out_buf);
#elif defined(BENCH_REQUANT_XQ)
    requantize_per_channel_xq(acc_buf, P_MID, P_HW, qfixed_buf, shift_buf, 0, 1, out_buf);
#elif defined(BENCH_FULL)
    conv1x1_int(in_buf, 0, w_buf, P_CIN, P_MID, P_HW, acc_buf);
    requantize_per_channel(acc_buf, P_MID, P_HW, qfixed_buf, shift_buf, 0, 1, out_buf);
    depthwise_conv3x3_int(out_buf, 0, w_buf, P_MID, P_DIM, P_DIM, acc_buf);
    requantize_per_channel(acc_buf, P_MID, P_HW, qfixed_buf, shift_buf, 0, 1, out_buf);
#else
#error "define one of BENCH_CONV1X1 / BENCH_CONV1X1_XQ / BENCH_DEPTHWISE / BENCH_DEPTHWISE_B1 / BENCH_REQUANT / BENCH_REQUANT_XQ / BENCH_FULL"
#endif
    return 0;
}
