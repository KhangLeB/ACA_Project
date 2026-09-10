/* Bare-metal integer-only forward pass for MobileNetV3-Large features[3] (InvertedResidual,
 * no SE): expand 1x1 conv -> depthwise 3x3 conv -> project 1x1 conv -> residual add.
 * Must reproduce scripts/model_prep/export_block3.py bit-for-bit; verified against
 * block3_golden_output. Returns 0 on exact match (see src/common/crt0.S for exit encoding). */
#include "kernels/kernels.h"
#include "model_data/block3_params.h"
#include "model_data/block3_golden.h"

#define HW (BLOCK3_H * BLOCK3_W)

static int32_t acc0[BLOCK3_MID * HW];
static int8_t mid0[BLOCK3_MID * HW];
static int32_t acc1[BLOCK3_MID * HW];
static int8_t mid1[BLOCK3_MID * HW];
static int32_t acc2[BLOCK3_CIN * HW];
static int8_t pre_residual[BLOCK3_CIN * HW];
static int8_t final_out[BLOCK3_CIN * HW];

int main(void) {
    /* expand: 24 -> 72, 1x1 conv, +bias, requantize with ReLU */
    conv1x1_int(block3_test_input, BLOCK3_ZP_IN, block3_w0_q, BLOCK3_CIN, BLOCK3_MID, HW, acc0);
    for (int oc = 0; oc < BLOCK3_MID; oc++)
        for (int p = 0; p < HW; p++)
            acc0[oc * HW + p] += (int32_t)block3_b0_q[oc];
    requantize_per_channel(acc0, BLOCK3_MID, HW, block3_m0_fixed, block3_m0_shift,
                            BLOCK3_ZP_MID0, 1, mid0);

    /* depthwise: 72 channels, 3x3 conv, +bias, requantize with ReLU */
    depthwise_conv3x3_int(mid0, BLOCK3_ZP_MID0, block3_w1_q, BLOCK3_MID, BLOCK3_H, BLOCK3_W, acc1);
    for (int oc = 0; oc < BLOCK3_MID; oc++)
        for (int p = 0; p < HW; p++)
            acc1[oc * HW + p] += (int32_t)block3_b1_q[oc];
    requantize_per_channel(acc1, BLOCK3_MID, HW, block3_m1_fixed, block3_m1_shift,
                            BLOCK3_ZP_MID1, 1, mid1);

    /* project: 72 -> 24, 1x1 conv, +bias, requantize (linear, no ReLU) */
    conv1x1_int(mid1, BLOCK3_ZP_MID1, block3_w2_q, BLOCK3_MID, BLOCK3_CIN, HW, acc2);
    for (int oc = 0; oc < BLOCK3_CIN; oc++)
        for (int p = 0; p < HW; p++)
            acc2[oc * HW + p] += (int32_t)block3_b2_q[oc];
    requantize_per_channel(acc2, BLOCK3_CIN, HW, block3_m2_fixed, block3_m2_shift,
                            BLOCK3_ZP_MID2, 0, pre_residual);

    /* residual add: dequantize both branches to a shared scale, add, requantize to output */
    for (int i = 0; i < BLOCK3_CIN * HW; i++) {
        int64_t left = multiply_by_quantized_multiplier(
            (int64_t)(pre_residual[i] - BLOCK3_ZP_MID2), BLOCK3_ADD_MULT_A, BLOCK3_ADD_SHIFT_A);
        int64_t right = multiply_by_quantized_multiplier(
            (int64_t)(block3_test_input[i] - BLOCK3_ZP_IN), BLOCK3_ADD_MULT_B, BLOCK3_ADD_SHIFT_B);
        int64_t v = left + right + BLOCK3_ZP_OUT;
        if (v < -128) v = -128;
        if (v > 127) v = 127;
        final_out[i] = (int8_t)v;
    }

    for (int i = 0; i < BLOCK3_CIN * HW; i++) {
        if (final_out[i] != block3_golden_output[i]) return 1;
    }
    return 0;
}
