/* Quantized INT8 conv/requantization kernels, mirroring scripts/model_prep/export_block3.py exactly. */
#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

/* acc[oc][h][w] += sum_ic (in[ic][h][w] - zp_in) * w[oc][ic]; in/out flattened row-major. */
void conv1x1_int(const int8_t *in, int zp_in, const int8_t *w,
                  int c_in, int c_out, int hw, int32_t *acc);

/* Depthwise 3x3, stride 1, pad 1 (zero-point padding), one weight set per channel. */
void depthwise_conv3x3_int(const int8_t *in, int zp_in, const int8_t *w,
                            int channels, int h, int w_dim, int32_t *acc);

/* result = round(acc * real_multiplier), real_multiplier = q_fixed * 2^(shift-31). */
int64_t multiply_by_quantized_multiplier(int64_t acc, int32_t q_fixed, int32_t shift);

/* Per-channel: requantize acc[oc][...] using q_fixed[oc]/shift[oc], add zp_out, clamp [-128,127]
 * (or [zp_out,127] if relu). */
void requantize_per_channel(const int32_t *acc, int c, int hw,
                             const int32_t *q_fixed, const int32_t *shift,
                             int zp_out, int relu, int8_t *out);

#endif /* KERNELS_H */
