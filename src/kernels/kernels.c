#include "kernels.h"
#include "xqmac.h"

void conv1x1_int(const int8_t *in, int zp_in, const int8_t *w,
                  int c_in, int c_out, int hw, int32_t *acc) {
    for (int oc = 0; oc < c_out; oc++) {
        for (int p = 0; p < hw; p++) {
            int64_t sum = 0;
            for (int ic = 0; ic < c_in; ic++) {
                sum += (int64_t)(in[ic * hw + p] - zp_in) * (int64_t)w[oc * c_in + ic];
            }
            acc[oc * hw + p] = (int32_t)sum;
        }
    }
}

/* Xqmac8-based conv1x1, bit-identical to conv1x1_int. Zero-point is folded out of the inner
 * loop via sum((a-z)*w) = sum(a*w) - z*sum(w), so Xqmac8 sees true packed INT8. For each output
 * pixel the c_in activations are gathered once into a contiguous buffer and reused across all
 * c_out channels, so both operands feed packed 64-bit loads (weights are already contiguous). */
static int32_t xq_colsum[XQMAC_MAX_CH];
static int8_t  xq_avec[XQMAC_MAX_CH];

void conv1x1_xqmac(const int8_t *in, int zp_in, const int8_t *w,
                   int c_in, int c_out, int hw, int32_t *acc) {
    for (int oc = 0; oc < c_out; oc++) {
        int32_t cs = 0;
        for (int ic = 0; ic < c_in; ic++)
            cs += w[oc * c_in + ic];
        xq_colsum[oc] = cs;
    }
    for (int p = 0; p < hw; p++) {
        for (int ic = 0; ic < c_in; ic++)
            xq_avec[ic] = in[ic * hw + p];
        for (int oc = 0; oc < c_out; oc++) {
            const int8_t *wrow = &w[oc * c_in];
            int32_t raw = 0;
            int ic = 0;
            for (; ic + 8 <= c_in; ic += 8)
                raw = xqmac8(raw, xq_ld8(&xq_avec[ic]), xq_ld8(&wrow[ic]));
            for (; ic < c_in; ic++)
                raw += (int32_t)xq_avec[ic] * (int32_t)wrow[ic];
            acc[oc * hw + p] = raw - zp_in * xq_colsum[oc];
        }
    }
}

void depthwise_conv3x3_int(const int8_t *in, int zp_in, const int8_t *w,
                            int channels, int h, int w_dim, int32_t *acc) {
    for (int c = 0; c < channels; c++) {
        const int8_t *wc = &w[c * 9];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w_dim; x++) {
                int64_t sum = 0;
                for (int kh = 0; kh < 3; kh++) {
                    int iy = y + kh - 1;
                    for (int kw = 0; kw < 3; kw++) {
                        int ix = x + kw - 1;
                        int8_t v = (iy < 0 || iy >= h || ix < 0 || ix >= w_dim)
                                       ? (int8_t)zp_in
                                       : in[c * h * w_dim + iy * w_dim + ix];
                        sum += (int64_t)(v - zp_in) * (int64_t)wc[kh * 3 + kw];
                    }
                }
                acc[c * h * w_dim + y * w_dim + x] = (int32_t)sum;
            }
        }
    }
}

int64_t multiply_by_quantized_multiplier(int64_t acc, int32_t q_fixed, int32_t shift) {
    int total_right_shift = 31 - shift;
    int64_t scaled = acc * (int64_t)q_fixed;
    if (total_right_shift > 0) {
        int64_t rounding = (int64_t)1 << (total_right_shift - 1);
        return (scaled + rounding) >> total_right_shift;
    } else if (total_right_shift < 0) {
        return scaled << (-total_right_shift);
    }
    return scaled;
}

void requantize_per_channel(const int32_t *acc, int c, int hw,
                             const int32_t *q_fixed, const int32_t *shift,
                             int zp_out, int relu, int8_t *out) {
    for (int oc = 0; oc < c; oc++) {
        int lower = relu ? zp_out : -128;
        for (int p = 0; p < hw; p++) {
            int64_t scaled = multiply_by_quantized_multiplier(acc[oc * hw + p], q_fixed[oc], shift[oc]);
            int64_t v = scaled + zp_out;
            if (v < lower) v = lower;
            if (v > 127) v = 127;
            out[oc * hw + p] = (int8_t)v;
        }
    }
}

/* Xqrequant-based requantization, bit-identical to requantize_per_channel. */
void requantize_per_channel_xq(const int32_t *acc, int c, int hw,
                                const int32_t *q_fixed, const int32_t *shift,
                                int zp_out, int relu, int8_t *out) {
    for (int oc = 0; oc < c; oc++) {
        int lower = relu ? zp_out : -128;
        for (int p = 0; p < hw; p++) {
            out[oc * hw + p] = xqrequant(acc[oc * hw + p], q_fixed[oc], shift[oc],
                                         zp_out, lower, 127);
        }
    }
}

/* B1 software-optimized depthwise: bit-identical to depthwise_conv3x3_int, but the interior
 * pixels (all 9 taps in-bounds) run with no per-tap boundary check; only the one-pixel border
 * ring pays the zero-padding test. Targets Phase-3 bottleneck #2 (28% branch in depthwise). */
void depthwise_conv3x3_int_b1(const int8_t *in, int zp_in, const int8_t *w,
                               int channels, int h, int w_dim, int32_t *acc) {
    for (int c = 0; c < channels; c++) {
        const int8_t *wc = &w[c * 9];
        const int8_t *inc = &in[c * h * w_dim];
        int32_t *accc = &acc[c * h * w_dim];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w_dim - 1; x++) {
                int64_t sum = 0;
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        sum += (int64_t)(inc[(y + kh - 1) * w_dim + (x + kw - 1)] - zp_in)
                             * (int64_t)wc[kh * 3 + kw];
                accc[y * w_dim + x] = (int32_t)sum;
            }
        }
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w_dim; x++) {
                if (y != 0 && y != h - 1 && x != 0 && x != w_dim - 1)
                    continue; // interior pixel already computed above
                int64_t sum = 0;
                for (int kh = 0; kh < 3; kh++) {
                    int iy = y + kh - 1;
                    for (int kw = 0; kw < 3; kw++) {
                        int ix = x + kw - 1;
                        int8_t v = (iy < 0 || iy >= h || ix < 0 || ix >= w_dim)
                                       ? (int8_t)zp_in
                                       : inc[iy * w_dim + ix];
                        sum += (int64_t)(v - zp_in) * (int64_t)wc[kh * 3 + kw];
                    }
                }
                accc[y * w_dim + x] = (int32_t)sum;
            }
        }
    }
}

