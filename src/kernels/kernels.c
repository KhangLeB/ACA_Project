#include "kernels.h"

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
