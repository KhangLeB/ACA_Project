"""
Extract MobileNetV3-Large features[3] (InvertedResidual block, no SE), fold BatchNorm into
the preceding conv, quantize to INT8 (per-channel symmetric weights, per-tensor asymmetric
activations, TFLite-style multiplier+shift requantization), run an integer-only reference
forward pass on a fixed calibration input, and export weights + golden I/O as C headers.
"""
import math
import os
import numpy as np
import torch
import torchvision

QMIN, QMAX = -128, 127
SEED = 1234


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def asym_qparams(x: np.ndarray):
    """Per-tensor asymmetric INT8 quantization params from observed min/max."""
    xmin = min(0.0, float(x.min()))
    xmax = max(0.0, float(x.max()))
    if xmax == xmin:
        xmax = xmin + 1e-6
    scale = (xmax - xmin) / (QMAX - QMIN)
    zero_point = round(QMIN - xmin / scale)
    zero_point = int(np.clip(zero_point, QMIN, QMAX))
    return scale, zero_point


def quantize_asym(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(x / scale) + zero_point
    return np.clip(q, QMIN, QMAX).astype(np.int32)


def quantize_weight_per_channel(w: np.ndarray):
    """w shape: (out_channels, ...). Symmetric per-output-channel INT8 (zero_point=0)."""
    oc = w.shape[0]
    flat = w.reshape(oc, -1)
    scale = np.abs(flat).max(axis=1) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.round(flat / scale[:, None])
    q = np.clip(q, -127, 127).astype(np.int32).reshape(w.shape)
    return scale.astype(np.float64), q


def quantize_multiplier(real_multiplier: float):
    """Decompose a positive real multiplier < 1 into (q_fixed int32, shift) such that
    real_multiplier == q_fixed * 2**(shift - 31)."""
    if real_multiplier == 0.0:
        return 0, 0
    m, e = math.frexp(real_multiplier)  # real_multiplier = m * 2**e, 0.5 <= m < 1
    q_fixed = int(round(m * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        e += 1
    return q_fixed, e


def multiply_by_quantized_multiplier(acc: np.ndarray, q_fixed: int, shift: int) -> np.ndarray:
    """acc is int64 numpy array; returns rounded acc * real_multiplier as int64."""
    total_right_shift = 31 - shift
    acc = acc.astype(np.int64) * np.int64(q_fixed)
    if total_right_shift > 0:
        rounding = np.int64(1) << (total_right_shift - 1)
        return (acc + rounding) >> total_right_shift
    elif total_right_shift < 0:
        return acc << (-total_right_shift)
    return acc


def requantize(acc_int32: np.ndarray, q_fixed: int, shift: int, zero_point_out: int,
               relu: bool) -> np.ndarray:
    scaled = multiply_by_quantized_multiplier(acc_int32, q_fixed, shift)
    out = scaled + zero_point_out
    lower = zero_point_out if relu else QMIN
    return np.clip(out, lower, QMAX).astype(np.int32)


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------
def fold_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    w = conv.weight.detach().numpy().astype(np.float64)  # (oc, ic/groups, kh, kw)
    gamma = bn.weight.detach().numpy().astype(np.float64)
    beta = bn.bias.detach().numpy().astype(np.float64)
    mean = bn.running_mean.detach().numpy().astype(np.float64)
    var = bn.running_var.detach().numpy().astype(np.float64)
    eps = bn.eps
    std = np.sqrt(var + eps)
    w_folded = w * (gamma / std)[:, None, None, None]
    b_folded = beta - gamma * mean / std
    return w_folded, b_folded


# ---------------------------------------------------------------------------
# Integer-only conv ops (numpy, small tensors -> direct implementation is fine)
# ---------------------------------------------------------------------------
def conv1x1_int(x_int, zp_in, w_int, groups=1):
    """x_int: (C_in, H, W) int32 (already includes zero-point offset applied by caller? no,
    raw quantized values). w_int: (C_out, C_in, 1, 1). Returns int32 acc (C_out, H, W)."""
    c_in, h, w = x_int.shape
    c_out = w_int.shape[0]
    x_shifted = (x_int - zp_in).reshape(c_in, h * w).astype(np.int64)  # (Cin, HW)
    wk = w_int.reshape(c_out, c_in).astype(np.int64)                  # (Cout, Cin)
    acc = wk @ x_shifted  # (Cout, HW)
    return acc.reshape(c_out, h, w)


def depthwise_conv3x3_int(x_int, zp_in, w_int, pad=1):
    """x_int: (C, H, W) int32. w_int: (C, 1, 3, 3). Stride 1, same padding. Pads with
    zero-point value so padded contribution is exactly zero after subtracting zp_in."""
    c, h, w = x_int.shape
    xp = np.pad(x_int, ((0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=zp_in)
    xp = (xp - zp_in).astype(np.int64)
    acc = np.zeros((c, h, w), dtype=np.int64)
    for kh in range(3):
        for kw in range(3):
            wk = w_int[:, 0, kh, kw].astype(np.int64)[:, None, None]
            acc += wk * xp[:, kh:kh + h, kw:kw + w]
    return acc


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )
    model.eval()

    # Calibration batch: synthetic ImageNet-normalized random images.
    calib = torch.randn(16, 3, 224, 224)

    prefix = torch.nn.Sequential(model.features[0], model.features[1], model.features[2])
    block = model.features[3]
    assert block.use_res_connect

    with torch.no_grad():
        block_in = prefix(calib)             # calibration input to block 3
        block_out = block(block_in)          # calibration output of block 3 (post residual)

        c0 = block.block[0]  # expand: Conv2dNormActivation (conv,bn,relu)
        c1 = block.block[1]  # depthwise: Conv2dNormActivation (conv,bn,relu)
        c2 = block.block[2]  # project: Conv2dNormActivation (conv,bn)  -- no activation

        mid0 = c0(block_in)                  # after expand conv+bn+relu
        mid1 = c1(mid0)                      # after depthwise conv+bn+relu
        mid2 = c2(mid1)                      # after project conv+bn (pre-residual, linear)

    w0_f, b0_f = fold_bn(c0[0], c0[1])
    w1_f, b1_f = fold_bn(c1[0], c1[1])
    w2_f, b2_f = fold_bn(c2[0], c2[1])

    # Activation quantization params, calibrated over the batch.
    s_in, zp_in = asym_qparams(block_in.numpy())
    s_mid0, zp_mid0 = asym_qparams(mid0.numpy())
    s_mid1, zp_mid1 = asym_qparams(mid1.numpy())
    s_mid2, zp_mid2 = asym_qparams(mid2.numpy())
    s_out, zp_out = asym_qparams(block_out.numpy())

    # Weight quantization (per-output-channel symmetric).
    s_w0, w0_q = quantize_weight_per_channel(w0_f)
    s_w1, w1_q = quantize_weight_per_channel(w1_f)
    s_w2, w2_q = quantize_weight_per_channel(w2_f)

    # Bias quantization: int32, scale = s_in_layer * s_w[oc].
    b0_q = np.round(b0_f / (s_in * s_w0)).astype(np.int64)
    b1_q = np.round(b1_f / (s_mid0 * s_w1)).astype(np.int64)
    b2_q = np.round(b2_f / (s_mid1 * s_w2)).astype(np.int64)

    # Per-channel requantization multiplier/shift for each layer.
    def layer_mults(s_layer_in, s_w, s_layer_out):
        mults = [quantize_multiplier((s_layer_in * sw) / s_layer_out) for sw in s_w]
        return np.array([m[0] for m in mults], dtype=np.int64), \
            np.array([m[1] for m in mults], dtype=np.int32)

    m0_fixed, m0_shift = layer_mults(s_in, s_w0, s_mid0)
    m1_fixed, m1_shift = layer_mults(s_mid0, s_w1, s_mid1)
    m2_fixed, m2_shift = layer_mults(s_mid1, s_w2, s_mid2)

    # ---- Integer-only reference forward pass on ONE fixed test sample ----
    test_input = block_in[0].numpy()  # (24, 56, 56) float
    x_int = quantize_asym(test_input, s_in, zp_in)[:, 0] if test_input.ndim == 4 else None
    x_int = quantize_asym(test_input, s_in, zp_in)  # (24, H, W) int32

    acc0 = conv1x1_int(x_int, zp_in, w0_q) + b0_q[:, None, None]
    a0 = requantize(acc0, 0, 0, zp_mid0, relu=True)  # placeholder overwritten below per-channel

    # per-channel requantize (multiplier differs per output channel)
    def requantize_per_channel(acc, q_fixed, shift, zp, relu):
        out = np.zeros_like(acc)
        for oc in range(acc.shape[0]):
            out[oc] = requantize(acc[oc], int(q_fixed[oc]), int(shift[oc]), zp, relu)
        return out

    a0 = requantize_per_channel(acc0, m0_fixed, m0_shift, zp_mid0, relu=True)

    acc1 = depthwise_conv3x3_int(a0, zp_mid0, w1_q, pad=1) + b1_q[:, None, None]
    a1 = requantize_per_channel(acc1, m1_fixed, m1_shift, zp_mid1, relu=True)

    acc2 = conv1x1_int(a1, zp_mid1, w2_q) + b2_q[:, None, None]
    a2 = requantize_per_channel(acc2, m2_fixed, m2_shift, zp_mid2, relu=False)  # pre-residual

    # Residual add: dequantize both branches to a shared int32 fixed-point domain, add, requantize.
    # real = (a2 - zp_mid2) * s_mid2 + (x_int - zp_in) * s_in ; quantize to s_out/zp_out.
    add_mult_a, add_shift_a = quantize_multiplier(s_mid2 / s_out)
    add_mult_b, add_shift_b = quantize_multiplier(s_in / s_out)
    left = multiply_by_quantized_multiplier((a2 - zp_mid2).astype(np.int64), add_mult_a, add_shift_a)
    right = multiply_by_quantized_multiplier((x_int - zp_in).astype(np.int64), add_mult_b, add_shift_b)
    final_acc = left + right + zp_out
    final_int = np.clip(final_acc, QMIN, QMAX).astype(np.int32)

    # Sanity check against the float reference (expect small quantization error only).
    ref_float = block_out[0].numpy()
    dequant_final = (final_int - zp_out) * s_out
    max_abs_err = np.max(np.abs(dequant_final - ref_float))
    rel_err = max_abs_err / (np.abs(ref_float).max() + 1e-8)
    print(f"Block3 INT8 reference vs float: max_abs_err={max_abs_err:.4f}  rel_err={rel_err:.4%}")
    print(f"Shapes: input={x_int.shape} output={final_int.shape}")

    os.makedirs(os.path.expanduser("~/riscv-tools/artifacts"), exist_ok=True)
    np.savez(
        os.path.expanduser("~/riscv-tools/artifacts/block3_export.npz"),
        x_int=x_int, zp_in=zp_in, s_in=s_in,
        w0_q=w0_q, s_w0=s_w0, b0_q=b0_q, m0_fixed=m0_fixed, m0_shift=m0_shift,
        w1_q=w1_q, s_w1=s_w1, b1_q=b1_q, m1_fixed=m1_fixed, m1_shift=m1_shift,
        w2_q=w2_q, s_w2=s_w2, b2_q=b2_q, m2_fixed=m2_fixed, m2_shift=m2_shift,
        s_mid0=s_mid0, zp_mid0=zp_mid0, s_mid1=s_mid1, zp_mid1=zp_mid1,
        s_mid2=s_mid2, zp_mid2=zp_mid2, s_out=s_out, zp_out=zp_out,
        add_mult_a=add_mult_a, add_shift_a=add_shift_a,
        add_mult_b=add_mult_b, add_shift_b=add_shift_b,
        final_int=final_int,
    )
    print("Saved ~/riscv-tools/artifacts/block3_export.npz")


if __name__ == "__main__":
    main()
