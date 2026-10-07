// Spike custom extension "xqnn": the two quantized-NN instructions designed in Phase 4.
// Built as a ROCC-style extension on the custom-0 opcode (0x0b); load with `spike --extension=xqnn`.
// Semantics are identical to the C reference models in Project/src/kernels/xqmac.h (so Spike,
// the C golden reference, and the future CV-Wally RTL all agree by construction).
//
// Encoding (standard R-type on opcode 0x0b, funct3 = 0b111 so ROCC reads rs1,rs2 and writes rd):
//   funct7 = 0  ->  xqmac8    rd, rs1, rs2    : rd = (int32)rd + sum_{i=0..7} int8(rs1_i)*int8(rs2_i)
//   funct7 = 1  ->  xqrequant rd, rs1, rs2    : rd = requantize(acc=rs1, params=rs2)
// where for xqrequant rs2 packs: [31:0]=q_fixed, [39:32]=shift(int8), [47:40]=zero_point(int8),
// [48]=relu flag; result is the saturated INT8 output sign-extended into rd.

#include "rocc.h"
#include "trap.h"
#include <cstring>

static inline int64_t xqnn_dot8(reg_t a, reg_t b)
{
  int64_t s = 0;
  for (int i = 0; i < 8; i++) {
    int8_t ai = (int8_t)((a >> (8 * i)) & 0xff);
    int8_t bi = (int8_t)((b >> (8 * i)) & 0xff);
    s += (int64_t)ai * (int64_t)bi;
  }
  return s;
}

class xqnn_t : public rocc_t
{
 public:
  const char* name() const override { return "xqnn"; }

  reg_t custom0(processor_t* p, rocc_insn_t insn, reg_t xs1, reg_t xs2) override
  {
    switch (insn.funct) {
      case 0: { // xqmac8: 32-bit accumulator in rd, read-modify-write
        int32_t acc = (int32_t)p->get_state()->XPR[insn.rd];
        int32_t res = (int32_t)((int64_t)acc + xqnn_dot8(xs1, xs2));
        return (reg_t)(int64_t)res; // sign-extend the 32-bit accumulator
      }
      case 1: { // xqrequant: fixed-point multiply -> round -> shift -> +zp -> saturate
        int32_t acc     = (int32_t)xs1;
        int32_t q_fixed = (int32_t)(uint32_t)(xs2 & 0xffffffffu);
        int32_t shift   = (int32_t)(int8_t)((xs2 >> 32) & 0xff);
        int32_t zp      = (int32_t)(int8_t)((xs2 >> 40) & 0xff);
        int     relu    = (int)((xs2 >> 48) & 0x1);

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
        int64_t v  = r + zp;
        int64_t lo = relu ? zp : -128;
        int64_t hi = 127;
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        return (reg_t)(int64_t)(int8_t)v; // sign-extend saturated INT8 result
      }
      default:
        illegal_instruction(*p);
    }
    return 0;
  }
};

REGISTER_EXTENSION(xqnn, []() { static xqnn_t ext; return &ext; })
