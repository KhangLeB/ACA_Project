/* Minimal bare-metal startup for RV64IMC targets running under Spike (HTIF tohost/fromhost). */
.section .text.init
.global _start
_start:
    la sp, _stack_top
    call main
    /* encode return code per HTIF exit protocol: (code << 1) | 1 */
    slli a0, a0, 1
    ori  a0, a0, 1
    la   t0, tohost
    sd   a0, 0(t0)
1:
    j 1b

.section .tohost,"aw",@progbits
.align 6
.global tohost
tohost: .dword 0

.section .fromhost,"aw",@progbits
.align 6
.global fromhost
fromhost: .dword 0
