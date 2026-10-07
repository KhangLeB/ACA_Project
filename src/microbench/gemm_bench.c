/* Phase 3 required "one microbenchmark": standalone INT8xINT8->INT32 GEMM, independent of the
 * model, naive triple-nested loop (no tiling/blocking) — represents the B0 scalar baseline. */
#include <stdint.h>

#define GEMM_M 32
#define GEMM_K 32
#define GEMM_N 32

static int8_t A[GEMM_M * GEMM_K];
static int8_t B[GEMM_K * GEMM_N];
/* volatile: prevents -O2 from proving the result is unused and eliminating the loops */
static volatile int32_t C[GEMM_M * GEMM_N];

int main(void) {
    for (int i = 0; i < GEMM_M * GEMM_K; i++) A[i] = (int8_t)(i * 3 + 1);
    for (int i = 0; i < GEMM_K * GEMM_N; i++) B[i] = (int8_t)(i * 5 + 2);

    for (int i = 0; i < GEMM_M; i++) {
        for (int j = 0; j < GEMM_N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < GEMM_K; k++) {
                sum += (int32_t)A[i * GEMM_K + k] * (int32_t)B[k * GEMM_N + j];
            }
            C[i * GEMM_N + j] = sum;
        }
    }
    return 0;
}
