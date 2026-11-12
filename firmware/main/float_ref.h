#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Float32 reference of TinyCNN (BN folded). weights: packed blob from quantize.py. */
int float_ref_alloc(void);
int float_ref_init(const float *weights_blob);
/* in: normalized float NHWC 32x32x3. returns argmax, writes 10 logits if logits != NULL */
int float_ref_run(const float *in, float *logits);
void *float_ref_buf_a(void);
#ifdef __cplusplus
}
#endif
