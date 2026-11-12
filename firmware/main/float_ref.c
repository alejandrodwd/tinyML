#include "float_ref.h"
#include <string.h>
#include "esp_heap_caps.h"

/* conv layers: cin, cout, spatial size, maxpool-after */
static const int L_CIN[5] = {3, 16, 32, 32, 64};
static const int L_COUT[5] = {16, 32, 32, 64, 64};
static const int L_POOL[5] = {0, 1, 0, 1, 1};

static const float *W[5], *B[5], *FCW, *FCB;
static float *buf_a, *buf_b;

int float_ref_alloc(void)
{
    const size_t n = 32 * 32 * 32 * sizeof(float);
    buf_a = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!buf_a) buf_a = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
    buf_b = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!buf_b) buf_b = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
    return (buf_a && buf_b) ? 0 : -1;
}

int float_ref_init(const float *blob)
{
    const float *p = blob;
    for (int l = 0; l < 5; l++) {
        W[l] = p; p += L_COUT[l] * 9 * L_CIN[l];
        B[l] = p; p += L_COUT[l];
    }
    FCW = p; p += 10 * 64;
    FCB = p; p += 10;
    return (int)(p - blob);
}

/* 3x3 conv, pad 1, stride 1, NHWC, fused ReLU */
/*
 * Optimized scalar kernel (the P4 has no float SIMD, so this is about memory and loop overhead):
 *  - weight-stationary: 8 output channels at a time, their weights (<= 18 KB) copied (interleaved) to internal SRAM,
 *    then swept over every output pixel, so weights stream from PSRAM once per layer instead of once per pixel;
 *  - in NHWC, the valid kx-span of one kernel row is a single contiguous run of input AND weights,
 *    so each pixel is at most 3 branch-free runs of length 3*cin; one input load feeds 8 independent fmadd chains.
 */
#define CO_BLK 8
static float wblk[9 * 64 * CO_BLK];  /* interleaved: wblk[k*CO_BLK + j] = w[co+j][k] */

static void conv3x3_relu(const float *in, float *out, int hw, int cin, int cout, const float *w, const float *b)
{
    const int kc = 9 * cin, rowlen = 3 * cin;
    for (int co = 0; co < cout; co += CO_BLK) {
        for (int k = 0; k < kc; k++)
            for (int j = 0; j < CO_BLK; j++) wblk[k * CO_BLK + j] = w[(co + j) * kc + k];
        const float *bb = b + co;
        for (int y = 0; y < hw; y++) {
            const int ky0 = y == 0 ? 1 : 0, ky1 = y == hw - 1 ? 1 : 2;
            for (int x = 0; x < hw; x++) {
                const int kx0 = x == 0 ? 1 : 0, kx1 = x == hw - 1 ? 1 : 2;
                const int len = (kx1 - kx0 + 1) * cin;
                float a0 = bb[0], a1 = bb[1], a2 = bb[2], a3 = bb[3], a4 = bb[4], a5 = bb[5], a6 = bb[6], a7 = bb[7];
                for (int ky = ky0; ky <= ky1; ky++) {
                    const float *ip = in + ((y + ky - 1) * hw + (x + kx0 - 1)) * cin;
                    const float *wp = wblk + (ky * rowlen + kx0 * cin) * CO_BLK;
                    const float *end = ip + len;
                    while (ip < end) {
                        const float v = *ip++;
                        a0 += v * wp[0]; a1 += v * wp[1]; a2 += v * wp[2]; a3 += v * wp[3];
                        a4 += v * wp[4]; a5 += v * wp[5]; a6 += v * wp[6]; a7 += v * wp[7];
                        wp += CO_BLK;
                    }
                }
                float *o = out + (y * hw + x) * cout + co;
                o[0] = a0 > 0.f ? a0 : 0.f; o[1] = a1 > 0.f ? a1 : 0.f; o[2] = a2 > 0.f ? a2 : 0.f; o[3] = a3 > 0.f ? a3 : 0.f;
                o[4] = a4 > 0.f ? a4 : 0.f; o[5] = a5 > 0.f ? a5 : 0.f; o[6] = a6 > 0.f ? a6 : 0.f; o[7] = a7 > 0.f ? a7 : 0.f;
            }
        }
    }
}

static void maxpool2(const float *in, float *out, int hw, int c)
{
    int oh = hw / 2;
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < oh; x++)
            for (int ch = 0; ch < c; ch++) {
                const float *p = in + ((2 * y) * hw + 2 * x) * c + ch;
                float m = p[0];
                if (p[c] > m) m = p[c];
                if (p[hw * c] > m) m = p[hw * c];
                if (p[hw * c + c] > m) m = p[hw * c + c];
                out[(y * oh + x) * c + ch] = m;
            }
}

int float_ref_run(const float *in, float *logits)
{
    const float *cur = in;
    float *a = buf_a, *b = buf_b;
    int hw = 32;
    for (int l = 0; l < 5; l++) {
        /* invariant: `a` is the free buffer, `cur` holds the current activation */
        conv3x3_relu(cur, a, hw, L_CIN[l], L_COUT[l], W[l], B[l]);
        cur = a; { float *t = a; a = b; b = t; }
        if (L_POOL[l]) {
            maxpool2(cur, a, hw, L_COUT[l]); hw /= 2;
            cur = a; { float *t = a; a = b; b = t; }
        }
    }
    /* cur: 4x4x64 -> GAP */
    float g[64] = {0};
    for (int i = 0; i < hw * hw; i++)
        for (int c = 0; c < 64; c++) g[c] += cur[i * 64 + c];
    for (int c = 0; c < 64; c++) g[c] *= 1.f / (hw * hw);
    int best = 0; float bv = -1e30f;
    for (int k = 0; k < 10; k++) {
        float acc = FCB[k];
        for (int c = 0; c < 64; c++) acc += FCW[k * 64 + c] * g[c];
        if (logits) logits[k] = acc;
        if (acc > bv) { bv = acc; best = k; }
    }
    return best;
}

void *float_ref_buf_a(void) { return buf_a; }
