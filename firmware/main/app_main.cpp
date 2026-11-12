// Float32 (plain C) vs int8 (ESP-DL + ESP-PPQ) TinyCNN benchmark on ESP32-P4.
#include <cmath>
#include <cstdio>
#include <cstring>
#include "dl_model_base.hpp"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_clk_tree.h"
#include "esp_memory_utils.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "float_ref.h"

extern const uint8_t model_start[] asm("_binary_tinycnn_int8_espdl_start");
extern const uint8_t model_end[] asm("_binary_tinycnn_int8_espdl_end");
extern const uint8_t fw_start[] asm("_binary_float_weights_bin_start");
extern const uint8_t fw_end[] asm("_binary_float_weights_bin_end");
extern const uint8_t ts_start[] asm("_binary_testset_bin_start");

static const float MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
static const float STD[3] = {0.2470f, 0.2435f, 0.2616f};
static const int LAT_RUNS = 200;

static inline void normalize(const uint8_t *img, float *out)
{
    for (int i = 0; i < 32 * 32; i++)
        for (int c = 0; c < 3; c++) out[i * 3 + c] = (img[i * 3 + c] / 255.f - MEAN[c]) / STD[c];
}

static int argmax_tensor(dl::TensorBase *t)
{
    int best = 0;
    if (t->dtype == dl::DATA_TYPE_INT8) {
        int8_t *d = (int8_t *)t->data;
        for (int k = 1; k < 10; k++) if (d[k] > d[best]) best = k;
    } else if (t->dtype == dl::DATA_TYPE_INT16) {
        int16_t *d = (int16_t *)t->data;
        for (int k = 1; k < 10; k++) if (d[k] > d[best]) best = k;
    } else {
        float *d = (float *)t->data;
        for (int k = 1; k < 10; k++) if (d[k] > d[best]) best = k;
    }
    return best;
}

extern "C" void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(1500));
    uint32_t cpu_hz = 0;
    esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU, ESP_CLK_TREE_SRC_FREQ_PRECISION_EXACT, &cpu_hz);
    const uint32_t n = *(const uint32_t *)ts_start;
    const uint8_t *labels = ts_start + 4;
    const uint8_t *imgs = ts_start + 4 + n;
    printf("BENCH_START cpu_mhz=%lu n_test=%lu\n", (unsigned long)(cpu_hz / 1000000), (unsigned long)n);

    // ---------------- float32 reference ----------------
    // Activation buffers first (internal SRAM), then weights: internal if room, else PSRAM.
    if (float_ref_alloc() < 0) { printf("ERROR float buffers alloc failed\n"); return; }
    float *in_f = (float *)heap_caps_malloc(32 * 32 * 3 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    size_t fw_bytes = fw_end - fw_start;
    float *fw = (float *)heap_caps_malloc(fw_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!fw) fw = (float *)heap_caps_malloc(fw_bytes, MALLOC_CAP_SPIRAM);
    memcpy(fw, fw_start, fw_bytes);
    float_ref_init(fw);
    printf("FLOAT weights_bytes=%u weights_internal=%d act_buf_internal=%d\n", (unsigned)fw_bytes,
           (int)esp_ptr_internal(fw), (int)esp_ptr_internal(float_ref_buf_a()));

    int correct_f = 0;
    for (uint32_t i = 0; i < n; i++) {
        normalize(imgs + i * 3072, in_f);
        correct_f += float_ref_run(in_f, nullptr) == labels[i];
        if (i % 500 == 499) { printf("float progress %lu\n", (unsigned long)i + 1); vTaskDelay(1); }
    }
    normalize(imgs, in_f);
    float_ref_run(in_f, nullptr);  // warm-up
    int64_t t0 = esp_timer_get_time();
    for (int r = 0; r < LAT_RUNS; r++) float_ref_run(in_f, nullptr);
    double float_us = (double)(esp_timer_get_time() - t0) / LAT_RUNS;
    printf("RESULT float correct=%d n=%lu latency_us=%.1f\n", correct_f, (unsigned long)n, float_us);

    // ---------------- int8 ESP-DL ----------------
    size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    dl::Model *model = new dl::Model((const char *)model_start, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    size_t heap_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    dl::TensorBase *inp = model->get_input();
    dl::TensorBase *out = model->get_output();
    int in_exp = inp->exponent;
    printf("INT8 espdl_bytes=%u runtime_heap_bytes=%u in_shape=[%d,%d,%d,%d] in_exp=%d in_dtype=%d out_dtype=%d\n",
           (unsigned)(model_end - model_start), (unsigned)(heap_before - heap_after), inp->shape[0], inp->shape[1],
           inp->shape[2], inp->shape[3], in_exp, (int)inp->dtype, (int)out->dtype);
    const float inv_scale = 1.f / ldexpf(1.f, in_exp);
    int8_t *in_q = (int8_t *)inp->data;
    auto quantize_into = [&](const uint8_t *img) {
        for (int i = 0; i < 32 * 32; i++)
            for (int c = 0; c < 3; c++) {
                float v = roundf(((img[i * 3 + c] / 255.f - MEAN[c]) / STD[c]) * inv_scale);
                in_q[i * 3 + c] = (int8_t)(v > 127.f ? 127.f : (v < -128.f ? -128.f : v));
            }
    };

    int correct_q = 0;
    for (uint32_t i = 0; i < n; i++) {
        quantize_into(imgs + i * 3072);
        model->run();
        correct_q += argmax_tensor(out) == labels[i];
        if (i % 500 == 499) { printf("int8 progress %lu\n", (unsigned long)i + 1); vTaskDelay(1); }
    }
    quantize_into(imgs);
    model->run();  // warm-up
    t0 = esp_timer_get_time();
    for (int r = 0; r < LAT_RUNS; r++) model->run();
    double int8_us = (double)(esp_timer_get_time() - t0) / LAT_RUNS;
    // include input quantization cost too, for an end-to-end number
    t0 = esp_timer_get_time();
    for (int r = 0; r < LAT_RUNS; r++) { quantize_into(imgs); model->run(); }
    double int8_e2e_us = (double)(esp_timer_get_time() - t0) / LAT_RUNS;
    printf("RESULT int8 correct=%d n=%lu latency_us=%.1f e2e_us=%.1f\n", correct_q, (unsigned long)n, int8_us,
           int8_e2e_us);
    printf("BENCH_DONE\n");
    model->profile_module(true);
    while (true) vTaskDelay(pdMS_TO_TICKS(10000));
}
