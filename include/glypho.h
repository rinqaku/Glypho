#ifndef GLYPHO_H
#define GLYPHO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GlyphoBuffer {
    uint8_t *data;
    size_t len;
    size_t capacity;
} GlyphoBuffer;

typedef struct GlyphoResult {
    int32_t status;
    GlyphoBuffer body;
} GlyphoResult;

/*
 * Returns a UTF-8 JSON document when status is 0. On failure, body contains
 * a UTF-8 error message. The caller must release body with glypho_buffer_free.
 * Path and options are borrowed only for the duration of this call.
 */
GlyphoResult glypho_recognize_json(
    const uint8_t *path_data,
    size_t path_len,
    const uint8_t *options_data,
    size_t options_len
);

/*
 * Loads and retains the ONNX detector and recognizers selected by options.
 * Returns a UTF-8 JSON status object. The process-wide cache is shared by
 * subsequent recognition calls with the same models, quality, and threads.
 */
GlyphoResult glypho_warmup_json(
    const uint8_t *options_data,
    size_t options_len
);

/* Returns runtime, model, languages, cache and resolved-device information. */
GlyphoResult glypho_info_json(
    const uint8_t *options_data,
    size_t options_len
);

void glypho_buffer_free(GlyphoBuffer buffer);

/* Returns a static, null-terminated UTF-8 string. */
const uint8_t *glypho_version(void);

#ifdef __cplusplus
}
#endif

#endif
