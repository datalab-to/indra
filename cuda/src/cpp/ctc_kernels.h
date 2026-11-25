// The recognizers' CTC outputs from logits left on the device ([B, T, C],
// the engine's "logits" output): what indra_ocr/ctc.py computes on the host,
// for the frames and lines that need it, so the logits never leave the GPU.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace indra_cuda {

// conf[b, t]: the softmax probability of frame t's best class (ctc.peak_prob),
// for t < lengths[b]; 0 after.
void frame_conf(const float* logits, const int64_t* lengths, int B, int T, int C, float* conf,
                cudaStream_t stream);

// For line rows[r] of the batch, frames t < n[r]: the blank's log-probability
// and the k (<= 8) best classes' log-probabilities and ids, best first, equal
// values with the lower class first (ctc.lm_inputs, ONNX Runtime's TopK).
// Frames from n[r] on hold blank 0, topv -inf, topi 0.
void lm_inputs(const float* logits, const int32_t* rows, const int32_t* n, int R, int T, int C, int k,
               float* blank, float* topv, int64_t* topi, cudaStream_t stream);

// dst[r] = src[rows[r]] for r < R, rows of row_bytes bytes (a multiple of 4).
void gather_rows(const void* src, void* dst, const int32_t* rows, int R, size_t row_bytes, cudaStream_t stream);

}  // namespace indra_cuda
