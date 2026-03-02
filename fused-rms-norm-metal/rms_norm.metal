#include <metal_stdlib>
#include "utils.metal"

using namespace metal;

// Maximum number of simdgroups per threadgroup for reduction.
// 512 threads / 32 threads per simdgroup = 16 simdgroups max.
constant constexpr int MAX_SIMDGROUPS = 16;

// Threadgroup-wide sum reduction using simdgroups.
// Each thread contributes a value; returns the total sum to all threads.
static inline float threadgroup_reduce_sum(
    float value,
    threadgroup float *shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

  // Phase 1: reduce within each simdgroup.
  float simd_val = simd_sum(value);

  // Phase 2: first thread of each simdgroup writes to shared memory.
  uint simdgroup_id = tid / 32;
  uint lane_id = tid % 32;
  if (lane_id == 0) {
    shared[simdgroup_id] = simd_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 3: first simdgroup reduces across simdgroup partial sums.
  uint num_simdgroups = (tg_size + 31) / 32;
  float result = 0.0f;
  if (tid < num_simdgroups) {
    result = shared[tid];
  }
  result = simd_sum(result);

  // Broadcast result to all threads via shared memory.
  if (tid == 0) {
    shared[0] = result;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return shared[0];
}

// RMS normalization kernel.
// out[token, i] = (input[token, i] / RMS(input[token, :])) * weight[i]
// where RMS = sqrt(mean(x^2) + epsilon)
//
// One threadgroup per token. Threads stride across hidden_size.
template <typename scalar_t>
kernel void rms_norm_kernel(
    device scalar_t *out [[buffer(0)]],
    const device scalar_t *input [[buffer(1)]],
    const device scalar_t *weight [[buffer(2)]],
    const device float &epsilon [[buffer(3)]],
    const device int &num_tokens [[buffer(4)]],
    const device int &hidden_size [[buffer(5)]],
    const device int64_t &input_stride [[buffer(6)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

  // Phase 1: accumulate sum of squares for variance.
  float variance = 0.0f;
  for (int i = tid; i < hidden_size; i += tg_size) {
    float x = static_cast<float>(input[token_idx * input_stride + i]);
    variance += x * x;
  }

  // Phase 2: reduce variance across threadgroup.
  variance = threadgroup_reduce_sum(variance, shared, tid, tg_size);

  // Phase 3: compute scaling factor.
  float s_variance = rsqrt(variance / static_cast<float>(hidden_size) + epsilon);

  // Phase 4: normalize and scale.
  for (int i = tid; i < hidden_size; i += tg_size) {
    float x = static_cast<float>(input[token_idx * input_stride + i]);
    float w = static_cast<float>(weight[i]);
    out[token_idx * hidden_size + i] = static_cast<scalar_t>(x * s_variance * w);
  }
}

// Fused residual addition + RMS normalization kernel.
//
// After execution:
//   residual[token, i] = old_residual[token, i] + old_input[token, i]
//   input[token, i]    = rms_norm(new_residual[token, :]) * weight[i]
//
// This fuses two memory passes into one: the residual addition and variance
// accumulation happen in the same loop, saving memory bandwidth.
template <typename scalar_t>
kernel void fused_add_rms_norm_kernel(
    device scalar_t *input [[buffer(0)]],
    device scalar_t *residual [[buffer(1)]],
    const device scalar_t *weight [[buffer(2)]],
    const device float &epsilon [[buffer(3)]],
    const device int &num_tokens [[buffer(4)]],
    const device int &hidden_size [[buffer(5)]],
    const device int64_t &input_stride [[buffer(6)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

  // Phase 1: add residual and accumulate variance in one pass.
  float variance = 0.0f;
  for (int i = tid; i < hidden_size; i += tg_size) {
    float inp = static_cast<float>(input[token_idx * input_stride + i]);
    float res = static_cast<float>(residual[token_idx * hidden_size + i]);
    float z = inp + res;
    variance += z * z;
    residual[token_idx * hidden_size + i] = static_cast<scalar_t>(z);
  }

  // Phase 2: reduce variance across threadgroup.
  variance = threadgroup_reduce_sum(variance, shared, tid, tg_size);

  // Phase 3: compute scaling factor.
  float s_variance = rsqrt(variance / static_cast<float>(hidden_size) + epsilon);

  // Phase 4: read updated residual, normalize, and write to input.
  for (int i = tid; i < hidden_size; i += tg_size) {
    float x = static_cast<float>(residual[token_idx * hidden_size + i]);
    float w = static_cast<float>(weight[i]);
    input[token_idx * input_stride + i] = static_cast<scalar_t>(x * s_variance * w);
  }
}

// Instantiate kernel variants.
#define instantiate_rms_norm(type)                                              \
  template [[host_name("rms_norm_" #type)]] [[kernel]] void                    \
  rms_norm_kernel<type>(                                                       \
      device type *out [[buffer(0)]],                                          \
      const device type *input [[buffer(1)]],                                  \
      const device type *weight [[buffer(2)]],                                 \
      const device float &epsilon [[buffer(3)]],                               \
      const device int &num_tokens [[buffer(4)]],                              \
      const device int &hidden_size [[buffer(5)]],                             \
      const device int64_t &input_stride [[buffer(6)]],                        \
      threadgroup float *shared [[threadgroup(0)]],                            \
      uint token_idx [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint tg_size [[threads_per_threadgroup]]);

#define instantiate_fused_add_rms_norm(type)                                   \
  template [[host_name("fused_add_rms_norm_" #type)]] [[kernel]] void          \
  fused_add_rms_norm_kernel<type>(                                             \
      device type *input [[buffer(0)]],                                        \
      device type *residual [[buffer(1)]],                                     \
      const device type *weight [[buffer(2)]],                                 \
      const device float &epsilon [[buffer(3)]],                               \
      const device int &num_tokens [[buffer(4)]],                              \
      const device int &hidden_size [[buffer(5)]],                             \
      const device int64_t &input_stride [[buffer(6)]],                        \
      threadgroup float *shared [[threadgroup(0)]],                            \
      uint token_idx [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint tg_size [[threads_per_threadgroup]]);

instantiate_rms_norm(float);
instantiate_rms_norm(half);
instantiate_rms_norm(bfloat16_t);

instantiate_fused_add_rms_norm(float);
instantiate_fused_add_rms_norm(half);
instantiate_fused_add_rms_norm(bfloat16_t);
