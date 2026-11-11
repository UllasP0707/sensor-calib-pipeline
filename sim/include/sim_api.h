#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Return 0 on success, -1 on failure. Error text is thread-local; no C++ exceptions cross ABI.
// Each handle is single-caller; independent handles may run concurrently.
const char* sim_last_error(void);
int sim_abi_version(void);
int sim_cuda_available(void);
int sim_create(int count, float dt, int horizon, unsigned workers, int backend, void** handle);
void sim_destroy(void* handle);
int sim_reset(void* handle, uint32_t seed, int all);
int sim_step(void* handle, const float* actions);
// Caller allocates n*6 floats, n floats, n bytes, n bytes respectively.
int sim_read(void* handle, float* observations, float* rewards, uint8_t* terminated, uint8_t* truncated);
int sim_read_state(void* handle, float* state_aos);
// Last step: upload, compute, download in milliseconds. CUDA events; CPU wall time.
int sim_timing(void* handle, float* three_values);
#ifdef __cplusplus
}
#endif
