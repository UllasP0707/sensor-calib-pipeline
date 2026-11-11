#include "sim_api.h"
#include "sim_core.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
struct Handle { sim::Config config; std::unique_ptr<sim::Backend> backend; };
Handle& get(void* pointer) {
    if (!pointer) throw std::invalid_argument("Null simulation handle");
    return *static_cast<Handle*>(pointer);
}
template<class Function> int guarded(Function operation) {
    try { operation(); error.clear(); return 0; }
    catch (const std::exception& e) { error = e.what(); return -1; }
    catch (...) { error = "Unknown simulation failure"; return -1; }
}
}
extern "C" {
const char* sim_last_error() { return error.c_str(); }
int sim_abi_version() { return 1; }
int sim_cuda_available() {
#ifdef SIM_ENABLE_CUDA
    return sim::cuda_available() ? 1 : 0;
#else
    return 0;
#endif
}
int sim_create(int count, float dt, int horizon, unsigned workers, int backend, void** output) {
    if (output) *output = nullptr;
    return guarded([&] {
        if (!output || count < 1 || count > 1000000 || !std::isfinite(dt) || dt <= 0 || dt > 0.2F ||
            horizon < 1 || horizon > 100000 || workers < 1 || workers > 256)
            throw std::invalid_argument("Invalid configuration: count 1..1e6, dt (0,0.2], horizon 1..1e5, workers 1..256");
        auto h = std::make_unique<Handle>();
        h->config = {count, dt, horizon, workers};
        if (backend == 0) h->backend = sim::make_cpu(h->config);
        else if (backend == 1) {
#ifdef SIM_ENABLE_CUDA
            h->backend = sim::make_cuda(h->config);
#else
            throw std::runtime_error("CUDA backend was not compiled; rebuild with -DENABLE_CUDA=ON");
#endif
        } else throw std::invalid_argument("Unknown backend");
        h->backend->reset(0, true);
        *output = h.release();
    });
}
void sim_destroy(void* handle) { delete static_cast<Handle*>(handle); }
int sim_reset(void* handle, uint32_t seed, int all) {
    return guarded([&] { get(handle).backend->reset(seed, all != 0); });
}
int sim_step(void* handle, const float* actions) {
    return guarded([&] {
        auto& h = get(handle);
        if (!actions) throw std::invalid_argument("Null actions");
        for (int i = 0; i < h.config.count * 2; ++i)
            if (!std::isfinite(actions[i])) throw std::invalid_argument("Actions must be finite");
        h.backend->step(actions);
    });
}
int sim_read(void* handle, float* observations, float* rewards, uint8_t* terminated, uint8_t* truncated) {
    return guarded([&] {
        auto& h = get(handle);
        if (!observations || !rewards || !terminated || !truncated) throw std::invalid_argument("Null output");
        const auto b = h.backend->host();
        const auto n = static_cast<std::size_t>(h.config.count);
        std::memcpy(observations, b.observation, n * sim::obs_dim * sizeof(float));
        std::memcpy(rewards, b.reward, n * sizeof(float));
        std::memcpy(terminated, b.terminated, n);
        std::memcpy(truncated, b.truncated, n);
    });
}
int sim_read_state(void* handle, float* output) {
    return guarded([&] {
        auto& h = get(handle);
        if (!output) throw std::invalid_argument("Null state output");
        const auto b = h.backend->host();
        for (int i = 0; i < h.config.count; ++i)
            for (int j = 0; j < sim::state_dim; ++j) output[i * sim::state_dim + j] = b.state[j * h.config.count + i];
    });
}
int sim_timing(void* handle, float* output) {
    return guarded([&] {
        if (!output) throw std::invalid_argument("Null timing output");
        const auto t = get(handle).backend->timing();
        output[0] = t.upload_ms; output[1] = t.kernel_ms; output[2] = t.download_ms;
    });
}
}
