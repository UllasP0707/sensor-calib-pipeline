#include "sim_core.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sim {
namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(result));
}
template<class T> class Array {
    T* pointer_ = nullptr;
    bool pinned_;
public:
    Array(std::size_t count, bool pinned) : pinned_(pinned) {
        if (pinned) check(cudaMallocHost(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
        else check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
    }
    ~Array() { if (pinned_) cudaFreeHost(pointer_); else cudaFree(pointer_); }
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    T* get() { return pointer_; }
};
struct Storage {
    Array<float> state, obs, reward;
    Array<std::uint32_t> episode;
    Array<int> steps;
    Array<std::uint8_t> terminated, truncated;
    Storage(int n, bool pinned) : state(n * state_dim, pinned), obs(n * obs_dim, pinned),
        reward(n, pinned), episode(n, pinned), steps(n, pinned), terminated(n, pinned), truncated(n, pinned) {}
    Buffers buffers() { return {state.get(), obs.get(), reward.get(), episode.get(), steps.get(), terminated.get(), truncated.get()}; }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};
__global__ void reset_kernel(Buffers b, Config c, std::uint32_t seed, bool all) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < c.count) reset_one(b, c, i, seed, all);
}
__global__ void step_kernel(Buffers b, Config c, const float* actions) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < c.count) step_one(b, c, i, actions);
}
class Cuda final : public Backend {
    Config c_;
    Stream stream_;
    Event begin_, uploaded_, computed_, downloaded_;
    Storage device_, host_;
    Array<float> device_actions_, host_actions_;
    Timing timing_;
    void download() {
        const auto d = device_.buffers(), h = host_.buffers();
        const auto n = static_cast<std::size_t>(c_.count);
        check(cudaMemcpyAsync(h.state, d.state, n * state_dim * sizeof(float), cudaMemcpyDeviceToHost, stream_.value));
        check(cudaMemcpyAsync(h.observation, d.observation, n * obs_dim * sizeof(float), cudaMemcpyDeviceToHost, stream_.value));
        check(cudaMemcpyAsync(h.reward, d.reward, n * sizeof(float), cudaMemcpyDeviceToHost, stream_.value));
        check(cudaMemcpyAsync(h.terminated, d.terminated, n, cudaMemcpyDeviceToHost, stream_.value));
        check(cudaMemcpyAsync(h.truncated, d.truncated, n, cudaMemcpyDeviceToHost, stream_.value));
    }
public:
    explicit Cuda(Config c) : c_(c), device_(c.count, false), host_(c.count, true),
        device_actions_(c.count * 2, false), host_actions_(c.count * 2, true) {}
    ~Cuda() override { cudaStreamSynchronize(stream_.value); }
    Buffers host() override { return host_.buffers(); }
    void reset(std::uint32_t seed, bool all) override {
        reset_kernel<<<(c_.count + 255) / 256, 256, 0, stream_.value>>>(device_.buffers(), c_, seed, all);
        check(cudaGetLastError());
        download();
        check(cudaStreamSynchronize(stream_.value));
    }
    void step(const float* actions) override {
        const auto bytes = static_cast<std::size_t>(c_.count) * 2 * sizeof(float);
        std::memcpy(host_actions_.get(), actions, bytes);
        check(cudaEventRecord(begin_.value, stream_.value));
        check(cudaMemcpyAsync(device_actions_.get(), host_actions_.get(), bytes, cudaMemcpyHostToDevice, stream_.value));
        check(cudaEventRecord(uploaded_.value, stream_.value));
        step_kernel<<<(c_.count + 255) / 256, 256, 0, stream_.value>>>(device_.buffers(), c_, device_actions_.get());
        check(cudaGetLastError());
        check(cudaEventRecord(computed_.value, stream_.value));
        download();
        check(cudaEventRecord(downloaded_.value, stream_.value));
        check(cudaEventSynchronize(downloaded_.value));
        check(cudaEventElapsedTime(&timing_.upload_ms, begin_.value, uploaded_.value));
        check(cudaEventElapsedTime(&timing_.kernel_ms, uploaded_.value, computed_.value));
        check(cudaEventElapsedTime(&timing_.download_ms, computed_.value, downloaded_.value));
    }
    Timing timing() const override { return timing_; }
};
}
bool cuda_available() {
    int devices = 0;
    return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}
std::unique_ptr<Backend> make_cuda(Config config) {
    if (!cuda_available()) throw std::runtime_error("No accessible CUDA device/driver; CPU fallback must be selected explicitly");
    return std::make_unique<Cuda>(config);
}
}  // namespace sim
