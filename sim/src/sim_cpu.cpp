#include "sim_core.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace sim {
class Cpu final : public Backend {
    Config c_;
    std::vector<float> state_, observation_, reward_;
    std::vector<std::uint32_t> episode_;
    std::vector<int> steps_;
    std::vector<std::uint8_t> terminated_, truncated_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_, complete_;
    std::uint64_t generation_ = 0;
    unsigned remaining_ = 0;
    bool stop_ = false, resetting_ = true, all_ = true;
    std::uint32_t seed_ = 0;
    const float* actions_ = nullptr;
    Timing timing_;
    void chunk(unsigned id) {
        const auto begin = static_cast<int>(static_cast<std::uint64_t>(c_.count) * id / c_.workers);
        const auto end = static_cast<int>(static_cast<std::uint64_t>(c_.count) * (id + 1) / c_.workers);
        const auto b = host();
        for (int i = begin; i < end; ++i) {
            if (resetting_) reset_one(b, c_, i, seed_, all_);
            else step_one(b, c_, i, actions_);
        }
    }
    void worker(unsigned id) {
        std::uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stop_ || seen != generation_; });
            if (stop_) return;
            seen = generation_;
            lock.unlock();
            chunk(id);
            lock.lock();
            if (--remaining_ == 0) complete_.notify_one();
        }
    }
    void dispatch() {
        if (c_.workers == 1) { chunk(0); return; }
        std::unique_lock<std::mutex> lock(mutex_);
        remaining_ = c_.workers;
        ++generation_;
        ready_.notify_all();
        complete_.wait(lock, [&] { return remaining_ == 0; });
    }
    void stop() noexcept {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        ready_.notify_all();
        for (auto& t : workers_) t.join();
    }
public:
    explicit Cpu(Config c) : c_(c), state_(c.count * state_dim), observation_(c.count * obs_dim),
        reward_(c.count), episode_(c.count), steps_(c.count), terminated_(c.count), truncated_(c.count) {
        c_.workers = std::min<unsigned>(c_.workers, static_cast<unsigned>(c_.count));
        try {
            if (c_.workers > 1)
                for (unsigned i = 0; i < c_.workers; ++i) workers_.emplace_back([this, i] { worker(i); });
        } catch (...) { stop(); throw; }
    }
    ~Cpu() override { stop(); }
    Buffers host() override { return {state_.data(), observation_.data(), reward_.data(),
        episode_.data(), steps_.data(), terminated_.data(), truncated_.data()}; }
    void reset(std::uint32_t seed, bool all) override {
        resetting_ = true; seed_ = seed; all_ = all; dispatch();
    }
    void step(const float* actions) override {
        const auto start = std::chrono::steady_clock::now();
        resetting_ = false; actions_ = actions; dispatch();
        timing_.kernel_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    Timing timing() const override { return timing_; }
};
std::unique_ptr<Backend> make_cpu(Config config) { return std::make_unique<Cpu>(config); }
}  // namespace sim
