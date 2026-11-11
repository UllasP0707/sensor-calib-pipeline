#pragma once
#include <cmath>
#include <cstdint>
#include <memory>

#ifdef __CUDACC__
#define SIM_HD __host__ __device__
#else
#define SIM_HD
#endif

namespace sim {
constexpr int state_dim = 6;  // x, y, heading, speed, lead_x, lead_speed
constexpr int obs_dim = 6;
struct Config { int count; float dt; int horizon; unsigned workers; };
struct Buffers {
    float* state;          // SoA: state[field * count + environment]
    float* observation;    // AoS: Python-compatible [count, obs_dim]
    float* reward;
    std::uint32_t* episode;
    int* steps;
    std::uint8_t* terminated;
    std::uint8_t* truncated;
};
struct Timing { float upload_ms = 0; float kernel_ms = 0; float download_ms = 0; };
SIM_HD inline float clamp(float x, float lo, float hi) { return fminf(hi, fmaxf(lo, x)); }
SIM_HD inline std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
SIM_HD inline float random_unit(std::uint32_t seed, int i, std::uint32_t episode, unsigned field) {
    const auto bits = mix(seed ^ mix(static_cast<std::uint32_t>(i)) ^
                          mix(episode + 0x9e3779b9U) ^ mix(field + 17U));
    return static_cast<float>(bits >> 8) * (1.0F / 16777216.0F);
}
SIM_HD inline void observe(Buffers b, Config c, int i) {
    const int n = c.count;
    float* o = b.observation + obs_dim * i;
    o[0] = b.state[n + i] / 2.25F;
    o[1] = b.state[2 * n + i];
    o[2] = b.state[3 * n + i] / 20.0F;
    o[3] = (b.state[4 * n + i] - b.state[i]) / 100.0F;
    o[4] = b.state[5 * n + i] / 20.0F;
    o[5] = b.state[i] / 300.0F;  // progress makes goal-dependent rewards observable
}
SIM_HD inline void reset_one(Buffers b, Config c, int i, std::uint32_t seed, bool all) {
    if (!all && !(b.terminated[i] || b.truncated[i])) return;
    const auto ep = all ? 0U : b.episode[i] + 1U;
    b.episode[i] = ep;
    b.state[i] = 0;
    b.state[c.count + i] = (random_unit(seed, i, ep, 0) - 0.5F) * 1.5F;
    b.state[2 * c.count + i] = (random_unit(seed, i, ep, 1) - 0.5F) * 0.08F;
    b.state[3 * c.count + i] = 10.0F + 8.0F * random_unit(seed, i, ep, 2);
    b.state[4 * c.count + i] = 35.0F + 50.0F * random_unit(seed, i, ep, 3);
    b.state[5 * c.count + i] = 12.0F + 10.0F * random_unit(seed, i, ep, 4);
    b.steps[i] = 0; b.reward[i] = 0; b.terminated[i] = 0; b.truncated[i] = 0;
    observe(b, c, i);
}
SIM_HD inline void step_one(Buffers b, Config c, int i, const float* actions) {
    b.reward[i] = 0;
    if (b.terminated[i] || b.truncated[i]) return;  // explicit reset; terminal observation retained
    const int n = c.count;
    float x = b.state[i], y = b.state[n + i], yaw = b.state[2 * n + i];
    const float old_gap = b.state[4 * n + i] - x;
    const float acceleration = clamp(actions[2 * i], -1, 1) * 3.0F;
    const float steer = clamp(actions[2 * i + 1], -1, 1) * 0.35F;
    const float speed = clamp(b.state[3 * n + i] + acceleration * c.dt, 0, 35);
    // Semi-implicit Euler kinematic bicycle, wheelbase 5m, fixed 0.1s default step.
    x += speed * cosf(yaw) * c.dt;
    y += speed * sinf(yaw) * c.dt;
    yaw += speed / 5.0F * tanf(steer) * c.dt;
    yaw = atan2f(sinf(yaw), cosf(yaw));
    const float lead_x = b.state[4 * n + i] + b.state[5 * n + i] * c.dt;
    const float gap = lead_x - x;
    // Swept longitudinal interval catches passing through the lead car in one step.
    const bool collision = fminf(old_gap, gap) <= 7.0F && fmaxf(old_gap, gap) >= -7.0F &&
                           fminf(fabsf(b.state[n + i]), fabsf(y)) < 2.5F;
    const bool offroad = fabsf(y) > 2.25F;  // 3.5m road half-width minus 1.25m ego half-width
    b.steps[i] += 1;
    b.terminated[i] = static_cast<std::uint8_t>(collision || offroad || x >= 300.0F);
    b.truncated[i] = static_cast<std::uint8_t>(!b.terminated[i] && b.steps[i] >= c.horizon);
    const float speed_error = (speed - 20.0F) / 20.0F;
    b.reward[i] = c.dt * (speed * cosf(yaw) / 20.0F - 0.5F * y * y -
                         speed_error * speed_error - 0.05F * steer * steer);
    if (collision || offroad) b.reward[i] -= 10;
    else if (x >= 300.0F) b.reward[i] += 10;
    b.state[i] = x; b.state[n + i] = y; b.state[2 * n + i] = yaw;
    b.state[3 * n + i] = speed; b.state[4 * n + i] = lead_x;
    observe(b, c, i);
}
class Backend {
public:
    virtual ~Backend() = default;
    virtual void reset(std::uint32_t seed, bool all) = 0;
    virtual void step(const float* actions) = 0;
    virtual Buffers host() = 0;
    virtual Timing timing() const = 0;
};
std::unique_ptr<Backend> make_cpu(Config config);
#ifdef SIM_ENABLE_CUDA
std::unique_ptr<Backend> make_cuda(Config config);
bool cuda_available();
#endif
}  // namespace sim
