#include "sim_core.hpp"
#include "sim_api.h"
#include <gtest/gtest.h>
#include <limits>
#include <vector>

TEST(Dynamics, StraightMotionAndTimeLimit) {
    auto backend = sim::make_cpu({1, 0.1F, 2, 1});
    backend->reset(42, true);
    auto b = backend->host();
    b.state[0] = 0; b.state[1] = 0; b.state[2] = 0; b.state[3] = 10;
    b.state[4] = 100; b.state[5] = 20;
    const float actions[]{1, 0};
    backend->step(actions);
    EXPECT_NEAR(b.state[0], 1.03F, 1e-6F);
    EXPECT_FLOAT_EQ(b.state[1], 0);
    EXPECT_NEAR(b.state[3], 10.3F, 1e-6F);
    EXPECT_FALSE(b.terminated[0]); EXPECT_FALSE(b.truncated[0]);
    backend->step(actions);
    EXPECT_TRUE(b.truncated[0]); EXPECT_FALSE(b.terminated[0]);
    const auto x = b.state[0];
    backend->step(actions);
    EXPECT_FLOAT_EQ(b.state[0], x); EXPECT_FLOAT_EQ(b.reward[0], 0);
}
TEST(Dynamics, CollisionAndOffroadTerminate) {
    auto backend = sim::make_cpu({2, 0.1F, 100, 1});
    backend->reset(3, true);
    auto b = backend->host();
    b.state[0] = 0; b.state[2] = 0; b.state[4] = 0;
    b.state[6] = 20; b.state[8] = 7.1F; b.state[10] = 0;
    b.state[3] = 2.4F;
    const float actions[]{0, 0, 0, 0};
    backend->step(actions);
    EXPECT_TRUE(b.terminated[0]); EXPECT_LT(b.reward[0], -9);
    EXPECT_TRUE(b.terminated[1]); EXPECT_FALSE(b.truncated[1]);
}
TEST(Dynamics, ThreadCountDoesNotChangeTrajectory) {
    auto serial = sim::make_cpu({257, 0.1F, 300, 1});
    auto parallel = sim::make_cpu({257, 0.1F, 300, 4});
    serial->reset(123, true); parallel->reset(123, true);
    std::vector<float> actions(257 * 2, 0.05F);
    for (int step = 0; step < 40; ++step) {
        serial->step(actions.data()); parallel->step(actions.data());
        for (int i = 0; i < 257 * 6; ++i)
            ASSERT_FLOAT_EQ(serial->host().state[i], parallel->host().state[i]);
    }
}
TEST(Api, InvalidInputsNeverCrossExceptionBoundary) {
    void* handle = nullptr;
    EXPECT_EQ(sim_create(-1, 0.1F, 10, 1, 0, &handle), -1);
    EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(sim_create(1, 0.1F, 10, 1, 0, &handle), 0);
    float actions[]{std::numeric_limits<float>::quiet_NaN(), 0};
    EXPECT_EQ(sim_step(handle, actions), -1);
    EXPECT_EQ(sim_step(nullptr, actions), -1);
    EXPECT_EQ(sim_read_state(handle, nullptr), -1);
    sim_destroy(handle);
}
TEST(Cuda, ParityWithCpuWhenDeviceIsAvailable) {
    if (!sim_cuda_available()) GTEST_SKIP() << "CUDA compiler/device unavailable";
    void* cpu = nullptr; void* gpu = nullptr;
    ASSERT_EQ(sim_create(257, 0.1F, 300, 1, 0, &cpu), 0);
    ASSERT_EQ(sim_create(257, 0.1F, 300, 1, 1, &gpu), 0) << sim_last_error();
    ASSERT_EQ(sim_reset(cpu, 42, 1), 0); ASSERT_EQ(sim_reset(gpu, 42, 1), 0);
    std::vector<float> actions(514, 0.01F), a(257 * 6), b(257 * 6);
    std::vector<float> ra(257), rb(257);
    std::vector<std::uint8_t> ta(257), tb(257), xa(257), xb(257);
    for (int step = 0; step < 50; ++step) {
        ASSERT_EQ(sim_step(cpu, actions.data()), 0); ASSERT_EQ(sim_step(gpu, actions.data()), 0);
        ASSERT_EQ(sim_read(cpu, a.data(), ra.data(), ta.data(), xa.data()), 0);
        ASSERT_EQ(sim_read(gpu, b.data(), rb.data(), tb.data(), xb.data()), 0);
        for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 2e-4F);
        for (int i = 0; i < 257; ++i) {
            EXPECT_NEAR(ra[i], rb[i], 2e-4F); EXPECT_EQ(ta[i], tb[i]); EXPECT_EQ(xa[i], xb[i]);
        }
        ASSERT_EQ(sim_reset(cpu, 42, 0), 0); ASSERT_EQ(sim_reset(gpu, 42, 0), 0);
    }
    sim_destroy(cpu); sim_destroy(gpu);
}
