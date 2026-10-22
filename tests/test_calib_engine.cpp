#include "calib_engine.hpp"

#include <chrono>
#include <fstream>
#include <sstream>
#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
class CalibrationTest : public ::testing::Test {
protected:
    std::filesystem::path directory;
    void SetUp() override {
        cv::setNumThreads(1);
        directory = std::filesystem::temp_directory_path() /
            ("calib_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
    }
    void TearDown() override { std::filesystem::remove_all(directory); }
    void generate_ideal() {
        cv::Mat texture(700, 1000, CV_8UC1, cv::Scalar(245));
        for (int y = 0; y < 7; ++y)
            for (int x = 0; x < 10; ++x)
                if ((x + y) % 2 == 0)
                    cv::rectangle(texture, {x * 100, y * 100, 100, 100}, cv::Scalar(15), cv::FILLED);
        const cv::Mat k = (cv::Mat_<double>(3, 3) << 1000, 0, 640, 0, 1005, 480, 0, 0, 1);
        const std::vector<cv::Point3f> outer{{-0.125F, -0.0875F, 0}, {0.125F, -0.0875F, 0},
                                           {0.125F, 0.0875F, 0}, {-0.125F, 0.0875F, 0}};
        const std::vector<cv::Point2f> source{{0, 0}, {1000, 0}, {1000, 700}, {0, 700}};
        for (int i = 0; i < 15; ++i) {
            const cv::Vec3d rotation(0.10 * (i % 5 - 2), 0.12 * (i % 3 - 1), 0.025 * (i - 7));
            const cv::Vec3d translation(0.018 * (i % 3 - 1), 0.012 * (i % 5 - 2), 0.55 + 0.018 * i);
            std::vector<cv::Point2f> target;
            cv::projectPoints(outer, rotation, translation, k, cv::Mat::zeros(1, 5, CV_64F), target);
            for (auto& point : target) point *= 2;
            cv::Mat frame;
            cv::warpPerspective(texture, frame, cv::getPerspectiveTransform(source, target),
                                {2560, 1920}, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(195));
            cv::resize(frame, frame, {1280, 960}, 0, 0, cv::INTER_AREA);
            cv::GaussianBlur(frame, frame, {3, 3}, 0.55);
            ASSERT_TRUE(cv::imwrite((directory / ("frame_" + std::to_string(i) + ".png")).string(), frame));
        }
    }
};
TEST_F(CalibrationTest, IdealPatternAccuracyAndMatrixDimensions) {
    generate_ideal();
    const auto result = calib::calibrate({directory, 9, 6, 0.025});
    EXPECT_LT(result.rms, 0.35);
    EXPECT_EQ(result.camera_matrix.rows, 3);
    EXPECT_EQ(result.camera_matrix.cols, 3);
    EXPECT_EQ(result.distortion.rows, 1);
    EXPECT_EQ(result.distortion.cols, 5);
    EXPECT_EQ(result.per_view_errors.size(), 15U);
    EXPECT_NEAR(result.camera_matrix.at<double>(0, 0), 1000, 20);
    EXPECT_NEAR(result.camera_matrix.at<double>(1, 1), 1005, 20);
    EXPECT_EQ(result.to_json()["status"], "SUCCESS");
    EXPECT_GT(result.execution_time_ms, 0);
}
TEST_F(CalibrationTest, MissingPathIsStructuredError) {
    std::ostringstream out, err;
    EXPECT_EQ(calib::run_cli({"--image_dir", (directory / "missing").string()}, out, err), 1);
    const auto json = nlohmann::json::parse(out.str());
    EXPECT_EQ(json["status"], "ERROR");
    EXPECT_FALSE(json["error"]["message"].get<std::string>().empty());
    EXPECT_FALSE(err.str().empty());
}
TEST_F(CalibrationTest, MalformedArgumentsAreStructuredErrors) {
    const std::vector<std::vector<std::string>> cases{
        {}, {"--image_dir"}, {"--unknown", "x"},
        {"--image_dir", directory.string(), "--board_w", "9garbage"},
        {"--image_dir", directory.string(), "--board_h", "0"},
        {"--image_dir", directory.string(), "--square_size", "nan"},
        {"--image_dir", directory.string(), "--square_size", "-1"},
        {"--image_dir", directory.string(), "--board_w", "9", "--board_w", "8"}};
    for (const auto& args : cases) {
        std::ostringstream out, err;
        EXPECT_EQ(calib::run_cli(args, out, err), 1);
        EXPECT_EQ(nlohmann::json::parse(out.str())["status"], "ERROR");
    }
}
TEST_F(CalibrationTest, EmptyAndCorruptImagesFailGracefully) {
    EXPECT_THROW(calib::calibrate({directory}), std::invalid_argument);
    std::ofstream(directory / "broken.PNG") << "not a PNG";
    std::ostringstream out, err;
    EXPECT_EQ(calib::run_cli({"--image_dir", directory.string()}, out, err), 1);
    EXPECT_EQ(nlohmann::json::parse(out.str())["status"], "ERROR");
}
TEST_F(CalibrationTest, MixedResolutionsAreRejected) {
    ASSERT_TRUE(cv::imwrite((directory / "a.jpg").string(), cv::Mat(100, 100, CV_8U, cv::Scalar(255))));
    ASSERT_TRUE(cv::imwrite((directory / "b.jpeg").string(), cv::Mat(120, 100, CV_8U, cv::Scalar(255))));
    EXPECT_THROW(calib::calibrate({directory}), std::invalid_argument);
}
TEST_F(CalibrationTest, CorruptFrameCountsInTotalAndOutputMatchesStdout) {
    generate_ideal();
    std::ofstream(directory / "broken.PNG") << "invalid";
    const auto destination = directory / "result.json";
    std::ostringstream out, err;
    ASSERT_EQ(calib::run_cli({"--image_dir", directory.string(), "--output_json", destination.string()}, out, err), 0);
    const auto result = nlohmann::json::parse(out.str());
    EXPECT_EQ(result["frames"]["total_processed"], 16);
    EXPECT_EQ(result["frames"]["valid_used"], 15);
    EXPECT_EQ(result["rejected_files"].size(), 1U);
    std::ifstream file(destination);
    EXPECT_EQ(nlohmann::json::parse(file), result);
}
}  // namespace
