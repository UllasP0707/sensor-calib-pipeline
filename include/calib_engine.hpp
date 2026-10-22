#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace calib {
struct Options {
    std::filesystem::path image_dir;
    int board_w = 9;
    int board_h = 6;
    double square_size = 0.025;
};
struct Result {
    cv::Mat camera_matrix;       // CV_64F, 3 x 3
    cv::Mat distortion;          // CV_64F, 1 x 5: k1, k2, p1, p2, k3
    cv::Size image_size;
    std::size_t total_processed = 0;
    double rms = 0.0;
    double execution_time_ms = 0.0;
    std::vector<double> per_view_errors;
    std::vector<std::string> used_files;
    std::vector<std::string> rejected_files;
    nlohmann::json to_json() const;
};
// Throws std::exception on invalid inputs or insufficient usable views.
Result calibrate(const Options& options);
nlohmann::json error_json(const std::string& message);
// Shared CLI entrypoint allows testing the exact error/JSON contract.
int run_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);
}  // namespace calib
