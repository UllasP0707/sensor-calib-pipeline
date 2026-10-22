#include "calib_engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <exception>
#include <fstream>
#include <map>
#include <stdexcept>
#include <thread>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace calib {
namespace {
struct Frame {
    cv::Size size;
    std::vector<cv::Point2f> corners;
    std::string rejection;
};
void validate(const Options& o) {
    if (o.board_w < 3 || o.board_h < 3 || o.board_w > 100 || o.board_h > 100)
        throw std::invalid_argument("Board dimensions must be in [3, 100]");
    if (!std::isfinite(o.square_size) || o.square_size <= 0 || o.square_size > 100)
        throw std::invalid_argument("Square size must be finite and in (0, 100] meters");
    if (!std::filesystem::is_directory(o.image_dir))
        throw std::invalid_argument("Image directory does not exist: " + o.image_dir.string());
}
int parse_int(const std::string& value) {
    std::size_t n = 0;
    const int result = std::stoi(value, &n);
    if (n != value.size()) throw std::invalid_argument("Invalid integer: " + value);
    return result;
}
double parse_double(const std::string& value) {
    std::size_t n = 0;
    const double result = std::stod(value, &n);
    if (n != value.size()) throw std::invalid_argument("Invalid number: " + value);
    return result;
}
}  // namespace

Result calibrate(const Options& o) {
    const auto start = std::chrono::high_resolution_clock::now();
    validate(o);
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(o.image_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw std::invalid_argument("No PNG/JPG/JPEG frames found");

    // Bounded frame concurrency: no unbounded image queue, deterministic output order.
    // The executable sets OpenCV's internal thread count to 1 to avoid oversubscription.
    std::vector<Frame> frames(paths.size());
    std::atomic<std::size_t> next{0};
    const auto workers = std::min<std::size_t>({paths.size(), 8,
        std::max(1U, std::thread::hardware_concurrency())});
    auto detect = [&]() {
        for (;;) {
            const auto i = next.fetch_add(1);
            if (i >= paths.size()) return;
            auto& f = frames[i];
            try {
                const auto gray = cv::imread(paths[i].string(), cv::IMREAD_GRAYSCALE);
                if (gray.empty()) { f.rejection = "Unreadable image"; continue; }
                f.size = gray.size();
                const bool found = cv::findChessboardCorners(gray, {o.board_w, o.board_h},
                    f.corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                    cv::CALIB_CB_FAST_CHECK);
                if (!found) { f.corners.clear(); f.rejection = "Checkerboard not found"; continue; }
                cv::cornerSubPix(gray, f.corners, {11, 11}, {-1, -1},
                    {cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001});
            } catch (const std::exception& e) {
                f.corners.clear();
                f.rejection = e.what();
            }
        }
    };
    std::vector<std::thread> threads;
    try {
        for (std::size_t i = 0; i < workers; ++i) threads.emplace_back(detect);
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    for (auto& thread : threads) thread.join();

    Result r;
    r.total_processed = paths.size();
    // Mixing resolutions silently biases intrinsics; reject the entire run.
    for (const auto& f : frames) {
        if (f.size.empty()) continue;
        if (r.image_size.empty()) r.image_size = f.size;
        if (f.size != r.image_size) throw std::invalid_argument("Mixed image resolutions are not supported");
    }
    std::vector<std::vector<cv::Point2f>> image_points;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].rejection.empty() && !frames[i].corners.empty()) {
            image_points.push_back(frames[i].corners);
            r.used_files.push_back(paths[i].filename().string());
        } else {
            r.rejected_files.push_back(paths[i].filename().string() + ": " + frames[i].rejection);
        }
    }
    if (image_points.size() < 3) throw std::invalid_argument("At least 3 usable checkerboard views are required");
    std::vector<cv::Point3f> board;
    for (int y = 0; y < o.board_h; ++y)
        for (int x = 0; x < o.board_w; ++x)
            board.emplace_back(static_cast<float>(x * o.square_size),
                               static_cast<float>(y * o.square_size), 0.0F);
    const std::vector<std::vector<cv::Point3f>> object_points(image_points.size(), board);
    r.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    r.distortion = cv::Mat::zeros(1, 5, CV_64F);
    std::vector<cv::Mat> rotations, translations;
    cv::calibrateCamera(object_points, image_points, r.image_size, r.camera_matrix,
                        r.distortion, rotations, translations, 0,
                        {cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 100, 1e-9});
    r.distortion = r.distortion.reshape(1, 1).clone();
    if (!cv::checkRange(r.camera_matrix) || !cv::checkRange(r.distortion) ||
        r.camera_matrix.at<double>(0, 0) <= 0 || r.camera_matrix.at<double>(1, 1) <= 0 ||
        r.distortion.total() != 5) throw std::runtime_error("Calibration produced invalid parameters");
    double squared_error = 0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < image_points.size(); ++i) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(board, rotations[i], translations[i], r.camera_matrix, r.distortion, projected);
        const double sum = cv::norm(image_points[i], projected, cv::NORM_L2SQR);
        r.per_view_errors.push_back(std::sqrt(sum / static_cast<double>(projected.size())));
        squared_error += sum;
        count += projected.size();
    }
    r.rms = std::sqrt(squared_error / static_cast<double>(count));
    if (!std::isfinite(r.rms)) throw std::runtime_error("Non-finite reprojection error");
    r.execution_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
    return r;
}

nlohmann::json Result::to_json() const {
    std::vector<double> coefficients;
    for (int i = 0; i < 5; ++i) coefficients.push_back(distortion.at<double>(0, i));
    return {{"status", "SUCCESS"}, {"rms_reprojection_error_px", rms},
        {"execution_time_ms", execution_time_ms},
        {"image_dimensions", {{"width", image_size.width}, {"height", image_size.height}}},
        {"frames", {{"total_processed", total_processed}, {"valid_used", used_files.size()}}},
        {"intrinsics", {{"fx", camera_matrix.at<double>(0, 0)}, {"fy", camera_matrix.at<double>(1, 1)},
                        {"cx", camera_matrix.at<double>(0, 2)}, {"cy", camera_matrix.at<double>(1, 2)}}},
        {"distortion", {{"model", "brown_conrady_5"}, {"coefficients", coefficients}}},
        {"per_view_errors", per_view_errors}, {"used_files", used_files}, {"rejected_files", rejected_files}};
}
nlohmann::json error_json(const std::string& message) {
    return {{"status", "ERROR"}, {"error", {{"message", message}}}};
}
int run_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    try {
        Options options;
        std::filesystem::path output;
        std::map<std::string, bool> seen;
        for (std::size_t i = 0; i < args.size(); ++i) {
            const auto& key = args[i];
            if (key != "--image_dir" && key != "--board_w" && key != "--board_h" &&
                key != "--square_size" && key != "--output_json")
                throw std::invalid_argument("Unknown argument: " + key);
            if (seen[key]) throw std::invalid_argument("Duplicate argument: " + key);
            seen[key] = true;
            if (++i == args.size()) throw std::invalid_argument("Missing value for " + key);
            if (key == "--image_dir") options.image_dir = args[i];
            else if (key == "--board_w") options.board_w = parse_int(args[i]);
            else if (key == "--board_h") options.board_h = parse_int(args[i]);
            else if (key == "--square_size") options.square_size = parse_double(args[i]);
            else output = args[i];
        }
        if (options.image_dir.empty()) throw std::invalid_argument("--image_dir is required");
        const auto result = calibrate(options).to_json();
        const auto serialized = result.dump(2) + "\n";
        if (!output.empty()) {
            // Refuse replacement; Python owns atomic, durable audit log persistence.
            if (std::filesystem::exists(output)) throw std::runtime_error("Output file already exists");
            std::ofstream file(output);
            if (!file || !(file << serialized) || !file.flush())
                throw std::runtime_error("Unable to write output JSON: " + output.string());
        }
        out << serialized;
        return 0;
    } catch (const std::exception& e) {
        err << "Calibration failed: " << e.what() << '\n';
        out << error_json(e.what()).dump() << '\n';
        return 1;
    }
}
}  // namespace calib
