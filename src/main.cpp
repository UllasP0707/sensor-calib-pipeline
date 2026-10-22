#include "calib_engine.hpp"
#include <iostream>

int main(int argc, char** argv) {
    cv::setNumThreads(1);
    return calib::run_cli(std::vector<std::string>(argv + 1, argv + argc), std::cout, std::cerr);
}
