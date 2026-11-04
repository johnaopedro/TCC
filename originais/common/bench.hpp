#ifndef TCC_PARALELIZACAO_MINIMA_BENCH_HPP
#define TCC_PARALELIZACAO_MINIMA_BENCH_HPP

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>

static inline void configure_opencv() {
    cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);
}

static inline double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static inline void report(const std::string& algorithm, double seconds) {
    std::cout << algorithm << " time: " << seconds << " seconds\n";
}

static inline cv::Mat load_gray(const char* path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image: " << path << '\n';
        std::exit(1);
    }
    return image;
}

static inline void dump(const std::string& outdir, const std::string& name,
                        const cv::Mat& image) {
    if (outdir.empty()) return;
    const std::string path = outdir + "/" + name + ".png";
    try {
        if (cv::imwrite(path, image)) return;
    } catch (const cv::Exception& error) {
        std::cerr << "Error writing image: " << path << ": "
                  << error.what() << '\n';
        std::exit(1);
    }
    std::cerr << "Error writing image: " << path << '\n';
    std::exit(1);
}

#endif