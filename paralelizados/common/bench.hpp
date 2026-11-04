#ifndef TCC_ALTERADA_BENCH_HPP
#define TCC_ALTERADA_BENCH_HPP

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>

namespace tcc_alterada {

inline void configure_opencv() {
    cv::setNumThreads(1);
    cv::ocl::setUseOpenCL(false);
}

inline bool arguments(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Uso: " << argv[0] << " imagem [pasta_de_saida]\n";
        return false;
    }
    if (argc < 2 || argc > 3)
        throw std::runtime_error("uso: " + std::string(argv[0]) +
                                 " imagem [pasta_de_saida]");
    return true;
}

inline cv::Mat load_gray(const char* path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
        throw std::runtime_error("nao foi possivel carregar: " + std::string(path));
    if (image.total() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("imagem excede o limite de contagem da referencia");
    return image;
}

inline double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline void report(const std::string& algorithm, double seconds) {
    std::cout << algorithm << " time: " << std::setprecision(12)
              << std::fixed << seconds << " seconds\n";
}

inline void dump(const std::string& outdir, const std::string& name,
                 const cv::Mat& image) {
    if (outdir.empty()) return;
    std::filesystem::create_directories(outdir);
    const auto path = std::filesystem::path(outdir) / (name + ".png");
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("nao foi possivel gravar: " + path.string());
}

}  // namespace tcc_alterada

#endif
