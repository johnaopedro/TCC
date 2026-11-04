// prepare_images.cpp - gera as cinco categorias de resolucao a partir de uma
// imagem de origem, em tons de cinza e formato sem perdas (PNG), conforme a
// Secao "Conjunto de Dados de Teste" do protocolo.
#include <iostream>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

struct ImageConfig { const char* name; int width, height; };
static constexpr ImageConfig CONFIGS[] = {
    {"base", 500, 500}, {"media", 1024, 1024},
    {"alta", 2048, 2048}, {"uhd4k", 3840, 2160}, {"uhd8k", 7680, 4320}
};

static bool write_image(const std::string& path, const cv::Mat& image) {
    try {
        if (cv::imwrite(path, image)) return true;
        std::cerr << "erro ao gravar " << path << "\n";
    } catch (const cv::Exception& e) {
        std::cerr << "erro ao gravar " << path << ": " << e.what() << "\n";
    }
    return false;
}

int main(int argc, char** argv) {
    const bool verify_only = argc == 3 && std::string(argv[1]) == "--verify";
    const bool verify_source = argc == 4 &&
                               std::string(argv[1]) == "--verify-source";
    if (verify_only || verify_source) {
        cv::Mat source;
        if (verify_source) {
            source = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
            if (source.empty()) {
                std::cerr << "erro: imagem-fonte ausente ou ilegivel: "
                          << argv[2] << "\n";
                return 1;
            }
        }
        const std::string directory = verify_source ? argv[3] : argv[2];
        bool valid = true;
        for (const auto& config : CONFIGS) {
            const std::string path = directory + "/" + config.name + ".png";
            const cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (image.empty()) {
                std::cerr << "erro: imagem ausente ou ilegivel: " << path << "\n";
                valid = false;
                continue;
            }
            if (image.cols != config.width || image.rows != config.height ||
                image.type() != CV_8UC1) {
                std::cerr << "erro: " << path << " possui " << image.cols << "x"
                          << image.rows << ", tipo=" << image.type()
                          << "; esperado " << config.width << "x" << config.height
                          << ", CV_8UC1\n";
                valid = false;
                continue;
            }
            if (verify_source) {
                cv::Mat expected;
                cv::resize(source, expected,
                           cv::Size(config.width, config.height), 0, 0,
                           cv::INTER_AREA);
                if (cv::norm(image, expected, cv::NORM_INF) != 0.0) {
                    std::cerr << "erro: " << path
                              << " nao corresponde a reamostragem de "
                              << argv[2] << "\n";
                    valid = false;
                    continue;
                }
            }
            std::cout << "  OK " << config.name << " " << image.cols << "x"
                      << image.rows << " CV_8UC1";
            if (verify_source) std::cout << " fonte=OK";
            std::cout << "\n";
        }
        return valid ? 0 : 1;
    }
    if (argc != 3 || std::string(argv[1]).rfind("--", 0) == 0) {
        std::cerr << "uso:\n"
                  << "  prepare_images <origem> <destdir>\n"
                  << "  prepare_images --verify <destdir>\n"
                  << "  prepare_images --verify-source <origem> <destdir>\n";
        return 2;
    }
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) { std::cerr << "erro ao ler " << argv[1] << "\n"; return 1; }
    std::string d = argv[2];
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    const bool is_directory = !ec && std::filesystem::is_directory(d, ec);
    if (ec || !is_directory) {
        std::cerr << "erro ao preparar diretorio " << d;
        if (ec) std::cerr << ": " << ec.message();
        std::cerr << "\n";
        return 1;
    }
    for (const auto& config : CONFIGS) {
        cv::Mat r;
        cv::resize(src, r, cv::Size(config.width, config.height), 0, 0,
                   cv::INTER_AREA);
        if (!write_image(d + "/" + config.name + ".png", r)) return 1;
        std::cout << config.name << " " << config.width << "x"
                  << config.height << std::endl;
    }
    return 0;
}
