#include "common/bench.hpp"

namespace threshold_algorithm {

    void process(const cv::Mat& image, std::array<cv::Mat, 5>& output) {
        constexpr unsigned char threshold = 80;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < image.rows; ++i) {
            const unsigned char* src = image.ptr<unsigned char>(i);
            unsigned char* binary = output[0].ptr<unsigned char>(i);
            unsigned char* inverse = output[1].ptr<unsigned char>(i);
            unsigned char* trunc = output[2].ptr<unsigned char>(i);
            unsigned char* tozero = output[3].ptr<unsigned char>(i);
            unsigned char* tozero_inverse = output[4].ptr<unsigned char>(i);
            for (int j = 0; j < image.cols; ++j) {
                const unsigned char value = src[j];
                binary[j] = value > threshold ? 255 : 0;
                inverse[j] = value > threshold ? 0 : 255;
                trunc[j] = value > threshold ? threshold : value;
                tozero[j] = value > threshold ? value : 0;
                tozero_inverse[j] = value > threshold ? 0 : value;
            }
        }
    }

}  // namespace threshold_algorithm

#ifndef TCC_ALTERADA_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (!tcc_alterada::arguments(argc, argv)) return 0;
        tcc_alterada::configure_opencv();
        const cv::Mat image = tcc_alterada::load_gray(argv[1]);
        const std::string outdir = argc == 3 ? argv[2] : "";
        std::array<cv::Mat, 5> output;
        for (auto& matrix : output) matrix.create(image.size(), CV_8UC1);

        const double start = tcc_alterada::now_seconds();
        threshold_algorithm::process(image, output);
        const double end = tcc_alterada::now_seconds();
        tcc_alterada::report("threshold", end - start);

        const std::array<std::string, 5> names = {
            "threshold_binary", "threshold_binary_inv", "threshold_trunc",
            "threshold_tozero", "threshold_tozero_inv"
        };
        for (std::size_t k = 0; k < output.size(); ++k)
        tcc_alterada::dump(outdir, names[k], output[k]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERRO: " << error.what() << '\n';
        return 1;
    }
}
#endif