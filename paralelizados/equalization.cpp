#include "common/bench.hpp"

namespace equalization_algorithm {

    void process(const cv::Mat& image, cv::Mat& output) {
        int histogram[256] = {};
        float pmf[256] = {}, cdf[256] = {};
        unsigned char lut[256] = {};

        #pragma omp parallel
        {
            #pragma omp for schedule(static) reduction(+:histogram[:256])
            for (int i = 0; i < image.rows; ++i) {
                for (int j = 0; j < image.cols; ++j) {
                    histogram[image.at<unsigned char>(i, j)]++;
                }
            }

            //a barreira anterior consolida o histograma. A do single publica a LUT.
            #pragma omp single
            {
                float total = image.cols * image.rows;
                for (int k = 0; k < 256; ++k) {
                    pmf[k] = float(histogram[k]) / total;
                }
                cdf[0] = pmf[0];
                for (int k = 1; k < 256; ++k) {
                    cdf[k] = pmf[k] + cdf[k - 1];
                }
                for (int k = 0; k < 256; ++k) {
                    lut[k] = std::floor(255 * cdf[k]);
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; ++i) {
                const unsigned char* src = image.ptr<unsigned char>(i);
                unsigned char* dst = output.ptr<unsigned char>(i);
                for (int j = 0; j < image.cols; ++j) {
                    dst[j] = lut[src[j]];
                }
            }
        }
    }

}  // namespace equalization_algorithm

#ifndef TCC_ALTERADA_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (!tcc_alterada::arguments(argc, argv)) return 0;
        tcc_alterada::configure_opencv();
        const cv::Mat image = tcc_alterada::load_gray(argv[1]);
        const std::string outdir = argc == 3 ? argv[2] : "";
        cv::Mat output(image.size(), CV_8UC1, cv::Scalar(0));

        const double start = tcc_alterada::now_seconds();
        equalization_algorithm::process(image, output);
        const double end = tcc_alterada::now_seconds();
        tcc_alterada::report("equalization", end - start);

        tcc_alterada::dump(outdir, "equalization_out", output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERRO: " << error.what() << '\n';
        return 1;
    }
}
#endif
