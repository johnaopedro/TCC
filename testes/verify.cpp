#define TCC_ALTERADA_NO_MAIN
#include "../paralelizados/threshold.cpp"
#include "../paralelizados/equalization.cpp"
#include "../paralelizados/sobel-prewitt-roberts.cpp"
#include "reference.hpp"

int main() {
    try {
        tcc_alterada::configure_opencv();
        const auto cases = validation_cases();
        std::size_t executions = 0, matrices = 0;
        for (const auto& [case_name, image] : cases) {
            State expected(image), actual(image);
            edges_algorithm::Result group(image.size());
            const auto threads = EXP_PAR ? std::vector<int>{1, 2, 4, 6, 8} :
                                           std::vector<int>{1};
            for (const int p : threads) {
                configure(p, "static");
                for (int repeat = 0; repeat < 2; ++repeat) {
                    expected.prepare("threshold");
                    threshold_ref(expected);
                    threshold_algorithm::process(image, actual.thr);
                    matrices += compare(expected, actual, "threshold");
                    ++executions;

                    equalization_ref(expected);
                    equalization_algorithm::process(image, actual.out[0]);
                    matrices += compare(expected, actual, "equalization");
                    ++executions;

                    edges_algorithm::process(image, group);
                    const std::array<std::string, 3> names = {"sobel", "prewitt", "roberts"};
                    for (int k = 0; k < 3; ++k) {
                        expected.prepare(names[k]);
                        reference(names[k])(expected);
                        const std::array<const cv::Mat*, 6> obtained = {
                            &group.horizontal[k], &group.vertical[k], &group.magnitude[k],
                            &group.out_h[k], &group.out_v[k], &group.out_mag[k]
                        };
                        for (int m = 0; m < 6; ++m) {
                            const cv::Mat& baseline = m < 3 ? expected.grad[m] : expected.out[m - 3];
                            if (!exact(baseline, *obtained[m]))
                                throw std::runtime_error("divergencia: " + case_name + "/" + names[k]);
                            ++matrices;
                        }
                    }
                    ++executions;
                }
            }
            std::cout << "OK " << case_name << '\n';
        }
        std::cout << "OK: " << cases.size() << " entradas, " << executions
                  << " chamadas aos tres programas, " << matrices
                  << " matrizes identicas byte a byte (incluindo gradientes float).\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERRO: " << error.what() << '\n';
        return 1;
    }
}
