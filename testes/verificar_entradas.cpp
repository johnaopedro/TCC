#include "../originais/common/bench.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <cmath>
#include <math.h>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#if BASE_PAR_TEST
#ifndef _OPENMP
#error O teste paralelo exige compilacao com OpenMP.
#endif
#include <omp.h>
#endif

namespace test_io {
    cv::Mat input;
    std::map<std::string, cv::Mat> outputs;

    cv::Mat load_gray(const char*) {
        if (input.isContinuous()) return input.clone();
        cv::Mat parent(input.rows, input.cols + 4, input.type(), cv::Scalar(0));
        cv::Mat roi = parent(cv::Rect(1, 0, input.cols, input.rows));
        input.copyTo(roi);
        return roi;
    }

    void dump(const std::string&, const std::string& name, const cv::Mat& image) {
        if (!outputs.emplace(name, image.clone()).second)
            throw std::runtime_error("saida duplicada: " + name);
    }

    void report(const std::string&, double) {}
}

#define main original_entry
namespace threshold_program {
    using test_io::load_gray;
    using test_io::dump;
    using test_io::report;
#include "../originais/threshold.cpp"
}
namespace equalization_program {
    using test_io::load_gray;
    using test_io::dump;
    using test_io::report;
#include "../originais/equalization.cpp"
}
namespace edges_program {
    using test_io::load_gray;
    using test_io::dump;
    using test_io::report;
#include "../originais/sobel-prewitt-roberts.cpp"
}
#undef main
#include "reference.hpp"

using Outputs = std::map<std::string, cv::Mat>;
using Entry = int (*)(int, char**);

Outputs reference_outputs(const cv::Mat& input) {
    State state(input);
    Outputs expected;
    const std::array<std::string, 5> threshold_names = {
        "threshold_binary", "threshold_binary_inv", "threshold_trunc",
        "threshold_tozero", "threshold_tozero_inv"
    };
    state.prepare("threshold");
    threshold_ref(state);
    for (int k = 0; k < 5; ++k) expected.emplace(threshold_names[k], state.thr[k].clone());
    equalization_ref(state);
    expected.emplace("equalization_out", state.out[0].clone());
    for (const std::string name : {"sobel", "prewitt", "roberts"}) {
        state.prepare(name);
        reference(name)(state);
        expected.emplace(name + "_h", state.out[0].clone());
        expected.emplace(name + "_v", state.out[1].clone());
        expected.emplace(name + "_mag", state.out[2].clone());
    }
    return expected;
}

Outputs run_program(Entry entry, const cv::Mat& input) {
    test_io::input = input;
    test_io::outputs.clear();
    char executable[] = "verificacao";
    char input_path[] = "entrada-em-memoria";
    char output_path[] = "captura-em-memoria";
    char* arguments[] = {executable, input_path, output_path, nullptr};
    if (entry(3, arguments) != 0) throw std::runtime_error("programa retornou erro");
    return std::move(test_io::outputs);
}

std::size_t check_outputs(const Outputs& expected, const Outputs& actual,
                          const std::string& label, std::size_t required) {
    if (actual.size() != required) throw std::runtime_error("quantidade de saidas: " + label);
    for (const auto& [name, matrix] : actual) {
        const auto found = expected.find(name);
        if (found == expected.end() || !exact(found->second, matrix))
            throw std::runtime_error("divergencia: " + label + "/" + name);
    }
    return actual.size();
}


#define TCC_ALTERADA_NO_MAIN
#include "../paralelizados/threshold.cpp"
#include "../paralelizados/equalization.cpp"
#include "../paralelizados/sobel-prewitt-roberts.cpp"

int main(int argc, char** argv) {
    try {
        configure_opencv();
#if BASE_PAR_TEST
        omp_set_dynamic(0);
        if (argc < 3) throw std::runtime_error("informe threads e imagens");
        omp_set_num_threads(std::stoi(argv[1]));
#endif
        const std::array<Entry, 3> programs = {
            threshold_program::original_entry, equalization_program::original_entry,
            edges_program::original_entry
        };
        const std::array<std::string, 5> thr_names = {
            "threshold_binary", "threshold_binary_inv", "threshold_trunc",
            "threshold_tozero", "threshold_tozero_inv"
        };
        std::size_t comparisons = 0;
        if (argc < 3) throw std::runtime_error("informe threads e imagens preparadas");
        for (int arg = 2; arg < argc; ++arg) {
            const cv::Mat input = cv::imread(argv[arg], cv::IMREAD_GRAYSCALE);
            if (input.empty()) throw std::runtime_error("imagem ausente");
            std::array<cv::Mat, 5> threshold;
            for (auto& out : threshold) out.create(input.size(), CV_8UC1);
            threshold_algorithm::process(input, threshold);
            Outputs obtained;
            for (int k = 0; k < 5; ++k) obtained.emplace(thr_names[k], threshold[k]);
            comparisons += check_outputs(run_program(programs[0], input), obtained, argv[arg], 5);

            cv::Mat equalized(input.size(), CV_8UC1);
            equalization_algorithm::process(input, equalized);
            obtained = {{"equalization_out", equalized}};
            comparisons += check_outputs(run_program(programs[1], input), obtained, argv[arg], 1);

            edges_algorithm::Result edges(input.size());
            edges_algorithm::process(input, edges);
            obtained.clear();
            const std::array<std::string, 3> names = {"sobel", "prewitt", "roberts"};
            for (int k = 0; k < 3; ++k) {
                obtained.emplace(names[k] + "_h", edges.out_h[k]);
                obtained.emplace(names[k] + "_v", edges.out_v[k]);
                obtained.emplace(names[k] + "_mag", edges.out_mag[k]);
            }
            comparisons += check_outputs(run_program(programs[2], input), obtained, argv[arg], 9);
            std::cout << "OK " << argv[arg] << '\n';
        }
        std::cout << "OK: " << argc - 2 << " entradas, " << comparisons
                  << " saidas referencia/paralelizada identicas byte a byte.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERRO: " << error.what() << '\n';
        return 1;
    }
}
