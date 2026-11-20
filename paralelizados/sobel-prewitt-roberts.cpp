#include "common/bench.hpp"

namespace sobel_algorithm {
    using namespace std;
    using namespace cv;

    void process(const Mat& image, Mat& sobelH, Mat& sobelV, Mat& raiz,
        Mat& sobelHR, Mat& sobelVR, Mat& raizR) {
        double minH = 0, maxH = 0, minV = 0, maxV = 0, minR = 0, maxR = 0;
        #pragma omp parallel
        {
            int innerMatrixIndex=3;
            int horiz[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
            int vert[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
            int suma2=0, suma3=0, promedio2=0, promedio3=0;

            // sobel horizontal
            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    suma2 = 0;
                    promedio2 = 0;

                    int y = 0;
                    for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
                    {
                        int x = 0;
                        for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
                        {
                            if ((i + a) >= 0 &&
                                (i + a) < image.rows &&
                                (j + b) >= 0 &&
                                (j + b) < image.cols)
                            {
                                suma2 += (image.at<uchar>(i + a, j + b) * horiz[y][x]);
                            }
                            x++;
                        }
                        y++;
                    }
                    sobelH.at<float>(i, j) = abs(suma2);
                }
            }

            // sobel vertical
            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    suma3 = 0;
                    promedio3 = 0;

                    int y = 0;
                    for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
                    {
                        int x = 0;
                        for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
                        {
                            if ((i + a) >= 0 &&
                                (i + a) < image.rows &&
                                (j + b) >= 0 &&
                                (j + b) < image.cols)
                            {
                                suma3 += (image.at<uchar>(i + a, j + b) * vert[y][x]);
                            }
                            x++;
                        }
                        y++;
                    }
                    sobelV.at<float>(i, j) = abs(suma3);
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    float xValue = (sobelH.at<float>(i, j) * sobelH.at<float>(i, j));
                    float yValue = (sobelV.at<float>(i, j) * sobelV.at<float>(i, j));
                    raiz.at<float>(i, j) = (sqrt(xValue + yValue));
                }
            }

            //todos os pixels, inclusive as bordas zero de Roberts, entram nos extremos
            #pragma omp single
            {
                minH = minV = minR = numeric_limits<double>::infinity();
                maxH = maxV = maxR = -numeric_limits<double>::infinity();
            }
            #pragma omp for schedule(static) reduction(min:minH,minV,minR) reduction(max:maxH,maxV,maxR)
            for (int i = 0; i < image.rows; ++i) {
                for (int j = 0; j < image.cols; ++j) {
                    const double h = sobelH.at<float>(i, j);
                    const double v = sobelV.at<float>(i, j);
                    const double r = raiz.at<float>(i, j);
                    minH = min(minH, h); maxH = max(maxH, h);
                    minV = min(minV, v); maxV = max(maxV, v);
                    minR = min(minR, r); maxR = max(maxR, r);
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    float valH = 0;
                    float valV = 0;
                    float valR = 0;
                    if (maxH != minH) {
                        valH = (255 * (sobelH.at<float>(i, j) - minH)) / (maxH - minH);
                    }
                    if (maxV != minV) {
                        valV = (255 * (sobelV.at<float>(i, j) - minV)) / (maxV - minV);
                    }
                    if (maxR != minR) {
                        valR = (255 * (raiz.at<float>(i, j) - minR)) / (maxR - minR);
                    }
                    sobelHR.at<uchar>(i, j) = abs(valH);
                    sobelVR.at<uchar>(i, j) = abs(valV);
                    raizR.at<uchar>(i, j) = abs(valR);
                }
            }

        }
    }

}  // namespace sobel_algorithm

namespace prewitt_algorithm {
    using namespace std;
    using namespace cv;

    void process(const Mat& image, Mat& prewittH, Mat& prewittV, Mat& raizP,
        Mat& prewittHR, Mat& prewittVR, Mat& raizPR) {
        double minH = 0, maxH = 0, minV = 0, maxV = 0, minR = 0, maxR = 0;
        #pragma omp parallel
        {
            int innerMatrixIndex=3;
            int horizP[3][3]={{-1,0,1},{-1,0,1},{-1,0,1}};
            int vertP[3][3]={{-1,-1,-1},{0,0,0},{1,1,1}};
            int suma2=0, suma3=0, promedio2=0, promedio3=0;

            //---------------------------------------------- PREWITT ---------------------------------------
            suma2 = 0;
            promedio2 = 0;
            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    suma2 = 0;
                    promedio2 = 0;

                    int y = 0;
                    for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
                    {
                        int x = 0;
                        for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
                        {
                            if ((i + a) >= 0 &&
                                (i + a) < image.rows &&
                                (j + b) >= 0 &&
                                (j + b) < image.cols)
                            {
                                suma2 += (image.at<uchar>(i + a, j + b) * horizP[y][x]);
                            }
                            x++;
                        }
                        y++;
                    }
                    prewittH.at<float>(i, j) = abs(suma2);
                }
            }

            suma3 = 0;
            promedio3 = 0;
            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    suma3 = 0;
                    promedio3 = 0;

                    int y = 0;
                    for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
                    {
                        int x = 0;
                        for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
                        {
                            if ((i + a) >= 0 &&
                                (i + a) < image.rows &&
                                (j + b) >= 0 &&
                                (j + b) < image.cols)
                            {
                                suma3 += (image.at<uchar>(i + a, j + b) * vertP[y][x]);
                            }
                            x++;
                        }
                        y++;
                    }
                    prewittV.at<float>(i, j) = abs(suma3);
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    float xValue = (prewittH.at<float>(i, j) * prewittH.at<float>(i, j));
                    float yValue = (prewittV.at<float>(i, j) * prewittV.at<float>(i, j));
                    raizP.at<float>(i, j) = (sqrt(xValue + yValue));
                }
            }

            //todos os pixels, inclusive as bordas zero de Roberts, entram nos extremos
            #pragma omp single
            {
                minH = minV = minR = numeric_limits<double>::infinity();
                maxH = maxV = maxR = -numeric_limits<double>::infinity();
            }
            #pragma omp for schedule(static) reduction(min:minH,minV,minR) reduction(max:maxH,maxV,maxR)
            for (int i = 0; i < image.rows; ++i) {
                for (int j = 0; j < image.cols; ++j) {
                    const double h = prewittH.at<float>(i, j);
                    const double v = prewittV.at<float>(i, j);
                    const double r = raizP.at<float>(i, j);
                    minH = min(minH, h); maxH = max(maxH, h);
                    minV = min(minV, v); maxV = max(maxV, v);
                    minR = min(minR, r); maxR = max(maxR, r);
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    float valH = 0;
                    float valV = 0;
                    float valR = 0;
                    if (maxH != minH) {
                        valH = (255 * (prewittH.at<float>(i, j) - minH)) / (maxH - minH);
                    }
                    if (maxV != minV) {
                        valV = (255 * (prewittV.at<float>(i, j) - minV)) / (maxV - minV);
                    }
                    if (maxR != minR) {
                        valR = (255 * (raizP.at<float>(i, j) - minR)) / (maxR - minR);
                    }
                    prewittHR.at<uchar>(i, j) = abs(valH);
                    prewittVR.at<uchar>(i, j) = abs(valV);
                    raizPR.at<uchar>(i, j) = abs(valR);
                }
            }

        }
    }

}  // namespace prewitt_algorithm

namespace roberts_algorithm {
    using namespace std;
    using namespace cv;

    void process(const Mat& image, Mat& robertH, Mat& robertV, Mat& raizRob,
        Mat& robertHR, Mat& robertVR, Mat& raizRobR) {
        double minH = 0, maxH = 0, minV = 0, maxV = 0, minR = 0, maxR = 0;
        #pragma omp parallel
        {

            //------------------------------------ ROBERTS -----------------------------------
            #pragma omp for schedule(static)
            for (int i = 1; i < image.rows - 1; i++)
            {
                for (int j = 1; j < image.cols - 1; j++)
                {
                    float resH = 0;
                    float resV = 0;
                    resH = -1 * (image.at<uchar>(i, j)) + (image.at<uchar>(i + 1, j + 1));
                    resV = -1 * (image.at<uchar>(i, j + 1)) + (image.at<uchar>(i + 1, j));
                    robertH.at<float>(i, j) = abs(resH);
                    robertV.at<float>(i, j) = abs(resV);
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    float xValue = (robertH.at<float>(i, j) * robertH.at<float>(i, j));
                    float yValue = (robertV.at<float>(i, j) * robertV.at<float>(i, j));
                    raizRob.at<float>(i, j) = (sqrt(xValue + yValue));
                }
            }

            // todos os pixels, inclusive as bordas zero de Roberts, entram nos extremos
            #pragma omp single
            {
                minH = minV = minR = numeric_limits<double>::infinity();
                maxH = maxV = maxR = -numeric_limits<double>::infinity();
            }
            #pragma omp for schedule(static) reduction(min:minH,minV,minR) reduction(max:maxH,maxV,maxR)
            for (int i = 0; i < image.rows; ++i) {
                for (int j = 0; j < image.cols; ++j) {
                    const double h = robertH.at<float>(i, j);
                    const double v = robertV.at<float>(i, j);
                    const double r = raizRob.at<float>(i, j);
                    minH = min(minH, h); maxH = max(maxH, h);
                    minV = min(minV, v); maxV = max(maxV, v);
                    minR = min(minR, r); maxR = max(maxR, r);
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    float valH = 0;
                    float valV = 0;
                    float valR = 0;
                    if (maxH != minH) {
                        valH = (255 * (robertH.at<float>(i, j) - minH)) / (maxH - minH);
                    }
                    if (maxV != minV) {
                        valV = (255 * (robertV.at<float>(i, j) - minV)) / (maxV - minV);
                    }
                    if (maxR != minR) {
                        valR = (255 * (raizRob.at<float>(i, j) - minR)) / (maxR - minR);
                    }
                    robertHR.at<uchar>(i, j) = abs(valH);
                    robertVR.at<uchar>(i, j) = abs(valV);
                    raizRobR.at<uchar>(i, j) = abs(valR);
                }
            }

        }
    }

}  // namespace roberts_algorithm

namespace edges_algorithm {

    //nove matrizes de gradientes e nove de saida, alocadas antes do cronometro
    //indices de operador: 0=Sobel, 1=Prewitt, 2=Roberts
    struct Result {
        std::array<cv::Mat, 3> horizontal, vertical, magnitude;
        std::array<cv::Mat, 3> out_h, out_v, out_mag;

        explicit Result(cv::Size size) {
            for (int k = 0; k < 3; ++k) {
                horizontal[k] = cv::Mat(size, CV_32FC1, cv::Scalar(0));
                vertical[k] = cv::Mat(size, CV_32FC1, cv::Scalar(0));
                magnitude[k] = cv::Mat(size, CV_32FC1, cv::Scalar(0));
                out_h[k] = cv::Mat(size, CV_8UC1, cv::Scalar(0));
                out_v[k] = cv::Mat(size, CV_8UC1, cv::Scalar(0));
                out_mag[k] = cv::Mat(size, CV_8UC1, cv::Scalar(0));
            }
        }
    };

    void process(const cv::Mat& image, Result& result) {
        sobel_algorithm::process(image, result.horizontal[0], result.vertical[0],
            result.magnitude[0], result.out_h[0],
            result.out_v[0], result.out_mag[0]);
        prewitt_algorithm::process(image, result.horizontal[1], result.vertical[1],
            result.magnitude[1], result.out_h[1],
            result.out_v[1], result.out_mag[1]);
        roberts_algorithm::process(image, result.horizontal[2], result.vertical[2],
            result.magnitude[2], result.out_h[2],
            result.out_v[2], result.out_mag[2]);
    }

}  // namespace edges_algorithm

#ifndef TCC_ALTERADA_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (!tcc_alterada::arguments(argc, argv)) return 0;
        tcc_alterada::configure_opencv();
        const cv::Mat image = tcc_alterada::load_gray(argv[1]);
        const std::string outdir = argc == 3 ? argv[2] : "";
        edges_algorithm::Result result(image.size());

        const double start = tcc_alterada::now_seconds();
        edges_algorithm::process(image, result);
        const double end = tcc_alterada::now_seconds();
        tcc_alterada::report("sobel-prewitt-roberts", end - start);

        const std::array<std::string, 3> names = {"sobel", "prewitt", "roberts"};
        for (int k = 0; k < 3; ++k) {
            tcc_alterada::dump(outdir, names[k] + "_h", result.out_h[k]);
            tcc_alterada::dump(outdir, names[k] + "_v", result.out_v[k]);
            tcc_alterada::dump(outdir, names[k] + "_mag", result.out_mag[k]);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERRO: " << error.what() << '\n';
        return 1;
    }
}
#endif