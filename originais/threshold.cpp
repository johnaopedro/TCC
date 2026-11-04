// VERSAO SEQUENCIAL: baseline derivado diretamente do codigo de referencia.
#include "common/bench.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	configure_opencv();
	const char* imgname = (argc > 1) ? argv[1] : "tcc-images/base.png";
	string outdir = (argc > 2) ? argv[2] : "";

	//binary
	Mat image2 = load_gray(imgname);
	//binary inverse
	Mat image3 = load_gray(imgname);
	//trunc
	Mat image4 = load_gray(imgname);
	//tozero
	Mat image5 = load_gray(imgname);
	//tozero inv
	Mat image6 = load_gray(imgname);

	double t0 = now_seconds();

	for (int i = 0; i < image2.rows; i++) {
		for (int j = 0; j < image2.cols; j++) {
			if (image2.at<uchar>(i, j) > 80) {
				image2.at<uchar>(i, j) = 255;
			}
			else {
				image2.at<uchar>(i, j) = 0;
			}
		}
	}

	for (int i = 0; i < image2.rows; i++) {
		for (int j = 0; j < image2.cols; j++) {
			if (image3.at<uchar>(i, j) > 80) {
				image3.at<uchar>(i, j) = 0;
			}
			else {
				image3.at<uchar>(i, j) = 255;
			}
		}
	}

	for (int i = 0; i < image4.rows; i++) {
		for (int j = 0; j < image4.cols; j++) {
			if (image4.at<uchar>(i, j) > 80) {
				image4.at<uchar>(i, j) = 80;
			}
			else {

			}
		}
	}

	for (int i = 0; i < image5.rows; i++) {
		for (int j = 0; j < image5.cols; j++) {
			if (image5.at<uchar>(i, j) > 80) {

			}
			else {
				image5.at<uchar>(i, j) = 0;
			}
		}
	}

	for (int i = 0; i < image6.rows; i++) {
		for (int j = 0; j < image6.cols; j++) {
			if (image6.at<uchar>(i, j) > 80) {
				image6.at<uchar>(i, j) = 0;
			}
			else {

			}
		}
	}

	double t1 = now_seconds();
	report("threshold", t1 - t0);

	dump(outdir, "threshold_binary", image2);
	dump(outdir, "threshold_binary_inv", image3);
	dump(outdir, "threshold_trunc", image4);
	dump(outdir, "threshold_tozero", image5);
	dump(outdir, "threshold_tozero_inv", image6);
	return 0;
}
