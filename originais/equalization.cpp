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

	int arr[256] = { 0 };
	float arr2[256] = { 0 };
	float arr3[256] = { 0 };

	Mat image = load_gray(imgname);
	int height = image.rows;
	int width = image.cols;
	Mat myMat1(height, width, CV_8U, Scalar(0));

	double t0 = now_seconds();

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int index = (int)(image.at<uchar>(i, j));
			arr[index]++;
		}
	}

	//-- PMF
	float total = image.cols*image.rows;
	for (int i = 0; i < 256; i++)
	{
		arr2[i] = float(arr[i]) / total;
	}
	arr3[0] = arr2[0];

	//-- CDF
	for (int i = 1; i < 256; i++)
	{
		arr3[i] = arr2[i] + arr3[i - 1];
	}

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			myMat1.at<uchar>(i, j) = floor((256 - 1)*arr3[image.at<uchar>(i, j)]);

		}
	}

	double t1 = now_seconds();
	report("equalization", t1 - t0);

	dump(outdir, "equalization_out", myMat1);
	return 0;
}
