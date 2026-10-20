
#include <opencv5/opencv2/core.hpp>
#include <opencv5/opencv2/imgproc.hpp>
#include <opencv5/opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <numbers>
#include <omp.h>
using namespace cv;

#define DEBUG

int main() {
	//thread count
	const int tc = 4;

	Mat original = imread("../tcc-images/sudoku.png", IMREAD_GRAYSCALE);
	
	//-------------------------------- ROTACION
	int angulo = 45;
	double rads = (45 * std::numbers::pi_v<double> ) / 180.0f;
	Mat rotated(int(original.rows), int(original.cols), CV_8UC1, Scalar(0));

	#pragma omp parallel for num_threads(tc)
	for (int i = 0; i < rotated.rows; i++) {
		for (int j = 0; j < rotated.cols; j++) {
			int x0 = j - (rotated.cols / 2);
			int y0 = (rotated.rows / 2) - i;

			float u0 = x0 * cos(rads) + y0 * sin(rads);
			float v0 = -1*x0*sin(rads) + y0 * cos(rads);

			int u = u0 + (rotated.cols / 2);
			int v = (rotated.rows / 2) - v0;

			if (u >= original.cols || u < 0 || v >= original.rows || v < 0) continue;
			rotated.at<uchar>(i, j) = original.at<uchar>(v, u);
		}
	}

	#ifdef DEBUG
		namedWindow("original", WINDOW_AUTOSIZE);
		imshow("original", original);

		namedWindow("rotated", WINDOW_AUTOSIZE);
		imshow("rotated", rotated);
		waitKey(0);
	#endif

	imwrite("../tcc-outputs/019-rotated.png"   , rotated    );
}