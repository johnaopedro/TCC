
#include <opencv5/opencv2/core.hpp>
#include <opencv5/opencv2/imgproc.hpp>
#include <opencv5/opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <numbers>
#include <omp.h>
using namespace cv;

//#define DEBUG
constexpr double pi = std::numbers::pi_v<double>; 

int main() {
	//thread count
	constexpr int tc = 8;
	static_assert(tc >= 4);
	constexpr int l1 = tc-2;
	constexpr int l2 = 2;
	std::cout << tc << std::endl;

	//Mat original = imread("../tcc-images/sudoku.png", IMREAD_GRAYSCALE);
	Mat original = imread("../tcc-images/sudoku_bin_4000_4036.png", IMREAD_GRAYSCALE);
	
	//-------------------------------- ROTACION
	int angulo = 45;
	double rads = (45 * pi) / 180.0f;
	Mat rotated(int(original.rows), int(original.cols), CV_8UC1, Scalar(0));
	
	int x0  ;  
	int y0  ;  
	float u0;
	float v0;
	int u   ;
	int v   ;
	#pragma omp parallel for num_threads(tc) \
		private(x0, y0, u0, v0, u , v )
	//#pragma omp parallel for num_threads(l1)
	for (int i = 0; i < rotated.rows; i++) {
		//#pragma omp parallel for num_threads(l2) \
			private(x0, y0, u0, v0, u , v )
		for (int j = 0; j < rotated.cols; j++) {
			x0   = j - (rotated.cols / 2);
			y0   = (rotated.rows / 2) - i;
			u0   = x0 * cos(rads) + y0 * sin(rads);
			v0   = -1*x0*sin(rads) + y0 * cos(rads);
			u    = u0 + (rotated.cols / 2);
			v    = (rotated.rows / 2) - v0;
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