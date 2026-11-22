
#include <opencv5/opencv2/core.hpp>
#include <opencv5/opencv2/imgproc.hpp>
#include <opencv5/opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <numbers>
using namespace cv;

#define DEBUG

int main() {
	//initializing thread count!
	constexpr int tc = 8;
	static_assert(tc >= 4);
	constexpr int l1 = tc-2;
	constexpr int l2 = 2;
	std::cout << tc << std::endl;

	float cx = 3;
	float cy = 3;
	#ifdef DEBUG
	Mat original = imread("../tcc-images/sudoku.png", IMREAD_GRAYSCALE);
	#else
	Mat original = imread("../tcc-images/sudoku_bin_4000_4036.png", IMREAD_GRAYSCALE);
	#endif
	Mat scaled(int(original.rows*cy), int(original.cols*cx), CV_8UC1, Scalar(0));

	//------------------------------- ESCALAMIENTO 
	int	u; 
	int v; 
	#pragma omp parallel for num_threads(tc) \
		private(u, v)
	for (int i = 0; i < scaled.rows; i++) {
		for (int j = 0; j < scaled.cols; j++) {
			u = int(j / cx);
			v = int(i / cy);
			if ( u > original.cols ||
				 u < 0             ||
				 v > original.rows ||
				 v < 0
			) 
					continue;
			scaled.at<uchar>(i, j) = original.at<uchar>(v, u);
		}
	}

	
	#ifdef DEBUG
		namedWindow("original", WINDOW_AUTOSIZE);
		imshow("original", original);

		namedWindow("escalado simple", WINDOW_AUTOSIZE);
		imshow("escalado simple", scaled);

		waitKey(0);
	#endif

	imwrite("../tcc-outputs/019-scaled.png"    , scaled     );
}