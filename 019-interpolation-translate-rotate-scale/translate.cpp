
#include <opencv5/opencv2/core.hpp>
#include <opencv5/opencv2/imgproc.hpp>
#include <opencv5/opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <numbers>
using namespace cv;

#define DEBUG

int main() {

	Mat original = imread("../tcc-images/sudoku.png", IMREAD_GRAYSCALE);

	//------------------------------- TRASLACION
	int tx = 50;
	int ty = 50;
	Mat translated(int(original.rows), int(original.cols), CV_8UC1, Scalar(0));
	for (int i = 0; i < translated.rows; i++) {
		for (int j = 0; j < translated.cols; j++) {
			int	u = int(j - tx);
			int v = int(i - ty);
			if (u >= original.cols || u < 0 || v >= original.rows || v < 0) continue;
			translated.at<uchar>(i, j) = original.at<uchar>(v, u);
		}
	}

	#ifdef DEBUG
		namedWindow("original", WINDOW_AUTOSIZE);
		imshow("original", original);

		namedWindow("translated", WINDOW_AUTOSIZE);
		imshow("translated", translated);
		waitKey(0);
	#endif

	imwrite("../tcc-outputs/019-translated.png", translated );
}