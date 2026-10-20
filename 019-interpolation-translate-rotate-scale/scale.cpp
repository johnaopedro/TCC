
#include <opencv5/opencv2/core.hpp>
#include <opencv5/opencv2/imgproc.hpp>
#include <opencv5/opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <numbers>
using namespace cv;

#define DEBUG

int main() {

	float cx = 3;
	float cy = 3;
	Mat original = imread("../tcc-images/sudoku.png", IMREAD_GRAYSCALE);
	Mat scaled(int(original.rows*cy), int(original.cols*cx), CV_8UC1, Scalar(0));

	//------------------------------- ESCALAMIENTO 
	
	for (int i = 0; i < scaled.rows; i++) {
		for (int j = 0; j < scaled.cols; j++) {
			int	u = int(j / cx);
			int v = int(i / cy);
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