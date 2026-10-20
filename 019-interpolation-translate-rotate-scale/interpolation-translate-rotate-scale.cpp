
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

	//------------------------------- ESCALAMIENTO CON INTERPOLACION PARA NO PIXELEAR
	
	//scale by a factor of 3
	float ix = 3;
	float iy = 3;
	
	Mat interp( int(original.rows*iy), int(original.cols*ix), CV_8UC1, Scalar(0));

	for (int i = 0; i < interp.rows; i++) {
		for (int j = 0; j < interp.cols; j++) {
			int q11 = 0, q22 = 0, q21 = 0;
			float px = (j / ix);
			float py = (i / iy);

			/*
				we're scaling by 3, then

				og image might be:
					11 12 13
					21 22 23
					31 32 33
				and interp'd should go to:

					11 12 13 14 15 16 17 18 19
					21 22 23 24 25 26 27 28 29
					.	.
					.		.
					.			.
					91 92 93 94 95 96 97 98 99
			

				then, say we're treating point 44 of interped;
				4,4 -> 1.333..., 1.3333... -> 1,1;

				q12 ---------------------- q22
				 |                          |
				 |                          |
				 |                          |
				 |                          |
				 |                          |
				 |                          |
				q11------------------------q21


			*/

			int q12 = original.at<uchar>(int(py), int(px)); //this will truncate values, most certainly on float to int conversion.
			

			//this if block will handle edge cases alright...
			
			if (int(py + 1) < original.rows) //this might be a problem in a 1x1 image, i guess.
				q11 = original.at<uchar>(int(py + 1), int(px)); 
			
			if (int(px + 1) < original.cols) 
				q22 = original.at<uchar>(int(py), int(px + 1));
			
			if (int(px + 1) < original.cols &&
				int(py + 1) < original.rows
			)
				q21 = original.at<uchar>(int(py + 1), int(px + 1));

			//collection qs are intensities in the original image.
			
			int x1 = int(px);
			int x2 = int(px + 1);
			int y1 = int(py + 1);
			int y2 = int(py);

			int r1 = ( (q21 - q11) / (x2 - x1)) * (px - x1) + q11; //upper line
			int r2 = ( (q22 - q12) / (x2 - x1)) * (px - x1) + q12; //lower line 
			

			/*

				     (r2 - r1) * (py - y1)	  r1
				P =	--------------------- + ------
						(y2 - y1)			  1
			
			*/
			
			int P = ((r2 - r1) / (y2 - y1)) * (py - y1) + r1;
			interp.at<uchar>(i, j) = P;
		}
	}

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

	//-------------------------------- ROTACION
	int angulo = 45;
	double rads = (45 * std::numbers::pi_v<double> ) / 180.0f;
	Mat rotated(int(original.rows), int(original.cols), CV_8UC1, Scalar(0));
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

		namedWindow("escalado simple", WINDOW_AUTOSIZE);
		imshow("escalado simple", scaled);

		namedWindow("interpolacion", WINDOW_AUTOSIZE);
		imshow("interpolacion", interp);

		namedWindow("translated", WINDOW_AUTOSIZE);
		imshow("translated", translated);

		namedWindow("rotated", WINDOW_AUTOSIZE);
		imshow("rotated", rotated);
		waitKey(0);
	#endif

	imwrite("../tcc-outputs/019-scaled.png"    , scaled     );
	imwrite("../tcc-outputs/019-interp.png"    , interp     );
	imwrite("../tcc-outputs/019-translated.png", translated );
	imwrite("../tcc-outputs/019-rotated.png"   , rotated    );
}