
#include <opencv5/opencv2/core.hpp>
#include <opencv5/opencv2/imgproc.hpp>
#include <opencv5/opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <numbers>
#include <omp.h>
using namespace cv;

//#define DEBUG

int main() {

	Mat original = imread("../tcc-images/sudoku.png", IMREAD_GRAYSCALE);
	const int tc = 8; //omp_get_num_threads();	

	//------------------------------- ESCALAMIENTO CON INTERPOLACION PARA NO PIXELEAR
	
	//scale by a factor of 3
	float ix = 3;
	float iy = 3;
	
	Mat interp( int(original.rows*iy), int(original.cols*ix), CV_8UC1, Scalar(0));

	#pragma omp parallel for num_threads(tc/2)
	for (int i = 0; i < interp.rows; i++) {
		#pragma omp parallel for num_threads(tc/2)
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

	#ifdef DEBUG
		namedWindow("original", WINDOW_AUTOSIZE);
		imshow("original", original);

		namedWindow("interpolacion", WINDOW_AUTOSIZE);
		imshow("interpolacion", interp);

		waitKey(0);
	#endif

	imwrite("../tcc-outputs/019-interp.png"    , interp     );
}