//parallel version using omp.h
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#ifndef CV_LOAD_IMAGE_GRAYSCALE
#define CV_LOAD_IMAGE_GRAYSCALE cv::IMREAD_GRAYSCALE
#endif
#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif
#include <cmath>
#include "opencv2/video.hpp"
#include <algorithm>
#include <vector>
#include <omp.h>

//#define INSPECT
//#define DEBUG
#ifdef DEBUG
#define NTHREADS 1
#else
#define NTHREADS 4 //Put anything here :>
#endif

using namespace std;
using namespace cv;


/*
	WARNING!
	These ARE NOT absolute paths! Run from 13-...!!!
*/
Mat binary = imread("../tcc-images/sudoku_bin.png", CV_LOAD_IMAGE_GRAYSCALE); 
Mat rErosion(binary.rows, binary.cols, CV_8UC1, Scalar(0));
Mat rDilatacion(binary.rows, binary.cols, CV_8UC1, Scalar(0));

int mask[3][3] = { {0,1,0},{1,1,1},{0,1,0} };
int innerMtx = 3;

// con que uno de los bracitos del elemento este fuera de la mancha, ya no se pinta el origen
bool erosion(int r, int c) {
	int y = 0;
	for (int i = r - int(innerMtx / 2); i <= r + int(innerMtx / 2); i++) {
		int x = 0;
		for (int j = c - int(innerMtx / 2); j <= c + int(innerMtx / 2); j++) {
			if (i < 0 or i >= binary.rows or j < 0 or j >= binary.cols) {
				continue;
			}
			else {
				if (mask[y][x] == 1) {
					if (binary.at<uchar>(i, j) == 0) {
						return false;
					}
				}
			}
			x++;
		}
		y++;
	}
	return true;
}

// con que uno de los bracitos del elemento este dentro de la mancha, se pinta ese origen
bool dilatacion(int r, int c) {
	int y = 0;
	for (int i = r - int(innerMtx / 2); i <= r + int(innerMtx / 2); i++) {
		int x = 0;
		for (int j = c - int(innerMtx / 2); j <= c + int(innerMtx / 2); j++) {
			if (i < 0 or i >= binary.rows or j < 0 or j >= binary.cols) {
				continue;
			}
			else {
				if (mask[y][x] == 1) {
					if (binary.at<uchar>(i, j) == 255) {
						return true;
					}
				}
			}
			x++;
		}
		y++;
	}
	return false;
}

int main() {
	//---------------------------------------------- EROSION
	/*
		Principais atividades são identificação de pontos de alta.
		Subsequentemente, alteração da matriz.
	*/

	#pragma omp parallel for num_threads(NTHREADS)
	for (int i = 0; i < binary.rows; i++) {
		for (int j = 0; j < binary.cols; j++) {
			if (!erosion(i, j)) {
				rErosion.at<uchar>(i, j) = 25;
			}
			else {
				rErosion.at<uchar>(i, j) = 230;
			}
			#ifdef DEBUG
			namedWindow("Resultado", WINDOW_AUTOSIZE);
			imshow("Resultado", rErosion);
			waitKey(1);
			#endif
		}
	}
	

	//---------------------------------------------- DILATACION
	#pragma omp parallel for num_threads(NTHREADS)
	for (int i = 0; i < binary.rows; i++) {
		for (int j = 0; j < binary.cols; j++) {
			if (dilatacion(i, j)) {
				rDilatacion.at<uchar>(i, j) = 230;
			}
			else {
				rDilatacion.at<uchar>(i, j) = 25;
			}
		}
	}

	int dif = 0;
	int dif2 = 0;

	#pragma omp parallel for num_threads(NTHREADS)
	for (int i = 0; i < binary.rows; i++) {
		for (int j = 0; j < binary.cols; j++) {
			if (binary.at<uchar>(i, j) != rErosion.at<uchar>(i, j)) {
				dif++;
			}
			if (binary.at<uchar>(i, j) != rDilatacion.at<uchar>(i, j)) {
				dif2++;
			}
		}
	}

	cout << dif  << endl;
	cout << dif2 << endl;
	
	#ifdef INSPECT
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", binary);

	namedWindow("erosion", WINDOW_AUTOSIZE);
	imshow("erosion", rErosion);

	namedWindow("dilatacion", WINDOW_AUTOSIZE);
	imshow("dilatacion", rDilatacion);
	#endif

	imwrite("../tcc-outputs/013_poriginal.png", binary);
	imwrite("../tcc-outputs/013_perosion.png", rErosion);
	imwrite("../tcc-outputs/013_pdilation.png", rDilatacion);

	#ifdef INSPECT 
	waitKey(0);
	#endif
}