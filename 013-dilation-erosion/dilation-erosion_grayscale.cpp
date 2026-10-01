// most of this was NOT written by l0c. Do be sure to check the original work of the author!

#include <opencv5/opencv2/core/core.hpp>
#include <opencv5/opencv2/imgproc/imgproc.hpp>
#include <opencv5/opencv2/highgui/highgui.hpp>
#include <opencv5/opencv2/video.hpp>
#include <iostream>
#ifndef CV_LOAD_IMAGE_GRAYSCALE
#define CV_LOAD_IMAGE_GRAYSCALE cv::IMREAD_GRAYSCALE
#endif
#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;
using namespace cv;

Mat binary = imread("../tcc-images/sudoku.png", CV_LOAD_IMAGE_GRAYSCALE);
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

int grayscale_erosion(Mat &erodeme, int w, int z){
	//loop through image
	std::vector<int> process;
	for(int i = 0; i < erodeme.rows; i++){
		for(int j = 0; j < erodeme.cols; j++){
			process.clear();
			//loop through kernel
			for(int k = 0; k < w; k++){
				for(int l = 0; l < z; l++){
					if(!( 0 < i + k || i + k < erodeme.rows || 0 < j + l || j + l < erodeme.cols)){
						continue; //continues this one -> it would access forbidden places!
					}
					process.emplace_back(
						erodeme.at<char>(i+k, j+l) //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<fixme;
					);
				}
			}

			char res = *(std::min_element(process.begin(), process.end())); // max -> it -> *it -> int -> clamp it to an int; go crazy!
			erodeme.at<char>(i, j) = res;
		}
	}
}

int grayscale_dilate(Mat &erodeme, int w, int z){
	//loop through image
	std::vector<int> process;
	for(int i = 0; i < erodeme.rows; i++){
		for(int j = 0; j < erodeme.cols; j++){
			process.clear();
			//loop through kernel
			for(int k = 0; k < w; k++){
				for(int l = 0; l < z; l++){
					if(!( 0 < i + k || i + k < erodeme.rows || 0 < j + l || j + l < erodeme.cols)){
						continue; //continues this one -> it would access forbidden places!
					}
					process.emplace_back(
						erodeme.at<char>(i+k, j+l) //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<fixme;
					);
				}
			}

			char res = *(std::max_element(process.begin(), process.end())); // dilation simply changes that min to this max. fun, eh?
			erodeme.at<char>(i, j) = res;
		}

	}
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
	for (int i = 0; i < binary.rows; i++) {
		for (int j = 0; j < binary.cols; j++) {
			if (!erosion(i, j)) {
				rErosion.at<uchar>(i, j) = 25;
			}
			else {
				rErosion.at<uchar>(i, j) = 230;
			}
			//namedWindow("Resultado", WINDOW_AUTOSIZE);
			//imshow("Resultado", result);
			//waitKey(1);
		}
	}

	//---------------------------------------------- DILATACION
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

	cout << dif << endl;
	cout << dif2 << endl;
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", binary);

	namedWindow("erosion", WINDOW_AUTOSIZE);
	imshow("erosion", rErosion);

	namedWindow("dilatacion", WINDOW_AUTOSIZE);
	imshow("dilatacion", rDilatacion);

	imwrite("../tcc-outputs/013_original.png", binary);
	imwrite("../tcc-outputs/013_erosion.png", rErosion);
	imwrite("../tcc-outputs/013_dilation.png", rDilatacion);

	waitKey(0);
}