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

//#define DEBUG

using namespace std;
using namespace cv;

// con que uno de los bracitos del elemento este fuera de la mancha, ya no se pinta el origen
void grayscale_erosion(Mat &erodeme, int w, int z){
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

			char res = *(std::min_element(process.begin(), process.end())); // min -> it -> *it -> int -> clamp it to an int; go crazy!
			erodeme.at<char>(i, j) = res;
		}
	}
}

// con que uno de los bracitos del elemento este dentro de la mancha, se pinta ese origen
void grayscale_dilate(Mat &dilateme, int w, int z){
	//loop through image
	std::vector<int> process;
	for(int i = 0; i < dilateme.rows; i++){
		for(int j = 0; j < dilateme.cols; j++){
			process.clear();
			//loop through kernel
			for(int k = 0; k < w; k++){
				for(int l = 0; l < z; l++){
					if(!( 0 < i + k || i + k < dilateme.rows || 0 < j + l || j + l < dilateme.cols)){
						continue; //continues this one -> it would access forbidden places!
					}
					process.emplace_back(
						dilateme.at<char>(i+k, j+l) 
					);
				}
			}

			char res = *(std::max_element(process.begin(), process.end())); // dilation simply changes that min to this max. fun, eh?
			dilateme.at<char>(i, j) = res;
		}

	}
}

int main() {

	Mat binary = imread("../tcc-images/sudoku.png", IMREAD_GRAYSCALE);
	//Mat rErosion(binary.rows, binary.cols, CV_8UC1, Scalar(0));
	Mat rErosion = binary.clone();
	//Mat rDilatacion(binary.rows, binary.cols, CV_8UC1, Scalar(0));
	Mat rDilatacion = binary.clone();
	int mask[3][3] = { {0,1,0},{1,1,1},{0,1,0} };
	int innerMtx = 3;
	//---------------------------------------------- EROSION
	
	/*
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
	*/

	grayscale_erosion(rErosion, 3, 3);
	grayscale_dilate(rDilatacion, 3, 3);

	

	//---------------------------------------------- DILATACION
	
	/*
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
	*/

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

	cout << dif  << endl;
	cout << dif2 << endl;
	
	#ifdef DEBUG
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", binary);

	namedWindow("erosion", WINDOW_AUTOSIZE);
	imshow("erosion", rErosion);

	namedWindow("dilatacion", WINDOW_AUTOSIZE);
	imshow("dilatacion", rDilatacion);
	waitKey(0);
	#endif

	imwrite("../tcc-outputs/013_goriginal.png", binary     );
	imwrite("../tcc-outputs/013_gerosion.png" , rErosion   );
	imwrite("../tcc-outputs/013_gdilation.png", rDilatacion);

}