// VERSAO SEQUENCIAL: baseline derivado diretamente do codigo de referencia.
#include "common/bench.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	configure_opencv();

	const char* filenameD = (argc > 1) ? argv[1] : "tcc-images/base.png";
	const string outdir = (argc > 2) ? argv[2] : "";
	Mat image = load_gray(filenameD);

	// Todas as matrizes sao alocadas fora da regiao cronometrada.
	Mat sobelH(image.rows, image.cols, CV_32FC1);
	Mat sobelV(image.rows, image.cols, CV_32FC1);
	Mat raiz(image.rows, image.cols, CV_32FC1);
	Mat sobelHR(image.rows, image.cols, CV_8UC1, Scalar(0));
	Mat sobelVR(image.rows, image.cols, CV_8UC1, Scalar(0));
	Mat raizR(image.rows, image.cols, CV_8UC1, Scalar(0));

	Mat prewittV(image.rows, image.cols, CV_32FC1, Scalar(0));
	Mat prewittH(image.rows, image.cols, CV_32FC1, Scalar(0));
	Mat raizP(image.rows, image.cols, CV_32FC1, Scalar(0));
	Mat prewittHR(image.rows, image.cols, CV_8UC1, Scalar(0));
	Mat prewittVR(image.rows, image.cols, CV_8UC1, Scalar(0));
	Mat raizPR(image.rows, image.cols, CV_8UC1, Scalar(0));

	Mat robertH(image.rows, image.cols, CV_32FC1, Scalar(0));
	Mat robertV(image.rows, image.cols, CV_32FC1, Scalar(0));
	Mat raizRob(image.rows, image.cols, CV_32FC1, Scalar(0));
	Mat robertHR(image.rows, image.cols, CV_8UC1, Scalar(0));
	Mat robertVR(image.rows, image.cols, CV_8UC1, Scalar(0));
	Mat raizRobR(image.rows, image.cols, CV_8UC1, Scalar(0));

	int innerMatrixIndex = 3;
	int horiz[3][3] = { { -1,0,1 },{ -2,0,2 },{ -1,0,1 } };
	int vert[3][3] = { { -1,-2,-1 },{ 0,0,0 },{ 1,2,1 } };
	int horizP[3][3] = { { -1,0,1 },{ -1,0,1 },{ -1,0,1 } };
	int vertP[3][3] = { { -1,-1,-1 },{ 0,0,0 },{ 1,1,1 } };

	int suma2 = 0;
	int promedio2 = 0;
	int suma3 = 0;
	int promedio3 = 0;
	double minH = 0;
	double maxH = 0;
	double minR = 0;
	double maxR = 0;
	double minV = 0;
	double maxV = 0;

	double t0 = now_seconds();

	// sobel horizontal
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			suma2 = 0;
			promedio2 = 0;

			int y = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				int x = 0;
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						suma2 += (image.at<uchar>(i + a, j + b) * horiz[y][x]);
					}
					x++;
				}
				y++;
			}
			sobelH.at<float>(i, j) = abs(suma2);
		}
	}

	// sobel vertical
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			suma3 = 0;
			promedio3 = 0;

			int y = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				int x = 0;
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						suma3 += (image.at<uchar>(i + a, j + b) * vert[y][x]);
					}
					x++;
				}
				y++;
			}
			sobelV.at<float>(i, j) = abs(suma3);
		}
	}

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			float xValue = (sobelH.at<float>(i, j) * sobelH.at<float>(i, j));
			float yValue = (sobelV.at<float>(i, j) * sobelV.at<float>(i, j));
			raiz.at<float>(i, j) = (sqrt(xValue + yValue));
		}
	}

	minMaxLoc(sobelH, &minH, &maxH);
	minMaxLoc(sobelV, &minV, &maxV);
	minMaxLoc(raiz, &minR, &maxR);

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			float valH = 0;
			float valV = 0;
			float valR = 0;
			if (maxH != minH)
				valH = (255 * (sobelH.at<float>(i, j) - minH)) / (maxH - minH);
			if (maxV != minV)
				valV = (255 * (sobelV.at<float>(i, j) - minV)) / (maxV - minV);
			if (maxR != minR)
				valR = (255 * (raiz.at<float>(i, j) - minR)) / (maxR - minR);
			sobelHR.at<uchar>(i, j) = abs(valH);
			sobelVR.at<uchar>(i, j) = abs(valV);
			raizR.at<uchar>(i, j) = abs(valR);
		}
	}

	//---------------------------------------------- PREWITT ---------------------------------------
	suma2 = 0;
	promedio2 = 0;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			suma2 = 0;
			promedio2 = 0;

			int y = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				int x = 0;
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						suma2 += (image.at<uchar>(i + a, j + b) * horizP[y][x]);
					}
					x++;
				}
				y++;
			}
			prewittH.at<float>(i, j) = abs(suma2);
		}
	}

	suma3 = 0;
	promedio3 = 0;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			suma3 = 0;
			promedio3 = 0;

			int y = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				int x = 0;
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						suma3 += (image.at<uchar>(i + a, j + b) * vertP[y][x]);
					}
					x++;
				}
				y++;
			}
			prewittV.at<float>(i, j) = abs(suma3);
		}
	}

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			float xValue = (prewittH.at<float>(i, j) * prewittH.at<float>(i, j));
			float yValue = (prewittV.at<float>(i, j) * prewittV.at<float>(i, j));
			raizP.at<float>(i, j) = (sqrt(xValue + yValue));
		}
	}

	minH = 0;
	maxH = 0;
	minR = 0;
	maxR = 0;
	minV = 0;
	maxV = 0;
	minMaxLoc(prewittH, &minH, &maxH);
	minMaxLoc(prewittV, &minV, &maxV);
	minMaxLoc(raizP, &minR, &maxR);

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			float valH = 0;
			float valV = 0;
			float valR = 0;
			if (maxH != minH)
				valH = (255 * (prewittH.at<float>(i, j) - minH)) / (maxH - minH);
			if (maxV != minV)
				valV = (255 * (prewittV.at<float>(i, j) - minV)) / (maxV - minV);
			if (maxR != minR)
				valR = (255 * (raizP.at<float>(i, j) - minR)) / (maxR - minR);
			prewittHR.at<uchar>(i, j) = abs(valH);
			prewittVR.at<uchar>(i, j) = abs(valV);
			raizPR.at<uchar>(i, j) = abs(valR);
		}
	}

	//------------------------------------ ROBERTS -----------------------------------
	for (int i = 1; i < image.rows - 1; i++)
	{
		for (int j = 1; j < image.cols - 1; j++)
		{
			float resH = 0;
			float resV = 0;
			resH = -1 * (image.at<uchar>(i, j)) + (image.at<uchar>(i + 1, j + 1));
			resV = -1 * (image.at<uchar>(i, j + 1)) + (image.at<uchar>(i + 1, j));
			robertH.at<float>(i, j) = abs(resH);
			robertV.at<float>(i, j) = abs(resV);
		}
	}

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			float xValue = (robertH.at<float>(i, j) * robertH.at<float>(i, j));
			float yValue = (robertV.at<float>(i, j) * robertV.at<float>(i, j));
			raizRob.at<float>(i, j) = (sqrt(xValue + yValue));
		}
	}

	minH = 0;
	maxH = 0;
	minR = 0;
	maxR = 0;
	minV = 0;
	maxV = 0;
	minMaxLoc(robertH, &minH, &maxH);
	minMaxLoc(robertV, &minV, &maxV);
	minMaxLoc(raizRob, &minR, &maxR);

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			float valH = 0;
			float valV = 0;
			float valR = 0;
			if (maxH != minH)
				valH = (255 * (robertH.at<float>(i, j) - minH)) / (maxH - minH);
			if (maxV != minV)
				valV = (255 * (robertV.at<float>(i, j) - minV)) / (maxV - minV);
			if (maxR != minR)
				valR = (255 * (raizRob.at<float>(i, j) - minR)) / (maxR - minR);
			robertHR.at<uchar>(i, j) = abs(valH);
			robertVR.at<uchar>(i, j) = abs(valV);
			raizRobR.at<uchar>(i, j) = abs(valR);
		}
	}

	double t1 = now_seconds();
	report("sobel-prewitt-roberts", t1 - t0);

	dump(outdir, "sobel_h", sobelHR);
	dump(outdir, "sobel_v", sobelVR);
	dump(outdir, "sobel_mag", raizR);
	dump(outdir, "prewitt_h", prewittHR);
	dump(outdir, "prewitt_v", prewittVR);
	dump(outdir, "prewitt_mag", raizPR);
	dump(outdir, "roberts_h", robertHR);
	dump(outdir, "roberts_v", robertVR);
	dump(outdir, "roberts_mag", raizRobR);

	return 0;
}
