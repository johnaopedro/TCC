void threshold_ref(State &s) {
auto &image2=s.thr[0], &image3=s.thr[1], &image4=s.thr[2], &image5=s.thr[3], &image6=s.thr[4];


	for (int i = 0; i < image2.rows; i++) {
		for (int j = 0; j < image2.cols; j++) {
			if (image2.at<uchar>(i, j) > 80) {
				image2.at<uchar>(i, j) = 255;
			}
			else {
				image2.at<uchar>(i, j) = 0;
			}
		}
	}

	for (int i = 0; i < image2.rows; i++) {
		for (int j = 0; j < image2.cols; j++) {
			if (image3.at<uchar>(i, j) > 80) {
				image3.at<uchar>(i, j) = 0;
			}
			else {
				image3.at<uchar>(i, j) = 255;
			}
		}
	}

	for (int i = 0; i < image4.rows; i++) {
		for (int j = 0; j < image4.cols; j++) {
			if (image4.at<uchar>(i, j) > 80) {
				image4.at<uchar>(i, j) = 80;
			}
			else {

			}
		}
	}

	for (int i = 0; i < image5.rows; i++) {
		for (int j = 0; j < image5.cols; j++) {
			if (image5.at<uchar>(i, j) > 80) {

			}
			else {
				image5.at<uchar>(i, j) = 0;
			}
		}
	}

	for (int i = 0; i < image6.rows; i++) {
		for (int j = 0; j < image6.cols; j++) {
			if (image6.at<uchar>(i, j) > 80) {
				image6.at<uchar>(i, j) = 0;
			}
			else {

			}
		}
	}

	
}
void equalization_ref(State &s) {
auto &image=s.image, &myMat1=s.out[0];
int arr[256]={}; float arr2[256]={},arr3[256]={};


	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int index = (int)(image.at<uchar>(i, j));
			arr[index]++;
		}
	}

	//-- PMF
	float total = image.cols*image.rows;
	for (int i = 0; i < 256; i++)
	{
		arr2[i] = float(arr[i]) / total;
	}
	arr3[0] = arr2[0];

	//-- CDF
	for (int i = 1; i < 256; i++)
	{
		arr3[i] = arr2[i] + arr3[i - 1];
	}

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			myMat1.at<uchar>(i, j) = floor((256 - 1)*arr3[image.at<uchar>(i, j)]);

		}
	}

	
}
void sobel_ref(State &s) {
auto &image = s.image;
auto &sobelH=s.grad[0], &sobelV=s.grad[1], &raiz=s.grad[2];
auto &sobelHR=s.out[0], &sobelVR=s.out[1], &raizR=s.out[2];
auto &prewittH=s.grad[0], &prewittV=s.grad[1], &raizP=s.grad[2];
auto &prewittHR=s.out[0], &prewittVR=s.out[1], &raizPR=s.out[2];
auto &robertH=s.grad[0], &robertV=s.grad[1], &raizRob=s.grad[2];
auto &robertHR=s.out[0], &robertVR=s.out[1], &raizRobR=s.out[2];
int innerMatrixIndex=3;
int horiz[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
int vert[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
int horizP[3][3]={{-1,0,1},{-1,0,1},{-1,0,1}};
int vertP[3][3]={{-1,-1,-1},{0,0,0},{1,1,1}};
int suma2=0, suma3=0, promedio2=0, promedio3=0;
double minH=0, maxH=0, minV=0, maxV=0, minR=0, maxR=0;


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


}
void prewitt_ref(State &s) {
auto &image = s.image;
auto &sobelH=s.grad[0], &sobelV=s.grad[1], &raiz=s.grad[2];
auto &sobelHR=s.out[0], &sobelVR=s.out[1], &raizR=s.out[2];
auto &prewittH=s.grad[0], &prewittV=s.grad[1], &raizP=s.grad[2];
auto &prewittHR=s.out[0], &prewittVR=s.out[1], &raizPR=s.out[2];
auto &robertH=s.grad[0], &robertV=s.grad[1], &raizRob=s.grad[2];
auto &robertHR=s.out[0], &robertVR=s.out[1], &raizRobR=s.out[2];
int innerMatrixIndex=3;
int horiz[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
int vert[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
int horizP[3][3]={{-1,0,1},{-1,0,1},{-1,0,1}};
int vertP[3][3]={{-1,-1,-1},{0,0,0},{1,1,1}};
int suma2=0, suma3=0, promedio2=0, promedio3=0;
double minH=0, maxH=0, minV=0, maxV=0, minR=0, maxR=0;
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


}
void roberts_ref(State &s) {
auto &image = s.image;
auto &sobelH=s.grad[0], &sobelV=s.grad[1], &raiz=s.grad[2];
auto &sobelHR=s.out[0], &sobelVR=s.out[1], &raizR=s.out[2];
auto &prewittH=s.grad[0], &prewittV=s.grad[1], &raizP=s.grad[2];
auto &prewittHR=s.out[0], &prewittVR=s.out[1], &raizPR=s.out[2];
auto &robertH=s.grad[0], &robertV=s.grad[1], &raizRob=s.grad[2];
auto &robertHR=s.out[0], &robertVR=s.out[1], &raizRobR=s.out[2];
int innerMatrixIndex=3;
int horiz[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
int vert[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
int horizP[3][3]={{-1,0,1},{-1,0,1},{-1,0,1}};
int vertP[3][3]={{-1,-1,-1},{0,0,0},{1,1,1}};
int suma2=0, suma3=0, promedio2=0, promedio3=0;
double minH=0, maxH=0, minV=0, maxV=0, minR=0, maxR=0;
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

	
}
