#include <stdafx.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
void onTrackerSlid(Mat &inputimage1, Mat &inputimage2, Mat &outputimage, int pos);

int main()
{
	Mat srcimage1 = imread("b.jpg",1);
	Mat srcimage2 = imread("bb.jpg",1);
	Mat dstimage;
	Mat dstimage1;
	dstimage.create(srcimage1.rows, srcimage1.cols, srcimage1.type());
	dstimage1.create(srcimage1.rows, srcimage1.cols, srcimage1.type());
	onTrackerSlid(srcimage1, srcimage2, dstimage, 100);   //将图1与图2的差分结果保存在dstimage中
	namedWindow("srcimage1", CV_WINDOW_AUTOSIZE);
	namedWindow("srcimage2", CV_WINDOW_AUTOSIZE);
	namedWindow("dstimage", CV_WINDOW_AUTOSIZE);
	imshow("srcimage1", srcimage1);
	imshow("srcimage2", srcimage2);
	imshow("dstimage", dstimage);
	cvtColor(dstimage, dstimage, COLOR_RGB2GRAY);
	threshold(dstimage, dstimage1, 150, 255, THRESH_BINARY);   //记得加-lopencv_imgproc320
	namedWindow("dstimage1", CV_WINDOW_AUTOSIZE);
	imshow("dstimage1", dstimage1);

	waitKey(0);
	return 0;
}

void onTrackerSlid(Mat &inputimage1, Mat &inputimage2, Mat &outputimage, int pos)
{
	uchar *data1 = NULL;
	uchar *data2 = NULL;
	uchar *data3 = NULL;
	//uchar *data = NULL;
	int i, j;

	outputimage = inputimage1.clone();
	int rowNumber = outputimage.rows;
	int colNumber = outputimage.cols*outputimage.channels();
	int step = outputimage.step / sizeof(uchar);
	data1 = (uchar*)inputimage1.data;
	data2 = (uchar*)inputimage2.data;
	data3 = (uchar*)outputimage.data;

	for (i = 0; i < rowNumber; i++)
	{
		//data = (uchar*)outputimage.ptr<uchar>(i);   //获取第i行的首地址
		for (j = 0; j < colNumber; j++)
		{
			if (abs(data2[i*step + j] - data1[i*step + j]) > pos)
				data3[i*step + j] = 255;
			else
				data3[i*step + j] = 0;
		}
	}
}
