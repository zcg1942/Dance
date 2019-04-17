// SubImages.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "cv.h"
#include "highgui.h"
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;
void onTrackerSlid(Mat &inputimage1, Mat &inputimage2, Mat &outputimage, int pos);
void MergeImage(Mat &inputimage1, Mat &inputimage2, Mat &Mask, Mat &outputimage);

int _tmain(int argc, _TCHAR* argv[])

{
	Mat srcimage1 = imread("graf2.bmp",1);//原始图像，120KB
	imwrite("Boximg1to2.jpg", srcimage1);//读取之后再保存，358KB，如果以灰度图0读入，126KB
	Mat srcimage2 = imread("sal.bmp",1);
	int type1 = srcimage1.type();//imread默认读取时CV8UC3，加参数0读取灰度图变成单通道
	
	
	Mat imagegray = imread("graf2save1.png");
	int typegray = srcimage2.type();
	int depth = srcimage1.depth();
	Mat dstimage;
	Mat dstimage1;
	dstimage.create(srcimage1.rows, srcimage1.cols, srcimage1.type());
	dstimage1.create(srcimage1.rows, srcimage1.cols, srcimage1.type());
	onTrackerSlid(srcimage1, srcimage2, dstimage, 0);   //将图1与图2的差分结果保存在dstimage中
	namedWindow("srcimage1", CV_WINDOW_AUTOSIZE);
	namedWindow("srcimage2", CV_WINDOW_AUTOSIZE);
	
	imshow("srcimage1", srcimage1);
	imshow("srcimage2", srcimage2);
	imshow("BoxMaskjpg", dstimage);
	imwrite("Boatsal12siftSub.bmp", dstimage);

	double psnr = PSNR(srcimage1, srcimage2);
	cout << "峰值信噪比是" << psnr;
	//cvtColor(dstimage, dstimage, COLOR_RGB2GRAY);
	//将灰度图转换为二值图像，大于150就用255替换
	//threshold(dstimage, dstimage1, 150, 255, THRESH_BINARY);   //记得加-lopencv_imgproc320
	//namedWindow("dstimage1", CV_WINDOW_AUTOSIZE);
	//imshow("dstimage1", dstimage1);
	/*Mat img(3, 4, CV_16UC4, Scalar_<uchar>(1, 2, 3, 4));

	cout << img << endl;

	cout << "dims:" << img.dims << endl;
	cout << "rows:" << img.rows << endl;
	cout << "cols:" << img.cols << endl;
	cout << "channels:" << img.channels() << endl;
	cout << "type:" << img.type() << endl;
	cout << "depth:" << img.depth() << endl;
	cout << "elemSize:" << img.elemSize() << endl;
	cout << "elemSize1:" << img.elemSize1() << endl;
	cout << "Step[0]:" << img.step[0] << endl;
	cout << "Step[1]:" << img.step[1] << endl;*/


	////复原
	//Mat ImageR = imread("Boximg1to2.jpg");
	//Mat	ImageSub = imread("BoxSub.jpg");
	//Mat ImageOut;
	//Mat ImageMask = imread("BoxMask.jpg"); //mask图像，在配准之后的图像与参考图像做减法时记录正负值
	//MergeImage(ImageR, ImageSub, ImageMask, ImageOut);
	//imshow("复原图像", ImageOut);
	//imwrite("BoxRecover.jpg", ImageOut);
	waitKey(0);
	//return 0;
	
	//原文：https ://blog.csdn.net/juliarjuliar/article/details/79812683 
	//版权声明：本文为博主原创文章，转载请附上博文链接！
	return 0;
}
void onTrackerSlid(Mat &inputimage2, Mat &inputimage1, Mat &outputimage, int pos)//差分图是原图减配准
{
	uchar *data1 = NULL;
	uchar *data2 = NULL;
	uchar *data3 = NULL;
	//uchar *data = NULL;
	int i, j;

	outputimage = inputimage2.clone();
	int rowNumber = outputimage.rows;
	int colNumber = outputimage.cols*outputimage.channels();
	int step = outputimage.step / sizeof(uchar);//每一行元素占的字节数
	//int step = outputimage.step[0];// / sizeof(uchar);//step[0]是每一行字节数
	int uchar1 = sizeof(uchar);//sizeof(uchar)等于1
	data1 = (uchar*)inputimage1.data;
	data2 = (uchar*)inputimage2.data;
	data3 = (uchar*)outputimage.data;

	for (i = 0; i < rowNumber; i++)
	{
		//data = (uchar*)outputimage.ptr<uchar>(i);   //获取第i行的首地址
		for (j = 0; j < colNumber; j++)
		{
			//if ((data2[i*step + j] - data1[i*step + j]) > pos)//pos=0,标记差值的正负，得到掩膜
			//	data3[i*step + j] = 255;//正值标记为255
			//else
			//	data3[i*step + j] = 0;
			//得到差分图的绝对值
			data3[i*step + j] = abs(data2[i*step + j] - data1[i*step + j]);
		}
	}
}
void MergeImage(Mat &inputimageR, Mat &inputimageSub, Mat &Mask, Mat &outputimage)//第一个参数是配准之后的
{
	uchar *data1 = NULL;
	uchar *data2 = NULL;
	uchar *data3 = NULL;
	uchar *dataMask=NULL;
	//uchar *data = NULL;
	int i, j;

	outputimage = inputimageR.clone();
	int rowNumber = outputimage.rows;
	int colNumber = outputimage.cols*outputimage.channels();
	int step = outputimage.step / sizeof(uchar);
	data1 = (uchar*)inputimageR.data;
	data2 = (uchar*)inputimageSub.data;
	data3 = (uchar*)outputimage.data;
	dataMask = (uchar*)Mask.data;
	int data33;

	for (i = 0; i < rowNumber; i++)
	{
		//data = (uchar*)outputimage.ptr<uchar>(i);   //获取第i行的首地址
		for (j = 0; j < colNumber; j++)
			
		{
			/*if (data1[i*step + j] != 0)
			{
				data3[i*step + j] = data1[i*step + j] ;
			}
			else{*/
			//abs（原图 - 配准） = 差值
				if (dataMask[i*step + j] == 255)//标记差值的正负，得到掩膜。原图-配准1=差值2
					data3[i*step + j] =abs( data2[i*step + j] + data1[i*step + j]); //data2是差值图像
				else
					//data3[i*step + j] =data2[i*step + j];//-(原图-配准1)=差值2
				/*data3[i*step + j] = data2[i*step + j] - data1[i*step + j];*/
				data3[i*step + j] = abs(data1[i*step + j] - data2[i*step + j]); 
				data33= data1[i*step + j] - data2[i*step + j];//若不加abs，负数的地方灰度会异常高
				//data3[i*step + j] = abs(data2[i*step + j] - data1[i*step + j]);
			//}
			
		}
	}
}



