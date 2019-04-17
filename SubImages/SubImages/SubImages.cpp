// SubImages.cpp : �������̨Ӧ�ó������ڵ㡣
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
	Mat srcimage1 = imread("graf2.bmp",1);//ԭʼͼ��120KB
	imwrite("Boximg1to2.jpg", srcimage1);//��ȡ֮���ٱ��棬358KB������ԻҶ�ͼ0���룬126KB
	Mat srcimage2 = imread("sal.bmp",1);
	int type1 = srcimage1.type();//imreadĬ�϶�ȡʱCV8UC3���Ӳ���0��ȡ�Ҷ�ͼ��ɵ�ͨ��
	
	
	Mat imagegray = imread("graf2save1.png");
	int typegray = srcimage2.type();
	int depth = srcimage1.depth();
	Mat dstimage;
	Mat dstimage1;
	dstimage.create(srcimage1.rows, srcimage1.cols, srcimage1.type());
	dstimage1.create(srcimage1.rows, srcimage1.cols, srcimage1.type());
	onTrackerSlid(srcimage1, srcimage2, dstimage, 0);   //��ͼ1��ͼ2�Ĳ�ֽ��������dstimage��
	namedWindow("srcimage1", CV_WINDOW_AUTOSIZE);
	namedWindow("srcimage2", CV_WINDOW_AUTOSIZE);
	
	imshow("srcimage1", srcimage1);
	imshow("srcimage2", srcimage2);
	imshow("BoxMaskjpg", dstimage);
	imwrite("Boatsal12siftSub.bmp", dstimage);

	double psnr = PSNR(srcimage1, srcimage2);
	cout << "��ֵ�������" << psnr;
	//cvtColor(dstimage, dstimage, COLOR_RGB2GRAY);
	//���Ҷ�ͼת��Ϊ��ֵͼ�񣬴���150����255�滻
	//threshold(dstimage, dstimage1, 150, 255, THRESH_BINARY);   //�ǵü�-lopencv_imgproc320
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


	////��ԭ
	//Mat ImageR = imread("Boximg1to2.jpg");
	//Mat	ImageSub = imread("BoxSub.jpg");
	//Mat ImageOut;
	//Mat ImageMask = imread("BoxMask.jpg"); //maskͼ������׼֮���ͼ����ο�ͼ��������ʱ��¼����ֵ
	//MergeImage(ImageR, ImageSub, ImageMask, ImageOut);
	//imshow("��ԭͼ��", ImageOut);
	//imwrite("BoxRecover.jpg", ImageOut);
	waitKey(0);
	//return 0;
	
	//ԭ�ģ�https ://blog.csdn.net/juliarjuliar/article/details/79812683 
	//��Ȩ����������Ϊ����ԭ�����£�ת���븽�ϲ������ӣ�
	return 0;
}
void onTrackerSlid(Mat &inputimage2, Mat &inputimage1, Mat &outputimage, int pos)//���ͼ��ԭͼ����׼
{
	uchar *data1 = NULL;
	uchar *data2 = NULL;
	uchar *data3 = NULL;
	//uchar *data = NULL;
	int i, j;

	outputimage = inputimage2.clone();
	int rowNumber = outputimage.rows;
	int colNumber = outputimage.cols*outputimage.channels();
	int step = outputimage.step / sizeof(uchar);//ÿһ��Ԫ��ռ���ֽ���
	//int step = outputimage.step[0];// / sizeof(uchar);//step[0]��ÿһ���ֽ���
	int uchar1 = sizeof(uchar);//sizeof(uchar)����1
	data1 = (uchar*)inputimage1.data;
	data2 = (uchar*)inputimage2.data;
	data3 = (uchar*)outputimage.data;

	for (i = 0; i < rowNumber; i++)
	{
		//data = (uchar*)outputimage.ptr<uchar>(i);   //��ȡ��i�е��׵�ַ
		for (j = 0; j < colNumber; j++)
		{
			//if ((data2[i*step + j] - data1[i*step + j]) > pos)//pos=0,��ǲ�ֵ���������õ���Ĥ
			//	data3[i*step + j] = 255;//��ֵ���Ϊ255
			//else
			//	data3[i*step + j] = 0;
			//�õ����ͼ�ľ���ֵ
			data3[i*step + j] = abs(data2[i*step + j] - data1[i*step + j]);
		}
	}
}
void MergeImage(Mat &inputimageR, Mat &inputimageSub, Mat &Mask, Mat &outputimage)//��һ����������׼֮���
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
		//data = (uchar*)outputimage.ptr<uchar>(i);   //��ȡ��i�е��׵�ַ
		for (j = 0; j < colNumber; j++)
			
		{
			/*if (data1[i*step + j] != 0)
			{
				data3[i*step + j] = data1[i*step + j] ;
			}
			else{*/
			//abs��ԭͼ - ��׼�� = ��ֵ
				if (dataMask[i*step + j] == 255)//��ǲ�ֵ���������õ���Ĥ��ԭͼ-��׼1=��ֵ2
					data3[i*step + j] =abs( data2[i*step + j] + data1[i*step + j]); //data2�ǲ�ֵͼ��
				else
					//data3[i*step + j] =data2[i*step + j];//-(ԭͼ-��׼1)=��ֵ2
				/*data3[i*step + j] = data2[i*step + j] - data1[i*step + j];*/
				data3[i*step + j] = abs(data1[i*step + j] - data2[i*step + j]); 
				data33= data1[i*step + j] - data2[i*step + j];//������abs�������ĵط��ҶȻ��쳣��
				//data3[i*step + j] = abs(data2[i*step + j] - data1[i*step + j]);
			//}
			
		}
	}
}



