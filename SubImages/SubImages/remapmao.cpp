#include<stdafx.h>
#include <opencv2/opencv.hpp>
using namespace cv;
int main()
{
	Mat srcImage = imread("D:\\b.jpg");
	Mat dstImage, map_x, map_y;
	imshow("", srcImage);
	//������ԭʼͼ��һ����Ч��ͼ��x��ӳ��ͼ��y��ӳ��ͼ
	dstImage.create(srcImage.size(), srcImage.type());
	map_x.create(srcImage.size(), CV_32FC1);
	map_y.create(srcImage.size(), CV_32FC1);
	//˫��ѭ��������ÿһ�����ص㣬�ı�map_x��map_y��ֵ
	for (int j = 0; j<srcImage.rows; j++)
	{
		for (int i = 0; i<srcImage.cols; i++)
		{
			//�ı�map_x��map_y��ֵ
			map_x.at<float>(j, i) = static_cast<float>(i );//y������ǰimage.at(x1, x2)=image.at(Point(x2, x1))
			//map_x.at<float>(j, i) = static_cast<float>(srcImage.rows - i);
			map_y.at<float>(j, i) = static_cast<float>(srcImage.rows - j);
		}
	}
	//������ӳ�����
	remap(srcImage, dstImage, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 0, 0));
	//��ʾЧ��ͼ
	imshow("Ч��ͼ", dstImage);
	waitKey(0);
	return 0;

}