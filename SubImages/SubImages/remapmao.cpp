#include<stdafx.h>
#include <opencv2/opencv.hpp>
using namespace cv;
int main()
{
	Mat srcImage = imread("D:\\b.jpg");
	Mat dstImage, map_x, map_y;
	imshow("", srcImage);
	//创建和原始图像一样的效果图，x重映射图，y重映射图
	dstImage.create(srcImage.size(), srcImage.type());
	map_x.create(srcImage.size(), CV_32FC1);
	map_y.create(srcImage.size(), CV_32FC1);
	//双层循环，遍历每一个像素点，改变map_x和map_y的值
	for (int j = 0; j<srcImage.rows; j++)
	{
		for (int i = 0; i<srcImage.cols; i++)
		{
			//改变map_x和map_y的值
			map_x.at<float>(j, i) = static_cast<float>(i );//y坐标在前image.at(x1, x2)=image.at(Point(x2, x1))
			//map_x.at<float>(j, i) = static_cast<float>(srcImage.rows - i);
			map_y.at<float>(j, i) = static_cast<float>(srcImage.rows - j);
		}
	}
	//进行重映射操作
	remap(srcImage, dstImage, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 0, 0));
	//显示效果图
	imshow("效果图", dstImage);
	waitKey(0);
	return 0;

}