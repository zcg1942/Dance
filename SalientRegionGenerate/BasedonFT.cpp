#include<stdio.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<time.h>
using namespace std;
using namespace cv;
Mat SalientRegionDetectionBasedonFT(Mat &src,Mat &Sal);
Mat SegToBin(Mat &src);
void FloodFillProcess(Mat &Sal1);
#define LODIFF (6)//宏定义最好用大写字母，和正规代码分开，并且不能有分号
#define UPDIFF (6)
//http://ivrlwww.epfl.ch/~achanta/SalientRegionDetection/SalientRegionDetection.html
//http://ivrlwww.epfl.ch/supplementary_material/RK_CVPR09/
int main(void)
{
	clock_t start, finish,salstart;
	double totaltime,salTime,MeanshiftTime;
	//读取图像
	start = clock();
	Mat srcImg = imread("bike1.bmp");//https://blog.csdn.net/wangyaninglm/article/details/44020489测试集
	/*Mat srcImg2 = imread("bike3.bmp");
	warpPerspective()*/
	//求出原图均值和方差
	Mat tmp_m, tmp_sd;
	double m = 0, sd = 0;
	double mb = 0, mg = 0, mr = 0;

	mb = mean(srcImg)[0];
	mg = mean(srcImg)[1];//求三个通道的均值
	mr = mean(srcImg)[2];
	cout << "Mean: " << mb <<" "<<mg<<" "<<mr<<endl;

	Mat sal = imread("2grafsal.png");
	meanStdDev(sal, tmp_m, tmp_sd);
	m = tmp_m.at<double>(0, 0);
	sd = tmp_sd.at<double>(0, 0);
	cout << "Mean: " << m << " , StdDev: " << sd << endl;

	//Mat ground((srcImg.rows) * 2, (srcImg.cols) * 2, CV_8UC3, Scalar::all(m));
	////Mat ground((srcImg.rows) * 2, (srcImg.cols) * 2, CV_8UC3, Scalar(mb,mg,mr));//彩色填充 因为FT方法要利用亮度和颜色信息 所以这里通道数要改为3 scalar也填三个通道
	//Mat ROI = ground(Rect(0, 0, srcImg.cols, srcImg.rows));
	////Mat mask=
	//srcImg.copyTo(ROI);//copy当然只能复制到相同大小的图像中，只不过第一个参数可以是另一幅图的指定的ROI
	////Mat dstImg = imread("DesertSafari.jpg");
	//Mat res1,res2;// = srcImg.clone();
	if (srcImg.empty())
	{
		cout << "图像没有读取成功" << endl;
		getchar();
		return 0;
	}
	imshow("原图1", srcImg);
	imwrite("imagesrc.bmp", srcImg);
	
	//waitKey(0);
	Mat Sal1 = Mat::zeros(srcImg.size(), CV_8UC1);//Mat初始化的方法要会
	//Mat Sal2 = Mat::zeros(dstImg.size(), CV_8UC1);

	Sal1 = SalientRegionDetectionBasedonFT(srcImg, Sal1);
	//Sal2=SalientRegionDetectionBasedonFT(dstImg,Sal2);

	imshow("salmap1", Sal1);
	imwrite("salImage.bmp", Sal1);

	//maxPyrLevel = 3;//金字塔最大层数 
	////https://blog.csdn.net/gdfsg/article/details/50975422
	
	//MeanshiftTime = (double)(finish - start) / CLOCKS_PER_SEC;


	waitKey(0);
	
}

//https://blog.csdn.net/cai13160674275/article/details/72991049
//http://ivrlwww.epfl.ch/supplementary_material/RK_CVPR09/
Mat SalientRegionDetectionBasedonFT(Mat &src,Mat &Sal){
	Mat Lab,BGR;
	if (src.type()==16)
	cvtColor(src, Lab, CV_BGR2Lab);//第一个参数是三通道的，而在显著性提取之后是单通道
	else
	{
		cvtColor(src, BGR, CV_GRAY2BGR);//如果是灰度图，先转换为三色图
		cvtColor(BGR, Lab, CV_BGR2Lab);
	}


	int row = src.rows, col = src.cols;

	//int Sal_org[row][col];
	int **Sal_org;//https://zhidao.baidu.com/question/462803761.html 二级指针实现数组大小用变量定义
	Sal_org = new int*[row]; 
	for (int i = 0; i < row; i++)
		Sal_org[i] = new int[col];
	//memset(Sal_org, 0, sizeof(Sal_org));

	Point3_<uchar>* p;

	int MeanL = 0, Meana = 0, Meanb = 0;
	for (int i = 0; i<row; i++){
		for (int j = 0; j<col; j++){
			p = Lab.ptr<Point3_<uchar> >(i, j);
			MeanL += p->x;
			Meana += p->y;
			Meanb += p->z;
		}
	}
	MeanL /= (row*col);//平均值
	Meana /= (row*col);
	Meanb /= (row*col);

	GaussianBlur(Lab, Lab, Size(3, 3), 0, 0);

	

	int val;

	int max_v = 0;
	int min_v = 1 << 28;//???

	for (int i = 0; i<row; i++){
		for (int j = 0; j<col; j++){
			p = Lab.ptr<Point3_<uchar> >(i, j);
			val = sqrt((MeanL - p->x)*(MeanL - p->x) + (p->y - Meana)*(p->y - Meana) + (p->z - Meanb)*(p->z - Meanb));//lab空间的均值减去当前像素值 计算每一个像素的显著性
			Sal_org[i][j] = val;
			max_v = max(max_v, val);//返回两个数之间较大的
			min_v = min(min_v, val);
		}
	}

	cout << "\t\t\t" << "像素显著性最值:" << max_v << " " << min_v << endl;//输出最大值和最小值
	int X, Y, Mean_sal = 0;
	for (Y = 0; Y < row; Y++)
	{
		for (X = 0; X < col; X++)
		{
			Sal.at<uchar>(Y, X) = (Sal_org[Y][X] - min_v) * 255 / (max_v - min_v);        //    计算全图每个像素的显著性 归一化到0~255的灰度值
			//Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性
			//Mean_sal += Sal.at<uchar>(Y, X);
		
		}
	}
	return Sal;
	//imshow("sal", Sal);
	//waitKey(0);
}
Mat SegToBin(Mat &src)
{
	int row = src.rows, col = src.cols;
	int Mean_sal=0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			Mean_sal += src.at<uchar>(i, j);
		}
	}
	Mean_sal = Mean_sal / (row*col);

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (src.at<uchar>(i, j)>Mean_sal)//阈值是均值的2倍
				src.at<uchar>(i, j) = 255;
			else src.at<uchar>(i, j) = 0;
		}
	}
	return src;

}


