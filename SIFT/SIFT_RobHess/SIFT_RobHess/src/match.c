/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
*/

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> //opencv2����c++�Ľӿ�
//#include <cxcore.h>
//#include <cvaux.h>

#include <math.h> 
#include <getpsnr.h>
#include<time.h>
//#include<iostream>
//using namespace std;c++�в��б�׼�����cout
//using namespace cv;c����û�������ռ䣿

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

int display = 1;



int main( int argc, char** argv )
{
	//��ʱ
	clock_t start, finish,detect;
	double totaltime,detectTime;
	start = clock();
  IplImage* img1, * img2, * stacked;
  struct feature* feat1, * feat2, * feat,* feat3;
  struct feature** nbrs;
  struct kd_node* kd_root;
  CvPoint pt1, pt2;
  double d0, d1;
  int n1, n2,n3, k, i, m = 0;
  
  if( argc != 3 )
    fatal_error( "usage: %s <img1> <img2>", argv[0] );
  //����ͼ��
  img1 = cvLoadImage( argv[1], 1 );
  if( ! img1 )
    fatal_error( "unable to load image from %s", argv[1] );
  img2 = cvLoadImage( argv[2], 1 );
  if( ! img2 )
    fatal_error( "unable to load image from %s", argv[2] );
  stacked = stack_imgs( img1, img2 );
  //���������
  fprintf( stderr, "Finding features in %s...\n", argv[1] );
  IplImage* down1 = cvCreateImage(cvSize(img1->width / 2, img1->height / 2), img1->depth, img1->nChannels);
  cvPyrDown(img1, down1, 7);//filter=7 Ŀǰֻ֧��CV_GAUSSIAN_5x5
  n1 = sift_features( img1, &feat1 );
  printf("\nimg1��������%d", n1);
  n3 = sift_features(down1, &feat3);//�²����ļ��������
  fprintf( stderr, "\nFinding features in %s...\n", argv[2] );
  n2 = sift_features(img2, &feat2);
  printf("\nimg2��������%d", n2);
  detect = clock();

  //����siftfeat�еļ��仭������
  //if (display)
  //{
	
	 // draw_features(img1, feat1, n1);
	 // draw_features(img2, feat2, n2);
	 // display_big_img(img1, argv[1]);
	 // display_big_img(img2, argv[2]);
	 // cvShowImage("downsample", down1);
	 //
	 // fprintf(stderr, "Found %d features in img1.\n", n1);
	 // fprintf(stderr, "Found %d features in img2.\n", n2);
	 // //cvWaitKey(0);
  //}
  fprintf( stderr, "Building kd tree...\n" );
  kd_root = kdtree_build( feat2, n2 );//ֻ��ͼ2����kd��
  for( i = 0; i < n1; i++ )//��ͼ1���������������ͼ2��kd������knn
    {
      feat = feat1 + i;
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );//��������2
      if( k == 2 )//����������nbars
	{
	  d0 = descr_dist_sq( feat, nbrs[0] );
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR )//��ֵΪ0.49
	    {
	      pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	      pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );//�����С�ڴν��ڵ�0.49ʱ�����������Ϊ��Ӧ��
		  //c����û��vector������������������㶼�Ž�ȥ
	      pt2.y += img1->height;
		  //pt2.x += img1->width;
	      cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
	      m++;
		  feat1[i].fwd_match = nbrs[0];  //c����û��vector������������������㶼�Ž�ȥ������������ṹ����
	    }
	}
      free( nbrs );
    }

  fprintf( stderr, "Found %d total matches\n", m );
  display_big_img( stacked, "Matches" );
  const char* pathmatch;
  pathmatch = "E:\\000.png";
  cvSaveImage(pathmatch, stacked, 0);
  //cvWaitKey( 0 );
 

  /* 
     UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
     
     Note that this line above:
     
     feat1[i].fwd_match = nbrs[0];
     
     is important for the RANSAC function to work.
  */
  
  {
    CvMat* H1,*H2;
    IplImage* xformed2,* xformed1;
	//double xpsnr;
	CvScalar scalar1,scalar2;
	//���¼���һ�飬�任��ϣ�����������ͷӰ�������
	/*img1 = cvLoadImage(argv[1], 1);
	img2 = cvLoadImage(argv[2], 1);*/
	/*char filename1 = "E:\Local Repositories\SIFT_Snow\SIFT_RobHess\SIFT_RobHess\sal1.png";
	char filename2 = "E:\Local Repositories\SIFT_Snow\SIFT_RobHess\SIFT_RobHess\sal2.png";*/
	img1 = cvLoadImage("boat1.bmp", 1);
	img2 = cvLoadImage("boat2.bmp", 1);
	if (!img1)
		fatal_error("unable to load image from %s", argv[1]);
	img2 = cvLoadImage(argv[2], 1);
	if (!img2)
		fatal_error("unable to load image from %s", argv[2]);


    H1 = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//��任������4��ƥ���
		      homog_xfer_err, 3.0, NULL, NULL );//����������Ϊ0.01
	H2 = ransac_xform(feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,//��任������4��ƥ���
		homog_xfer_err, 3.0, NULL, NULL);//����������Ϊ0.01
    if( H1 )
      {
		//��ӡ����
		printf("\n͸�ӱ任����HΪ��");
		for (int i = 0; i < H1->rows; i++)//��  
		{
			for (int j = 0; j < H1->cols; j++)
			{
				//double H
				if (j % 3 == 0) printf("\n");
				printf("%9.3f", (float)cvGetReal2D(H1, i, j));
				//printf(" %f\t,H1->data.fl[i]");
			}
		}
		float h11 = (float)cvGetReal2D(H1, 0, 0);
		float h12 = (float)cvGetReal2D(H1, 0, 1);
		float h13 = (float)cvGetReal2D(H1, 0, 2);
		float h21 = (float)cvGetReal2D(H1, 1, 0);
		float h22 = (float)cvGetReal2D(H1, 1, 1);
		float h23 = (float)cvGetReal2D(H1, 1, 2);
		float h31 = (float)cvGetReal2D(H1, 2, 0);
		float h32 = (float)cvGetReal2D(H1, 2, 1);
		float h33 = (float)cvGetReal2D(H1, 2, 2);
		double x1, y1,  s,distance=0.0,sum=0.0;
		double x2m, y2m,x2,y2;
		for (i = 0; i < n1; i++){

			 x1=feat1[i].x;
			 y1=feat1[i].y;
			 s = h31*x1 + h32*y1 + h33;
			x2m = (h11*x1 + h12*y1 + h13) / s;
			y2m = (h21*x1 + h22*y1 + h23) / s;
			/* s = h13*x1 + h23*y1 + h33;
			 x2m = (h11*x1 + h21*y1 + h31) / s;͸�ӱ任�ľ���������⣿��
			 y2m = (h12*x1 + h22*y1 + h32) / s;*/
			if (!(feat1[i].fwd_match == NULL))
			{
				x2 = feat1[i].fwd_match->x;//Ҳ������Щ������1û�з�������ڱ��ʵĶ�Ӧ��
				y2 = feat1[i].fwd_match->y;
				distance = powf(x2m - x2, 2) + powf(y2m - y2, 2);//powf��ƽ����pow
				sum = sum + distance;
			}
			
			//sum = sum + distance;
		}
		double RMSE = sqrt(sum / n1);
		printf("\n");
		printf("RMSE=%f", RMSE);


		//cout << "cout���" << H1 << endl; cout��c++�еı�׼�����������������c����
	xformed1 = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );//ͨ����Ϊ1,��Ϊrob��ͼ���ǵ�ͨ���ģ����Ǹĳ���ͨ���ľͳ����� �����Ǽ�ͨ����
	cvWarpPerspective( img1, xformed1, H1, 
			   CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//���Բ�ֵ
			   cvScalarAll( 255 ) );//255��û����䲿�����Ϊ��ɫ
	const char* path;
	path = "E:\\siftboat12sal.bmp";
	cvSaveImage(path, xformed1, 0);
	//xformed2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);//ͨ����Ϊ3
	//cvWarpPerspective(img2, xformed2, H2,
	//	CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,//���Բ�ֵ
	//	cvScalarAll(0));//255��û����䲿�����Ϊ��ɫ
	//xformed1 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);//ͨ����Ϊ3
	//cvCopy(xformed, xformed1,NULL);
	//��һ����ͼ��û�и�ֵ���Ǻ�ɫ��
	//for (int i = 0; i<xformed1->height; i++)
	//{
	//	for (int j = 0; j<xformed1->width; j++)
	//	{
	//		scalar1 = cvGet2D(xformed1, i, j);
	//		scalar2 = cvGet2D(img2, i, j);
	//		if (scalar1.val[0] == 0)
	//		{
	//			for (int c = 0; c < 3; c++)
	//			{
	//				//xformed(i,j)
	//				scalar1.val[c] = scalar2.val[c];
	//			}
	//		}

	//	}
	//}
	
	psnr(img2, xformed1);
	
	cvNamedWindow( "Xformed1", 1 );
	cvShowImage( "Xformed1", xformed1);
	//cvShowImage("Xformed2", xformed2);
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	detectTime = (double)(detect - start) / CLOCKS_PER_SEC;
	printf("\n�˳��������ʱ��Ϊ%f", totaltime);
	printf("\n�˳�������������ʱΪ%f", detectTime);
	printf("\n");
	cvWaitKey( 0 );
	cvReleaseImage( &xformed1 );
	cvReleaseMat( &H1 );
	/*cvReleaseImage(&xformed2);
	cvReleaseMat(&H2);*/
      }
  }
  

  cvReleaseImage( &stacked );
  cvReleaseImage( &img1 );
  cvReleaseImage( &img2 );
  kdtree_release( kd_root );
  free( feat1 );
  free( feat2 );
  return 0;
}
void psnr(IplImage * src, IplImage * dst)
{
	IplImage * src_gray = cvCreateImage(cvGetSize(src), src->depth, 1);
	IplImage * dst_gray = cvCreateImage(cvGetSize(src), src->depth, 1);
	cvCvtColor(src, src_gray, CV_RGB2GRAY);
	cvCvtColor(dst, dst_gray, CV_RGB2GRAY);
	IplImage * img_gray = cvCreateImage(cvGetSize(src_gray), src_gray->depth, 1);
	cvAbsDiff(src_gray, dst_gray, img_gray);
	CvScalar scalar;
	double sum = 0;
	for (int i = 0; i<img_gray->height; i++)
	{
		for (int j = 0; j<img_gray->width; j++)
		{
			scalar = cvGet2D(img_gray, i, j);
			sum += scalar.val[0] * scalar.val[0];//��ȡ�ľ���ÿһ�����ص�ĻҶ�ֵ
	
			//����srcͼ��BGR�е�Bͨ����ֵ
		}
	}
	double mse = 0;
	mse = sum / (img_gray->width * img_gray->height);
	if (mse == 0)
	{
		printf("���ƶ�100%");
		printf("\n");
	}
	else
	{
		double psnr = 0;
		psnr = 10 * log10(255*255 / mse);
		printf("\n\n����ͼ��֮���ֵ�����Ϊ��%f\n", psnr);
	}
}