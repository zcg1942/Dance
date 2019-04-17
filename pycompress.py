#!/usr/bin/env.python
# -*- coding:utf-8 -#-
_author_ = 'M&G'
import sys
#reload(sys)

###################导入计算机视觉库opencv和图像处理库PIL####################
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import cv2
import time
import zlib
import numpy as np
time1 = time.time()
src1=cv2.imread("graf1.ppm")
src2=cv2.imread("graf1.ppm")



fs=cv2.FileStorage('vocabulary.xml',cv2.FileStorage_READ)

####################读入图像###############################
# image=cv2.imread("E:\Local Repositories/NiuKe\SubImages\SubImages/result/graSubImageJPG00.jpg")
# ####################双三次插值#############################
# res = cv2.resize(image, (800,640), interpolation=cv2.INTER_AREA)
# ####################写入图像########################
# cv2.imwrite("E:\Local Repositories/NiuKe\SubImages\SubImages/result/graSubImageJPG000.jpg",image)
###########################图像对比度增强##################
# imgE = Image.open("E:\Local Repositories/NiuKe\SubImages\SubImages/result/graf1.jpg")
# imgEH = ImageEnhance.Contrast(imgE)
# img1=imgEH.enhance(2.8)
# ########################图像转换为灰度图###############
# gray = img1.convert("L")
# gray.save("E:\Local Repositories/NiuKe\SubImages\SubImages/result/grafgray.jpg")
# ##########################图像增强###########################
#
# # 创建滤波器，使用不同的卷积核
# gary2=gray.filter(ImageFilter.DETAIL)
# gary2.save("E:\Local Repositories/NiuKe\SubImages\SubImages/result/graffilter.jpg")
#
# #############################图像点运算#################
# gary3=gary2.point(lambda i:i*0.9)
# gary3.save("E:\Local Repositories/NiuKe\SubImages\SubImages/result/grafdian.jpg")
# # img1.show("new_picture")
# time2=time.time()
# print(u'总共耗时：' + str(time2 - time1) + 's')
#https://blog.csdn.net/u013421629/article/details/76034225

#读入图像
gra1=Image.open("bikeimg1to2.bmp")
gra2=Image.open("bike3.bmp")

#把图像转换成字节流并计算其长度
graByte1=gra1.tobytes()
graByte2=gra2.tobytes()
print("length of subimage:",len(graByte1))
print("length of subimage:",len(graByte2))

#把图像转换成数组
img_array1 = np.asarray(gra1)
print("输出图像数组：",img_array1)
img_array2 = np.asarray(gra2)

#利用数组的减法实现图像之间的减法
#https://blog.csdn.net/wsp_1138886114/article/details/82801792
img_array3=img_array1- img_array2
print("输出差值图像数组：",img_array3)

#对减法得到的差值进行压缩
compressedBytes=zlib.compress(img_array3)
gracplen=len(compressedBytes)
print('length of compressed imnage:',gracplen)

#解压并转化成字节流，对比验证是否无损压缩
img1=Image.frombytes('RGB',gra1.size,zlib.decompress(compressedBytes))
if(img1.tobytes() == img_array3.tobytes()):
    print("True")

#对原图的压缩
compressedSubBytes=zlib.compress(graByte2)
graSubcplen=len(compressedSubBytes)
print('length of compressed Subimnage:',graSubcplen)

#验证对于原图的压缩是无损压缩
img2=Image.frombytes('RGB',gra2.size,zlib.decompress(compressedSubBytes))
if(img2.tobytes() == graByte2):
  print("True")

