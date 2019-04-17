#!/usr/bin/env.python
#encoding=utf-8


import numpy as np
import cv2
import math
import time
import logging
orig_image=cv2.imread("boat1.bmp",1)
skewed_image=cv2.imread("boat2.bmp",1)


cv2.imshow("graf1",orig_image)
cv2.imshow("graf2",skewed_image)

# img=cv2.imread("INTER_LINEAR.bmp",-1)
# height,width=img.shape[:2]
# size=(int(width*0.1),int(height*0.1))
# shrink = cv2.resize(img,(9,8), interpolation=cv2.INTER_AREA)
# cv2.imwrite("mysample2.bmp",shrink)


def dhash(img, hash_size=8):
    '''
    get image dhash string
    '''
    #img = plt.imread(image_path)  # 转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resize_img = cv2.resize(gray_img, (hash_size+1, hash_size))
    #filename="{}_{}".format(time.strftime('%d-%m_%I_%M'),".jpg")
    filename = "{}_{}".format(time.time(), ".jpg") #以时间戳保存文件名
    cv2.imwrite(filename,resize_img)
    # 计算水平梯度
    differences = []
    for t in range(resize_img.shape[1]-1):
        differences.append(resize_img[:, t] > resize_img[:, t+1])
    img_ = np.stack(differences).T
    # 二值化
    img_bi = ''.join(img_.astype('B').flatten().astype('U').tolist())
    # 切割，每4个字符一组，转成16进制字符
    return ''.join(map(lambda x: '%x' % int(img_bi[x:x+4], 2), range(0, 64, 4)))
#https://www.jianshu.com/p/193f0089b7a2
dhash1=dhash(orig_image,8)
dhash2=dhash(skewed_image,8)
def str_to_bin(s):
    return ' '.join([bin(ord(c)).replace('0b', '') for c in s]) #这个函数不用自己定义
#dhash1B=np.array(dhash1,dtype='B')
dhash1B=str_to_bin(dhash1) #将16进制字符串转换为二进制字符串
dhash2B=str_to_bin(dhash2)


def hash_haming(hash_img1,hash_img2):
    '''
    计算两张通过哈希感知算法编码的图片的汉明距离
    '''
    return np.array([hash_img1[x] != hash_img2[x] for x in range(16)], dtype='B').sum()
        #64bit,转换成16进制是16位，所以哈希值长度是16

hanming=hash_haming(dhash1,dhash2)
print("汉明距离:",hanming)

#定义函数
def mydrawMatches(orig_image, kp1, skewed_image, kp2, matches):
    rows1=orig_image.shape[0]#height(rows) of image
    cols1=orig_image.shape[1]#width(colums) of image
    #shape[2]#the pixels value is made up of three primary colors
    rows2=skewed_image.shape[0]
    cols2=skewed_image.shape[1]

#初始化输出的新图像，将两幅实验图像拼接在一起，便于画出特征点匹配对
    out=np.zeros((max([rows1,rows2]),cols1+cols2,3),dtype='uint8')

    out[:rows1,:cols1] = np.dstack([orig_image, orig_image, orig_image])#Python切片特性，初始化out中orig_image，skewed_image的部分
    out[:rows2,cols1:] = np.dstack([skewed_image, skewed_image, skewed_image])#dstack,对array的depth方向加深

    for mat in matches:
        orig_image_idx=mat.queryIdx
        skewed_image_idx=mat.trainIdx

        (x1,y1)=kp1[orig_image_idx].pt
        (x2,y2)=kp2[skewed_image_idx].pt

        cv2.circle(out,(int(x1),int(y1)),4,(255,255,0),1)#蓝绿色点，半径是4
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (0, 255, 255), 1)#绿加红得黄色点
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)#蓝色连线

    return out

def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")#想解决imshow中窗口中文标题乱码的问题


cv2.waitKey(0)