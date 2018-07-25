import numpy
import csv
import math
from PIL import Image
import pdb
import pandas as pd

#计算原始psnr
record=[]
for i in range(1,1003):
    path1='/home/jjlin/Desktop/image_inpainting/dataset/test_pol/%i.jpg'%i
    path2='/home/jjlin/Desktop/image_inpainting/dataset/test/%i.jpg'%i
    im=Image.open(path1)
    im2=Image.open(path2)

    #pdb.set_trace()
    im = numpy.array (im,'f')#将图像1数据转换为float型
    im2 = numpy.array (im2,'f')#将图像2数据转换为float型

    height,width=im.shape[0],im.shape[1]#图像的行数,列数
    
    #提取R通道
    r = im[:,:,0]
    #提取g通道
    g = im[:,:,1]
    #提取b通道
    b = im[:,:,2]

    #图像1,2各自分量相减，然后做平方；
    R = im[:,:,0]-im2[:,:,0]
    G = im[:,:,1]-im2[:,:,1]
    B = im[:,:,2]-im2[:,:,2]
    #做平方
    mser = R*R
    mseg = G*G
    mseb = B*B
    #三个分量差的平方求和
    SUM = mser.sum() + mseg.sum() + mseb.sum()
    MSE = SUM / (height * width )
    PSNR = 10*math.log ( (255.0*255.0/(MSE)) ,10)

    record.append(PSNR)

result=pd.Series(index=range(1,1003),data=record)
result.to_csv('/home/jjlin/Desktop/image_inpainting/dataset/ori_psnr_record.csv')


#计算优化后的psnr
record2=[]
for i in range(1,1003):
    path1='/home/jjlin/Desktop/image_inpainting/dataset/finally/final_output/%i.jpg'%i
    path2='/home/jjlin/Desktop/image_inpainting/dataset/test/%i.jpg'%i
    im=Image.open(path1)
    im2=Image.open(path2)

    #pdb.set_trace()
    im = numpy.array (im,'f')#将图像1数据转换为float型

    im2 = numpy.array (im2,'f')#将图像2数据转换为float型

    height,width=im.shape[0],im.shape[1]#图像的行数,列数
    
    #提取R通道
    r = im[:,:,0]
    #提取g通道
    g = im[:,:,1]
    #提取b通道
    b = im[:,:,2]

    #图像1,2各自分量相减，然后做平方；
    R = im[:,:,0]-im2[:,:,0]
    G = im[:,:,1]-im2[:,:,1]
    B = im[:,:,2]-im2[:,:,2]
    #做平方
    mser = R*R
    mseg = G*G
    mseb = B*B
    #三个分量差的平方求和
    SUM = mser.sum() + mseg.sum() + mseb.sum()
    MSE = SUM / (height * width )
    PSNR = 10*math.log ( (255.0*255.0/(MSE)) ,10)

    record2.append(PSNR)

result=pd.Series(index=range(1,1003),data=record2)
result.to_csv('/home/jjlin/Desktop/image_inpainting/dataset/or_psnr_record.csv')





