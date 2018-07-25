import cv2
import numpy as np
import math
import pdb
#傅里叶提取mask

def make_mask(img_path,img_pol_path,mask_path):
    img=cv2.imread(img_path)
    img_pol=cv2.imread(img_pol_path)

    #pdb.set_trace()
    (b0,g0,r0)=cv2.split(img)#类型是np.array
    #类型转换
    float_b0=b0.astype(np.float64)
    float_g0=g0.astype(np.float64)
    float_r0=r0.astype(np.float64)
    #FT
    b0_ft=np.fft.fftshift(float_b0)
    g0_ft=np.fft.fftshift(float_g0)
    r0_ft=np.fft.fftshift(float_r0)
    #求绝对值
    lb0_ft=np.abs(b0_ft)
    lg0_ft=np.abs(g0_ft)
    lr0_ft=np.abs(r0_ft)
    #
    lb0_ft=np.log(b0_ft+1)
    lg0_ft=np.log(g0_ft+1)
    lr0_ft=np.log(r0_ft+1)

    (b1,g1,r1)=cv2.split(img_pol)
    #类型转换
    float_b1=b1.astype(np.float64)
    float_g1=g1.astype(np.float64)
    float_r1=r1.astype(np.float64)
    #FT
    b1_ft=np.fft.fftshift(float_b1)
    g1_ft=np.fft.fftshift(float_g1)
    r1_ft=np.fft.fftshift(float_r1)
    #求绝对值
    lb1_ft=np.abs(b1_ft)
    lg1_ft=np.abs(g1_ft)
    lr1_ft=np.abs(r1_ft)
    #
    lb1_ft=np.log(b1_ft+1)
    lg1_ft=np.log(g1_ft+1)
    lr1_ft=np.log(r1_ft+1)

    #相减
    deta_b=b1_ft-b0_ft
    deta_g=g1_ft-g0_ft
    deta_r=r1_ft-r0_ft

    #IFT
    b=np.abs(np.fft.ifftshift(deta_b))
    g=np.abs(np.fft.ifftshift(deta_g))
    r=np.abs(np.fft.ifftshift(deta_r))


    ret, b = cv2.threshold(b, 12,255,cv2.THRESH_BINARY)
    ret, g = cv2.threshold(g, 12, 255,cv2.THRESH_BINARY)
    ret, r = cv2.threshold(r, 12, 255,cv2.THRESH_BINARY)
    

    mask=b+g+r
    #遍历置为255
    mask[mask>254]=255
    mask=np.where(mask>1,0,255)
    #print(mask)

    cv2.imwrite(mask_path,mask)


#train and train_pol to train_mask
for i in range(1,10014):
	img_path='/home/jjlin/Desktop/image_inpainting/dataset/train/%i.jpg'%i
	img_pol_path='/home/jjlin/Desktop/image_inpainting/dataset/train_pol/%i.jpg'%i
	mask_path='/home/jjlin/Desktop/image_inpainting/dataset/train_mask/%i.jpg'%i
	make_mask(img_path,img_pol_path,mask_path)















