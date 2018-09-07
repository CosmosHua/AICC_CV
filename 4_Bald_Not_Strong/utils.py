# coding:utf-8
#!/usr/bin/python3
""" Some codes from https://github.com/Newmu/dcgan_code """
# get_stddev = lambda x,h,w: (w*h*x.get_shape()[-1])**-0.5
# import pprint; pp = pprint.PrettyPrinter()
import os, cv2
import numpy as np

sz = [] # restore input images size for save_images
res = lambda im,ss: cv2.resize(im, ss, interpolation=cv2.INTER_LANCZOS4)


# Load single image from path: resize to size->concatenate
def load_image(path, size=(256,256), is_test=False): # path=(*.jpg)
    img_A = cv2.imread(path) # Load img_A = mesh image (input x)
    sz.append(img_A.shape[1::-1]) # (width,height), needn't global
    
    if is_test: img_B = img_A # For test: img_B is unnecessary
    else: # For train: Load img_B = clean image (label y)
        id = path.rfind('_'); img_B = cv2.imread(path[:id]+'.jpg') # clean
        # dir = path.split('_'); img_B = cv2.imread('_'.join(dir[:-1])+'.jpg')
        # print("Load :", path, path[:id]+'.jpg') # apply to the 1st Method
    
    #img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB) # im[:,:,::-1]
    #img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB) # im[:,:,::-1]
    
    img_A = res(img_A, size); img_B = res(img_B, size) # resize
    img_A = img_A/127.5-1; img_B = img_B/127.5-1 # normalize->center
    img_BA = np.concatenate((img_B, img_A), axis=2) # along channels
    return img_BA # clean+mesh: (size, img_B_channel+img_A_channel)

# Save images to path: resize or merge
def save_images(images, path, size): # to size
    im = (images+1.0)*127.5 # restore to [0,255]
    #im = im[:,:,:,::-1] # restore channels: RGB2BGR
    
    if len(im)<2: im = res(im[0], size) # resize single image
    else: im = Merge(im, (178,220)) # resize then merge images
    
    if path[-3:]=="png": cv2.imwrite(path, im, [cv2.IMWRITE_PNG_COMPRESSION,3])
    else: cv2.imwrite(path[:-4]+"_.jpg", im, [cv2.IMWRITE_JPEG_QUALITY,100])


# Join images after resize to uniform size
def Merge(IMs, size): # resize + merge
    N = len(IMs); w,h = size # size=(width,height)
    R = int(N**0.5); C = int(np.ceil(N/R)) # layout=(Row,Col)
    
    # Method 3: SUCCESS!
    pd = np.zeros((h, w, IMs.shape[-1])) # padding
    IMs = [res(im,size) for im in IMs] + [pd]*(R*C-N) # resize + pad
    IMs = [np.concatenate(IMs[i*C:i*C+C],axis=1) for i in range(R)] # widen
    return np.concatenate(IMs,axis=0) # heighten: join rows
    
    '''# Method 1: SUCCESS!
    img = np.zeros((R*h, C*w, IMs.shape[-1]))
    for id,im in enumerate(IMs): # (i,j)=(row,col)
        i,j = id//C, id%C;  img[i*h:i*h+h, j*w:j*w+w, :] = res(im,size)
    return img
    
    # Method 2: FAIL!
    pd = np.zeros((h, w, IMs.shape[-1])) # padding
    IMs = [res(im,size) for im in IMs] + [pd]*(R*C-N) # resize
    #IMs[N:] = [pd.copy() for i in range(R*C-N)] # unnecessary
    return np.array(IMs).reshape([R*h, C*w, -1]) # FAIL!'''


# Peak Signal to Noise Ratio
def PSNR_(I, K, ch=1, L=255):
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    # assert(I.shape == K.shape) # assert if False
    if I.shape!=K.shape: K = cv2.resize(K,I.shape[1::-1])

    IK = (I-K*1.0)**2; MAX = L**2; ee = MAX*1E-10
    if ch<2: MSE = np.mean(IK) # combine/average channels
    else: MSE = np.mean(IK,axis=(0,1)) # separate channels
    return 10 * np.log10(MAX/(MSE+ee)) # PSNR


# Structural Similarity (Index Metric)
def SSIM_(I, K, ch=1, k1=0.01, k2=0.03, L=255):
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    # assert(I.shape == K.shape) # assert if False
    if I.shape!=K.shape: K = cv2.resize(K,I.shape[1::-1])

    if ch<2: # combine/average channels->float
        mx, sx = np.mean(I), np.var(I,ddof=1)
        my, sy = np.mean(K), np.var(K,ddof=1)
        cov = np.sum((I-mx)*(K-my))/(I.size-1) # unbiased
        # cov = np.mean((I-mx)*(K-my)) # biased covariance
    else: # separate/individual/independent channels->np.array
        mx, sx = np.mean(I,axis=(0,1)), np.var(I,axis=(0,1),ddof=1)
        my, sy = np.mean(K,axis=(0,1)), np.var(K,axis=(0,1),ddof=1)
        cov = np.sum((I-mx)*(K-my),axis=(0,1))/(I.size/I.shape[-1]-1) # unbiased
        # cov = np.mean((I-mx)*(K-my),axis=(0,1)) # biased covariance
    
    c1, c2 = (k1*L)**2, (k2*L)**2 # stabilizer, avoid divisor=0
    SSIM = (2*mx*my+c1)/(mx**2+my**2+c1) * (2*cov+c2)/(sx+sy+c2)
    return SSIM # SSIM: separate or average channels


def BatchPS(test_dir, clean_dir="./Clean"):
    from glob import glob
    #mesh = glob(test_dir+"/*.jpg"); mesh.sort()
    recov = glob(test_dir+"/*.png"); recov.sort()
    clean = glob(clean_dir+"/*.jpg"); clean.sort()
    psnr = np.mean([PSNR_(i,k) for i,k in zip(clean,recov)])
    ssim = np.mean([SSIM_(i,k) for i,k in zip(clean,recov)])
    return np.array([psnr, ssim, psnr*ssim])

