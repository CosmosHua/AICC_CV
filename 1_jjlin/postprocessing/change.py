from numpy import *
from skimage import io
import pdb
import cv2

for i in range(1,1003):
	mask_path='/home/jjlin/Desktop/image_inpainting/dataset/finally/mask/%i.jpg'%i
	pol_path='/home/jjlin/Desktop/image_inpainting/dataset/test_pol/%i.jpg'%i
	recover_path='/home/jjlin/Desktop/image_inpainting/dataset/finally/recover_images/%i.jpg'%i
	save_path='/home/jjlin/Desktop/image_inpainting/dataset/finally/final_output/%i.bmp'%i
	
	mask=io.imread(mask_path,as_grey=True)
	pol=cv2.imread(pol_path)
	recover=cv2.imread(recover_path)

	rp,gp,bp=cv2.split(pol)
	rr,gr,br=cv2.split(recover)

	rp=rp.flatten()
	gp=gp.flatten()
	bp=bp.flatten()

	rr=rr.flatten()
	gr=gr.flatten()
	br=br.flatten()

	mask=mask.flatten()

	#pdb.set_trace()
	for i in range(250*250):
		if mask[i]>0.5:
			rr[i]=rp[i]
			gr[i]=gp[i]
			br[i]=bp[i]
	rr=rr.reshape(250,250)
	gr=gr.reshape(250,250)
	br=br.reshape(250,250)
	recover=cv2.merge([rr,gr,br])
	cv2.imwrite(save_path,recover)
	
