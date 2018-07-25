# -*- coding: utf-8 -*-
import shutil
import os
import xlwt
import glob
import cv2
import h5py
import numpy as np
from skimage.measure import compare_psnr

train_path = './AI_Train_1/'
test_path = './data/test/'
res_path = './result/'

input_path = './data/train/input/'
label_path = './data/train/label/'
map_path = './data/train/map/'


def move_file():
	num = 0
	for root, dirs, files in os.walk(train_path):
		label_files = glob.glob(root + '\\*_.jpg')
		for file in label_files:
			num += 1
			shutil.copy2(file, os.path.join(input_path, str(num) + '.jpg'))
			shutil.copy2(file.split('_.')[0] + '.jpg', os.path.join(label_path, str(num) + '.jpg'))
			# print(os.path.join(root, file))


def get_map():
	files = os.listdir(input_path)
	print(files)
	for file in files:
		im1 = cv2.imread(os.path.join(input_path, file), 0)
		im2 = cv2.imread(os.path.join(label_path, file), 0)
		sub = im1 - im2
		_, subbin = cv2.threshold(sub, 40, 255, cv2.THRESH_BINARY)
		img_medianBlur = cv2.medianBlur(subbin, 5)
		# cv2.imwrite(map_path + file, img_medianBlur)
		cv2.imshow('im_blur', img_medianBlur)
		cv2.waitKey(0)
		break

# read h5 files
def read_data(file):
	with h5py.File(file, 'r') as hf:
		data = hf.get('data')
		label = hf.get('label')
		map = hf.get('map')
		return np.array(data), np.array(label), np.array(map)


# guided filter
def guided_filter(data, num_patches, width=64, height=64, channel=3):
	r = 15
	eps = 1.0
	batch_q = np.zeros((num_patches, height, width, channel))
	for i in range(num_patches):
		for j in range(channel):
			I = data[i, :, :, j]
			p = data[i, :, :, j]
			ones_array = np.ones([height, width])
			N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
			mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
			mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
			mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
			cov_Ip = mean_Ip - mean_I * mean_p
			mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
			var_I = mean_II - mean_I * mean_I
			a = cov_Ip / (var_I + eps)
			b = mean_p - a * mean_I
			mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
			mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
			q = mean_a * I + mean_b
			batch_q[i, :, :, j] = q
	return batch_q


def gray2rgb(im_gray):
	temp = np.zeros((im_gray.shape[0], im_gray.shape[1], 3))
	temp[:, :, 0] = im_gray
	temp[:, :, 1] = im_gray
	temp[:, :, 2] = im_gray
	return temp


def getres(test_path, res_path, out_path):
	files = os.listdir(test_path)
	test_files = glob.glob(test_path + '\\*_.jpg')
	for x in test_files:
		files.remove(x.split('\\')[-1])

	# 创建一个workbook 设置编码
	workbook = xlwt.Workbook(encoding = 'utf-8')
	# 创建一个worksheet
	worksheet = workbook.add_sheet('sheet1')

	# 写入excel
	# 参数对应 行, 列, 值
	# worksheet.write(1,0, label = 'this is test')
	worksheet.write_merge(0, 0, 1, 4, '测试样本总计1000组') # Merges row 0's columns 1 through 4.
	worksheet.write_merge(1, 1, 1, 4, '') # Merges row 0's columns 1 through 4.
	worksheet.write(2, 1, label='图片编号')
	worksheet.write(2, 2, '网纹图PSNR\n(已给出)\r\n')
	worksheet.write(2, 3, '去网纹图PSNR\n(参赛者填入\r\n)')
	worksheet.write(2, 4, 'PSNR增益率\n（自动计算得出）\r\n')
	worksheet.write(2, 6, 'PSNR增益率\n（自动计算得出）\r\n')

	last = 0
	for i, file in enumerate(files):
		im1 = cv2.imread(os.path.join(test_path, file))
		im2 = cv2.imread(os.path.join(test_path, file.split('.')[0] + '_.jpg'))
		im3 = cv2.imread(os.path.join(res_path, file.split('.')[0] + '_.jpg'))

		worksheet.write(3 + i, 1, file)
		worksheet.write(3 + i, 2, compare_psnr(im1, im2))
		worksheet.write(3 + i, 3, compare_psnr(im1, im3))
		worksheet.write(3 + i, 4, xlwt.Formula('(D{}-C{}) / C{}'.format(4 + i, 4 + i, 4 + i)))
		last = i

	worksheet.write(2, 7, xlwt.Formula('AVERAGE(E4:E{})'.format(4 + last)))
	# 保存
	workbook.save(out_path)


def merge(path1='result3/', path2='res_final/'):
	files = os.listdir(test_path)
	test_files = glob.glob(test_path + '\\*_.jpg')
	for x in test_files:
		files.remove(x.split('\\')[-1])

	for i, file in enumerate(files):
		im = cv2.imread(os.path.join(test_path, file))
		im3 = cv2.imread(os.path.join(path1, file.split('.')[0] + '_.jpg'))
		im4 = cv2.imread(os.path.join(path2, file.split('.')[0] + '_.jpg'))

		if compare_psnr(im, im3) > compare_psnr(im, im4):
			shutil.copy2(os.path.join(path1, file.split('.')[0] + '_.jpg'), 'res_final2/' + file.split('.')[0] + '_.jpg')
		else:
			shutil.copy2(os.path.join(path2, file.split('.')[0] + '_.jpg'), 'res_final2/' + file.split('.')[0] + '_.jpg')

	getres(test_path, res_path='res_final2/', out_path='res53.xls')


def select_small():
	files = os.listdir(test_path)
	test_files = glob.glob(test_path + '\\*_.jpg')
	for x in test_files:
		files.remove(x.split('\\')[-1])

	for i, file in enumerate(files):
		im = cv2.imread(os.path.join(test_path, file))
		im2 = cv2.imread(os.path.join(test_path, file.split('.')[0] + '_.jpg'))
		im3 = cv2.imread(os.path.join('res_final/', file.split('.')[0] + '_.jpg'))
		t = compare_psnr(im, im2)
		if (compare_psnr(im, im3) - t) / t < 0.5:
			shutil.copy2(os.path.join('res_final/', file.split('.')[0] + '_.jpg'), 'temp/' + file.split('.')[0] + '_.jpg')


if __name__ == '__main__':
	# get_map()
	# getres(test_path, 'res_final/', 'res35.csv')
	get_map()
# 	select_small()