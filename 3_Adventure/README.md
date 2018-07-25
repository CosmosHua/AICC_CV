DeMesh:

1. 代码运行环境：

	python 3.5

	tensorflow 1.4.0
	
	matplotlib 2.0.2

	h5py 2.7.1

	xlwt 1.3.0

	skimage 0.13.0

2. 文件说明

	data 数据文件夹

		test 测试集存储位置（原图 + 网纹图）

		train 训练集存储位置（h5data、input、label、map）

	model 模型存储位置

	result 测试结果图片位置

	generate_data.py 生成训练数据

	testing.py 测试代码文件

	training.py 训练代码文件

	utils.py 基本工具文件（h5文件读取、评分结果生成、引导滤波、map文件生成、文件移动等）

3. 运行说明

	训练：执行training.py，代码从data/h5data文件夹下读取数据训练，如果model文件夹包含有模型，则读取后在其基础上继续训练；若没有模型则生成模型。

	测试：执行testing.py,代码从data/test读取网纹图片，生成处理后的图片存于result文件夹，评分结果存在文件result.xls