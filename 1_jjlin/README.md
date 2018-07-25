步骤1：
preprocessing文件夹中
***使用sort.py排序，排序之后请删除第19,21,29,329张图，因为这几张图原图与污染图的通道数不同，无法计算psnr，本次上传文件已完成排序，请忽略sort.py文件
使用FT_mask.py提取mask
使用stack.py扩充图像

步骤2：
detect-cell-edge-use-unet-master文件夹中
使用to_npy.py文件
使用unet.py文件中的unet.train()训练,得到unet.hdf5模型
使用unet.py文件中的unet.test()预测
使用save_image.py文件得到网纹图
使用threshold二值化网纹图

步骤3：
deep-image-prior-master文件夹中
使用train.py生成复原图，完成后会生成train.log日志，其中显示pass的图片为生成合格的图片，不合格的图片请再次使用train.py重新生成复原图，直至pass为止

步骤4：
postprocessing文件夹中
使用unstack.py复原为250*250图片
使用change.py文件进一步优化，得到的最后复原图片在/image_painting/dataset/finally/final_output文件夹中
使用psnr文件计算出原始psnr和优化后的psnr，分别生成文件ori_psnr.csv和op_psnr.csv