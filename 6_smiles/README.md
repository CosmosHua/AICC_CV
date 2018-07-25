srresnet：

将训练图片按照/train/input和/train/target划分，input为网纹图片，target为无网纹图片。
网纹图片命名为x_.jpg，例如0000114_232_.jpg，而对应的无网纹图片命名为x.jpg，例如0000114_232.jpg
input和target可在dataset.py中修改
train可在train.py中的DatasetFromFolder中修改

根路径为data_dir

对于test，其目录和文件命名和train一致，均可在test.py中修改

face_model中存放的为训练模型