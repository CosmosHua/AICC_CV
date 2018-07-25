import glob
import os
import shutil


def get_filepaths(path):
	list_images=glob.glob('%s/*.jpg'%path)#文件夹下的所有文件的完整路径
	return list_images


def rename_filenames(path,new_path,count):
	filelist=os.listdir(path)#文件夹下的所有文件的文件名
	
	filelist.sort()#in place
	for files in filelist:
		Olddir=os.path.join(path,files)#完整路径
		if os.path.isdir(Olddir):#为文件夹时则跳过
			continue
		filename=os.path.splitext(files)[0]#文件名
		if '_' in filename:
			filename.strip('_')
		filetype=os.path.splitext(files)[1]#扩展名
		Newdir=os.path.join(new_path,str(count)+filetype)
		#os.rename(Olddir,Newdir)
		shutil.copy(Olddir,Newdir)
		count+=1

ori_path  = '/home/jjlin/Desktop/train_ori'
pol_path  = '/home/jjlin/Desktop/train_pol'

#分类文件

#shutil.copytree(data_path,ori_path)
#shutil.copytree(data_path,pol_path)


new_path='/home/jjlin/Desktop/new_train_pol'
num=1
path=os.listdir(pol_path)
path.sort()
for i in path:
    filepaths=os.path.join(pol_path,i)
    #com_path=get_filepaths(filepaths)
    rename_filenames(filepaths,new_path,num)
    count=len(os.listdir(filepaths))
    num+=count
















