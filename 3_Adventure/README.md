DeMesh:

1. �������л�����

	python 3.5

	tensorflow 1.4.0
	
	matplotlib 2.0.2

	h5py 2.7.1

	xlwt 1.3.0

	skimage 0.13.0

2. �ļ�˵��

	data �����ļ���

		test ���Լ��洢λ�ã�ԭͼ + ����ͼ��

		train ѵ�����洢λ�ã�h5data��input��label��map��

	model ģ�ʹ洢λ��

	result ���Խ��ͼƬλ��

	generate_data.py ����ѵ������

	testing.py ���Դ����ļ�

	training.py ѵ�������ļ�

	utils.py ���������ļ���h5�ļ���ȡ�����ֽ�����ɡ������˲���map�ļ����ɡ��ļ��ƶ��ȣ�

3. ����˵��

	ѵ����ִ��training.py�������data/h5data�ļ����¶�ȡ����ѵ�������model�ļ��а�����ģ�ͣ����ȡ����������ϼ���ѵ������û��ģ��������ģ�͡�

	���ԣ�ִ��testing.py,�����data/test��ȡ����ͼƬ�����ɴ�����ͼƬ����result�ļ��У����ֽ�������ļ�result.xls