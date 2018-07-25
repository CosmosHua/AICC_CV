一共三个文件，保存三个文件在MATLAB工作目录后，在MATLAB命令窗口输入：

fname='*********';            %星号处是待填内容，应为要测试图片的绝对路径（如'E:\AI挑战赛\AI_CV_Test_1训练数据集\0000484001.jpg'）
[P,R]=calculate_psnr(fname);  % 返回值P为信噪比增益 ,R 为修复好的图片         

disp(p);
figure,imshow(R);
