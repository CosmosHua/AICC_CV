
function [P,R]=calculate_psnr(fname) % 主函数，fname为待修复图片文件夹的据对路径，如'E:\AI挑战赛\AI_CV_Test_1训练数据集\1.jpg'
                                     % 返回值P为信噪比增益          
                                     % R 为修复好的图片

  
   y=imread(fname);
   
   x=imread(fname);
    R=image_repair(x,10,15);
     if(numel(size(x))==3)
        x1=rgb2gray(x);
    end
    if(numel(size(y))==3)
        y1=rgb2gray(y);
    end
    R1=rgb2gray(R);
    P1=psnr(x1,y1);
    P2=psnr(R1,y1);
     P=abs(P2-P1)/P1;

end


    
   