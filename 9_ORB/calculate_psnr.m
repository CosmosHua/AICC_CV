
function [P,R]=calculate_psnr(fname) % ��������fnameΪ���޸�ͼƬ�ļ��еľݶ�·������'E:\AI��ս��\AI_CV_Test_1ѵ�����ݼ�\1.jpg'
                                     % ����ֵPΪ���������          
                                     % R Ϊ�޸��õ�ͼƬ

  
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


    
   