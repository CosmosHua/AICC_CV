function R=image_repair(im,w,t)%w一般取10，t取10
imhsv=rgb2hsv(im);
im=double(im);
im1=im;
imv=imhsv(:,:,3);
imv=imv*255;

R=zeros(250);
for i=1:250
    for j=1:250
        R(i,j,1)=im1(i,j,1)/imv(i,j);
        R(i,j,2)=im1(i,j,2)/imv(i,j);
        R(i,j,3)=im1(i,j,3)/imv(i,j);
    end
end
R=R*255;        %颜色图
iml=zeros(250);
repair_map=zeros(250);

for i=1:250
  for j=1:250
 iml(i,j)=max(im(i,j,:));
  end
end
% repair_map=iml/255;%亮度比
iml=padarray(iml,[w,w], 'symmetric');

R1=padarray(R,[w,w], 'symmetric');
map=find_shadow2(R1,iml,w,t); %求疑似阴影区域
zj=2*w*w;
for i=w+1:250+w
    for j=w+1:250+w
        if(map(i,j)==1)
         p=[R1(i,j,1),R1(i,j,2),R1(i,j,3)];
         count=1;
         cup=0;
         for m=i-w:i+w
             for n=j-w:j+w
                 q=[R1(m,n,1),R1(m,n,2),R1(m,n,3)];
                
                o=p-q;
                ao=norm(o);
                if(ao<t)
                cup(count,1)=iml(m,n);
               zj1=(m-i)*(m-i)+(n-j)*(n-j);
                zj2=zj-zj1;
                cup(count,2)=zj2*zj2*zj2;
                count=count+1;
                end
             end
         end
         th=mean(cup(:,1));
         sum=0;
        
          sumw=0;
        for k=1:count-1
         if(cup(k,1)>=th)
             sumw=sumw+cup(k,2);
         end
        end
     for k=1:count-1
         if(cup(k,1)>=th)
            c=cup(k,2)/sumw;
            sum=sum+cup(k,1)*c;
         end
     end
             
            
        
     repair_map(i-w,j-w)=sum;
        else
            repair_map(i-w,j-w)=iml(i,j);
        end
     
    end
end
repair_map=repair_map/255;
for i=1:250
    for j=1:250
        R(i,j,:)=R(i,j,:).*repair_map(i,j);
    end
end
R=uint8(R);

end

           
       