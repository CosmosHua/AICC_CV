function map=find_shadow2(R1,iml,w,t)
map=zeros(size(iml));
for i=w+1:250+w
    for j=w+1:250+w
         p=[R1(i,j,1),R1(i,j,2),R1(i,j,3)];
         count=1;
         cup=0;
         for m=i-w:i+w
             for n=j-w:j+w
                 q=[R1(m,n,1),R1(m,n,2),R1(m,n,3)];
                
                o=p-q;
                ao=norm(o);
                if(ao<t)
                    cup(count)=iml(m,n);
                    count=count+1;
                end
             end
         end
             th=mean(cup(:));
             td=std(cup(:));
             if(iml(i,j)<th-0.3*td)
                 map(i,j)=1;
             end
    end
end
map=bwareaopen(map,60);

end
                 