x = xlsread('w4.xlsx', 'sheet1', 'A1:B30');
[m,n]=size(x);
%��ǰ��͵�ƽ������ʼֵ��Ϊһ���ܴ����
old_ts=100;
%��Kֵ����������2��
for k=2:10
    %�����ֵ
    u=x(randperm(m,k),:);
    while 1
        %�����༯�����
        c=zeros(k,30);
        nums=zeros(k,1);
        %����������������ѡ������ļ���
        for i=1:m
           mind=100000;
           minl=0;
           for j=1:k
              d=norm(x(i,:)-u(j,:));
              if(d<mind)
                 mind=d;
                 minl=j;
              end
           end
           nums(minl)=nums(minl)+1;
           c(minl,nums(minl))=i;
        end   
        %�������ξ�ֵ���죬�����¾�ֵ
        ut=zeros(k,2);
        for i=1:k
           for j=1:nums(i)
               ut(i,:)=ut(i,:)+x(c(i,j),:);
           end
           ut(i,:)=ut(i,:)/nums(i);
        end

        du=norm(ut-u);
        if(du<0.001)
            break;
        else
            u=ut;
        end
    end
    %���㵱ǰ�ľ������
    ts=0;
    for i=1:k
         for j=1:nums(i)
            ts=ts+norm(x(c(i,j),:)-u(i,:))^2; 
         end
         %�ͷ���
         ts=ts-(nums(i)/m)*log(nums(i)/m)*0.5;
    end
    %���ts��ǰһ�ִ���ֹͣ,�������
    if(ts<old_ts)
        old_ts=ts;
        old_c=c;
        old_nums=nums;
    else
        break;
    end
end
ch='o*+>.';
%ȡǰһ�ֵ�kΪ��ѵ�kֵ
nums=old_nums;
c=old_c;
k=k-1;
%����͹�����
for i=1:k
   plot(x(c(i,1:nums(i)),1),x(c(i,1:nums(i)),2),ch(i));
   hold on;
   tc=x(c(i,1:nums(i)),:);
   chl=convhull(tc);
   line(tc(chl,1),tc(chl,2))
   hold on;
end

xlabel('�ܶ�');
ylabel('������');
title('K-means');
