x = xlsread('w4.xlsx', 'sheet1', 'A1:B30');
[m,n]=size(x);
%当前最低的平方误差，初始值设为一个很大的数
old_ts=100;
%对K值遍历，至少2类
for k=2:10
    %随机均值
    u=x(randperm(m,k),:);
    while 1
        %将各类集合清空
        c=zeros(k,30);
        nums=zeros(k,1);
        %对所有样本遍历，选择最近的集合
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
        %计算两次均值差异，并更新均值
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
    %计算当前的均方误差
    ts=0;
    for i=1:k
         for j=1:nums(i)
            ts=ts+norm(x(c(i,j),:)-u(i,:))^2; 
         end
         %惩罚项
         ts=ts-(nums(i)/m)*log(nums(i)/m)*0.5;
    end
    %如果ts比前一轮大则停止,否则更新
    if(ts<old_ts)
        old_ts=ts;
        old_c=c;
        old_nums=nums;
    else
        break;
    end
end
ch='o*+>.';
%取前一轮的k为最佳的k值
nums=old_nums;
c=old_c;
k=k-1;
%绘制凸包与点
for i=1:k
   plot(x(c(i,1:nums(i)),1),x(c(i,1:nums(i)),2),ch(i));
   hold on;
   tc=x(c(i,1:nums(i)),:);
   chl=convhull(tc);
   line(tc(chl,1),tc(chl,2))
   hold on;
end

xlabel('密度');
ylabel('含糖率');
title('K-means');
