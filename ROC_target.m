function [pd,pf]=ROC_target(results,XY,Outer_k)
%%h*w
[h,w,Band]=size(results);
n=h*w;
results=real(results);

Outer_flag=fix(Outer_k/2);
N=(h-Outer_flag*2)*(w-Outer_flag*2);

%真实目标点位XY（i,j）
[point,q]=size(XY);
%%真实目标点位对应的探测统计值，并排序
XYRst=zeros(point,Band);
for i=1:point
   XYRst(i,:)=results( XY(i,1),XY(i,2),:); 
end
[Result]=sort(XYRst,'descend'); %%对每一列进行排序

%%以真实目标点位对应的探测统计值中的某一个值为阈值，对结果图像进行阈值分割，记录探测出来的目标点位及个数
num=1:1:point;
pd=zeros(point,Band);
pf=zeros(point,Band);

for l=1:Band
    for k=1:point
            xy=zeros(h*w,2);
            mark=0;
            for i=Outer_flag+1 : h-Outer_flag%注意边界的像素不算在内----------------------------。
                for j=Outer_flag+1 : w-Outer_flag                 
                    if ( results(i,j,l)>=Result(num(k),l) )
                        mark=mark+1;
                        xy(mark,1)=i; 
                        xy(mark,2)=j; 
                    end
                end
            end

            %%探测出来的真实目标点个数及探测出来的错误目标点个数（虚假个数）
            pd_cnt=0;
            for i=1:mark
                for j=1:point
                    if(xy(i,1)==XY(j,1)&&xy(i,2)==XY(j,2))
                     pd_cnt=pd_cnt+1;
                     break;
                    end
                 end
            end
            pf_cnt=mark-pd_cnt;
            pd(k,l)=1.0*pd_cnt/point; %%探测率
            pf(k,l)=pf_cnt/N;     %%虚警率
    end
end