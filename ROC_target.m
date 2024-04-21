function [pd,pf]=ROC_target(results,XY,Outer_k)
%%h*w
[h,w,Band]=size(results);
n=h*w;
results=real(results);

Outer_flag=fix(Outer_k/2);
N=(h-Outer_flag*2)*(w-Outer_flag*2);

%��ʵĿ���λXY��i,j��
[point,q]=size(XY);
%%��ʵĿ���λ��Ӧ��̽��ͳ��ֵ��������
XYRst=zeros(point,Band);
for i=1:point
   XYRst(i,:)=results( XY(i,1),XY(i,2),:); 
end
[Result]=sort(XYRst,'descend'); %%��ÿһ�н�������

%%����ʵĿ���λ��Ӧ��̽��ͳ��ֵ�е�ĳһ��ֵΪ��ֵ���Խ��ͼ�������ֵ�ָ��¼̽�������Ŀ���λ������
num=1:1:point;
pd=zeros(point,Band);
pf=zeros(point,Band);

for l=1:Band
    for k=1:point
            xy=zeros(h*w,2);
            mark=0;
            for i=Outer_flag+1 : h-Outer_flag%ע��߽�����ز�������----------------------------��
                for j=Outer_flag+1 : w-Outer_flag                 
                    if ( results(i,j,l)>=Result(num(k),l) )
                        mark=mark+1;
                        xy(mark,1)=i; 
                        xy(mark,2)=j; 
                    end
                end
            end

            %%̽���������ʵĿ��������̽������Ĵ���Ŀ����������ٸ�����
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
            pd(k,l)=1.0*pd_cnt/point; %%̽����
            pf(k,l)=pf_cnt/N;     %%�龯��
    end
end