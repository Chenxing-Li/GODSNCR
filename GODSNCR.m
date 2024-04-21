[h,w,Band] = size(X);

num = 450

filename = "./Data/AVIRIS_6_10/BGAVIRIS";

filename1 = filename + num2str(num) + ".mat";
filename2 = filename + num2str(num) + "_purify.mat";
load(filename1);
load(filename2);

max_iter = 100;
epsilon = 1e-2;
lambda=[0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 0.9];
rho=[0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5];

AUC_Matrix=zeros(length(lambda),length(rho));
Y = reshape(X,h*w,Band)';    
sparsity = 10;

A=[BG S];

%parameter analysis
for i=1:length(lambda)
    for j=1:length(rho)
        lambda_iter=lambda(i);
        rho_iter=rho(j);

        Alpha = admm(Y, BG_purify, max_iter, epsilon, lambda_iter, rho_iter);
        result = zeros(size(Alpha,2),1);
        for k=1:size(Alpha,2)
            x = Y(:, k);
            [gama,~] = SparseRepresentationSolve_OMP(A,x,'maxiteration',sparsity);
            rt = norm(x-A*gama)^2;
        
            rb = norm(x - BG_purify*Alpha(:, k))^2;
            result(k) = rb - rt;
        end
        [h,w,Band] = size(X);
        result = hyperConvert3d(result,h,w,Band);
        
        
        [pd,pf]=ROC_target(result,XY,0);
        pd=[0;pd;1];
        pf=[0;pf;1];
        y=pd';x=pf';
        AUC_Matrix(i,j) = trapz(x,y)
    end
end