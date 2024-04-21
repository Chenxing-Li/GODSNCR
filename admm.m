function result = admm(Y, BG, max_iter, epsilon, lambda, rho)
%ADMM 
[L,N] = size(Y);
[L,m] = size(BG);
Em = ones(m,1);
En = ones(N,1);

% initialization
Alpha = zeros(m,N);
Z = zeros(m,N);
W = zeros(m,N);
I = eye(m);
for i=1:max_iter
    Alpha_new = inv(2*lambda*I + BG.'*BG + Em*Em.' + rho*I) * (BG.'*Y + Em*En.' + rho*Z - W);
    if norm(Alpha_new - Alpha,'fro')/norm(Alpha_new,'fro') < epsilon
        Alpha = Alpha_new;
        disp("break")
        disp(i);
        break
    end
%     disp(1/2*norm(Em.'*Alpha-En.','fro')^2);
%     disp(1/2*norm(Y-BG*Alpha,'fro')^2);
%     disp(lambda*norm(Alpha,'fro')^2);

    Alpha = Alpha_new;
    Z = max(Alpha_new + W/rho, 0);
    W = W + rho*(Alpha_new - Z);

end
result = Alpha;
